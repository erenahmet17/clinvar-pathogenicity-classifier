import numpy as np
import pandas as pd
import pytest

from clinvar_classifier.annotation import parse_annotation
from clinvar_classifier.data import add_identities, deduplicate, map_label, merge_annotations
from clinvar_classifier.features import make_features, score_direction
from clinvar_classifier.schemas import ValidationError, normalize_hgvs
from clinvar_classifier.split import assert_disjoint, gene_groups, make_splits


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Pathogenic", 1),
        ("Likely pathogenic", 1),
        (" pathogenic / likely pathogenic ", 1),
        ("Benign", 0),
        ("Likely benign", 0),
        ("Benign/Likely benign", 0),
        ("Uncertain significance", None),
        ("Pathogenic, low penetrance", None),
        ("Pathogenic; risk factor", None),
        ("Conflicting classifications of pathogenicity", None),
        (None, None),
    ],
)
def test_explicit_label_policy(text, expected):
    assert map_label(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "chr1:g.0A>G",
        "chr1:g.1A>A",
        "chr23:g.1A>G",
        "NM_000410.4:c.845G>A",
        "chr1:g.1del",
        "chr1:g.1N>G",
        "chr1:g.-1A>G",
    ],
)
def test_unsupported_hgvs_rejected(text):
    with pytest.raises(ValidationError):
        normalize_hgvs(text)


def test_trim_valid_genomic_identity():
    assert normalize_hgvs(" chr6:g.26092913G>A ") == "chr6:g.26092913G>A"


def test_missing_af_and_zero_have_distinct_features():
    frame = pd.DataFrame(
        {
            "cadd_phred": [20.0, 20.0],
            "phylop_100way": [2.0, 2.0],
            "gnomad_af": [None, 0.0],
            "af_available": [False, True],
        }
    )
    features = make_features(frame)
    assert np.isnan(features.loc[0, "gnomad_af_log"])
    assert features.loc[0, "gnomad_af_missing"] == 1
    assert features.loc[1, "gnomad_af_log"] == -9
    assert features.loc[1, "gnomad_af_missing"] == 0


@pytest.mark.parametrize(
    "column,value",
    [("cadd_phred", np.inf), ("gnomad_af", -1), ("gnomad_af", 1.1), ("phylop_100way", 20)],
)
def test_invalid_feature_values(column, value):
    raw = pd.DataFrame([{"cadd_phred": 20.0, "phylop_100way": 2.0, "gnomad_af": 0.01}])
    raw[column] = value
    with pytest.raises(ValidationError):
        make_features(raw)


def test_legacy_contract_does_not_silently_impute_unknown_af():
    with pytest.raises(ValidationError):
        make_features(
            pd.DataFrame([{"cadd_phred": 20, "phylop_100way": 2, "gnomad_af": None}]),
            "legacy-zero-af-v1",
        )


@pytest.mark.parametrize(
    "value,direction",
    [
        (0.0, "Benign direction"),
        (0.099, "Benign direction"),
        (0.1, "Indeterminate"),
        (0.5, "Indeterminate"),
        (0.9, "Indeterminate"),
        (0.901, "Pathogenic direction"),
        (1.0, "Pathogenic direction"),
    ],
)
def test_direction_boundary_semantics(value, direction):
    assert score_direction(value) == direction


def test_nullable_annotation_is_unknown_not_zero():
    result = parse_annotation(
        {
            "_id": "chr1:g.1A>G",
            "dbnsfp": {"cadd": {"phred": 20}, "phylop": {"100way_vertebrate": 2}},
            "gnomad_genome": None,
        }
    )
    assert result.annotation_status == "ok"
    assert result.gnomad_af is None and not result.af_available


def test_ambiguous_transcript_scores_rejected():
    with pytest.raises(ValidationError):
        parse_annotation({"_id": "chr1:g.1A>G", "dbnsfp": {"cadd": {"phred": [20, 30]}}})


def test_equal_multiple_source_values_are_accepted():
    annotation = parse_annotation(
        {
            "_id": "chr1:g.1A>G",
            "dbnsfp": {"cadd": {"phred": [20, 20]}, "phylop": {"100way_vertebrate": {"score": 2}}},
        }
    )
    assert annotation.cadd_phred == 20


def test_annotation_identity_mismatch_rejected():
    with pytest.raises(ValidationError):
        parse_annotation({"_id": "chr1:g.2A>G"}, "chr1:g.1A>G")


def test_notfound_retains_requested_identity():
    value = parse_annotation({"query": "chr1:g.1A>G", "notfound": True})
    assert value.hgvs_id == "chr1:g.1A>G" and value.annotation_status == "not_found"


def test_shared_gene_components_stay_together():
    frame = pd.DataFrame({"GeneSymbol": ["A;B", "B", "B|C", "C", "D"]})
    groups = gene_groups(frame)
    assert groups.iloc[:4].nunique() == 1
    assert groups.iloc[4] != groups.iloc[0]


def test_partition_reproducibility_and_gene_holdout(small_dataset):
    a, b = make_splits(small_dataset), make_splits(small_dataset)
    pd.testing.assert_frame_equal(a, b)
    assert_disjoint(
        *(a[a.split == key] for key in ["fit", "calibration", "variant_test", "gene_test"])
    )
    assert_disjoint(a[a.split != "gene_test"], a[a.split == "gene_test"], gene_disjoint=True)


def test_duplicate_variants_cannot_be_split(small_dataset):
    with pytest.raises(ValidationError):
        make_splits(pd.concat([small_dataset, small_dataset.iloc[:1]]))


def test_join_cannot_expand_duplicate_annotations(small_dataset):
    annotations = pd.DataFrame({"hgvs_id": small_dataset.hgvs_id, "annotation_status": "ok"})
    annotations = pd.concat([annotations, annotations.iloc[:1]])
    with pytest.raises(ValidationError):
        merge_annotations(small_dataset, annotations)


def test_conflicting_identity_labels_rejected():
    with pytest.raises(ValidationError):
        deduplicate(
            pd.DataFrame(
                {
                    "hgvs_id": ["chr1:g.1A>G"] * 2,
                    "ClinicalSignificance": ["Pathogenic", "Benign"],
                    "VariationID": ["1", "2"],
                }
            )
        )


def test_reference_and_fractional_positions_validated():
    frame = pd.DataFrame(
        {
            "Chromosome": ["1", "1", "1"],
            "PositionVCF": [1, 2.5, 3],
            "ReferenceAlleleVCF": ["A", "A", "N"],
            "AlternateAlleleVCF": ["G", "G", "G"],
        }
    )
    assert len(add_identities(frame)) == 1

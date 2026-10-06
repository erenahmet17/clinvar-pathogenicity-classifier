import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from clinvar_classifier.evaluate import external_cohort, metrics
from clinvar_classifier.features import FEATURES_V2, make_features
from clinvar_classifier.inference import Predictor
from clinvar_classifier.io import sha256, write_json, write_parquet
from clinvar_classifier.schemas import ValidationError
from clinvar_classifier.train import train_run


@pytest.fixture
def tiny_bundle(tmp_path, small_dataset):
    model = xgb.XGBClassifier(n_estimators=15, max_depth=2, n_jobs=1, random_state=42)
    model.fit(make_features(small_dataset), small_dataset.label)
    model.save_model(tmp_path / "model.ubj")
    vus = small_dataset.iloc[:5].copy()
    vus["model_score"] = model.predict_proba(make_features(vus))[:, 1]
    vus["direction"] = "Indeterminate"
    write_parquet(tmp_path / "vus_scores.parquet", vus)
    write_json(
        tmp_path / "manifest.json",
        {
            "run_id": "test-fixture",
            "assembly": "GRCh38",
            "feature_contract": "snv-af-missing-v2",
            "features": FEATURES_V2,
            "n_fit": 600,
            "known_scope_exclusions": {"chr6:g.26092913G>A": "Pathogenic, low penetrance"},
            "files": {"model.ubj": sha256(tmp_path / "model.ubj")},
            "optional_files": {"vus_scores.parquet": sha256(tmp_path / "vus_scores.parquet")},
        },
    )
    return tmp_path


def test_bundle_native_predictions_and_shap_additivity(tiny_bundle, small_dataset):
    predictor = Predictor(tiny_bundle)
    frame = small_dataset.iloc[:8]
    result = predictor.predict(frame)
    assert np.isfinite(result.model_score).all()
    assert result.model_score.between(0, 1).all()
    features, values, base = predictor.explain(frame)
    margin = predictor.model.predict(xgb.DMatrix(features), output_margin=True)
    np.testing.assert_allclose(values.sum(axis=1) + base, margin, atol=1e-5)


def test_bundle_checksum_prevents_model_drift(tiny_bundle):
    with (tiny_bundle / "model.ubj").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValidationError):
        Predictor(tiny_bundle)


def test_special_penetrance_variant_is_not_given_a_binary_score(tiny_bundle):
    predictor = Predictor(tiny_bundle)
    with pytest.raises(ValidationError, match="outside the binary model scope"):
        predictor.predict(
            pd.DataFrame(
                [
                    {
                        "hgvs_id": "chr6:g.26092913G>A",
                        "cadd_phred": 27,
                        "phylop_100way": 5.226,
                        "gnomad_af": 0.0373216,
                    }
                ]
            )
        )


def test_optional_vus_does_not_break_inference(tiny_bundle, small_dataset):
    (tiny_bundle / "vus_scores.parquet").unlink()
    assert len(Predictor(tiny_bundle).predict(small_dataset.iloc[:1])) == 1


def test_manifest_feature_order_is_enforced(tiny_bundle):
    manifest = json.loads((tiny_bundle / "manifest.json").read_text())
    manifest["features"] = manifest["features"][::-1]
    write_json(tiny_bundle / "manifest.json", manifest)
    with pytest.raises(ValidationError):
        Predictor(tiny_bundle)


def test_external_identity_filter_and_reclassified_vus(small_dataset):
    earlier = small_dataset.iloc[:100]
    previous_vus = small_dataset.iloc[100:110]
    later = small_dataset.iloc[:120]
    new, reclassified = external_cohort(earlier, later, previous_vus)
    assert len(new) == 20 and len(reclassified) == 10
    assert not set(new.hgvs_id) & set(earlier.hgvs_id)


def test_metrics_single_class_and_score_bounds():
    result = metrics([0, 0], [0.1, 0.2])
    assert result["roc_auc"] is None
    assert result["sensitivity"] is None
    with pytest.raises(ValidationError):
        metrics([0, 1], [0.1, np.inf])


def test_training_writes_disjoint_identity_manifest(tmp_path, small_dataset):
    report = train_run(small_dataset, tmp_path, {"test_fixture": True}, iterations=1, folds=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_fit"] + manifest["n_calibration"] < len(small_dataset)
    assert report["test"]["gene_test"]["calibrated_score"]["n"] > 0
    partitioned = pd.read_parquet(tmp_path / "split_manifest.parquet")
    assert partitioned.hgvs_id.is_unique
    assert partitioned.loc[partitioned.split != "fit", "cv_validation_fold"].eq(-1).all()
    assert partitioned.loc[partitioned.split == "fit", "cv_validation_fold"].between(1, 2).all()
    assert set(partitioned[partitioned.split == "gene_test"].gene_group).isdisjoint(
        partitioned[partitioned.split != "gene_test"].gene_group
    )
    predictor = Predictor(tmp_path)
    assert predictor.manifest["calibration"]["slope"] > 0
    assert len(predictor.predict(small_dataset.iloc[:2])) == 2
    with pytest.raises(ValidationError):
        train_run(small_dataset, tmp_path, {}, iterations=1, folds=2)

"""Versioned ClinVar cohorts with explicit label policy and one-pass VCF identity."""

import re
from pathlib import Path

import pandas as pd

from .io import sha256, utc_now, write_json, write_parquet
from .schemas import ValidationError, normalize_hgvs

GOOD_REVIEW = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}
LABEL_POLICY = "p-lp-vs-b-lb-exclude-special-v2"
LABELS = {
    "pathogenic": 1,
    "likely pathogenic": 1,
    "pathogenic/likely pathogenic": 1,
    "benign": 0,
    "likely benign": 0,
    "benign/likely benign": 0,
}
COLUMNS = [
    "#AlleleID",
    "VariationID",
    "Type",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "Assembly",
    "Chromosome",
    "ReviewStatus",
    "LastEvaluated",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "OriginSimple",
]


def normalized_label(value) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().casefold())
    return re.sub(r"\s*/\s*", "/", value)


def map_label(value):
    return LABELS.get(normalized_label(value))


def add_identities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    identities = []
    for chrom, position, ref, alt in frame[
        ["Chromosome", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    ].itertuples(index=False, name=None):
        try:
            chrom = "M" if str(chrom) == "MT" else str(chrom)
            number = float(position)
            if not number.is_integer():
                raise ValidationError("Fractional coordinate.")
            identities.append(normalize_hgvs(f"chr{chrom}:g.{int(number)}{ref}>{alt}"))
        except (ValidationError, ValueError, TypeError):
            identities.append(None)
    frame["hgvs_id"] = identities
    return frame.dropna(subset=["hgvs_id"])


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    categories = frame.groupby("hgvs_id")["ClinicalSignificance"].agg(
        lambda s: set(s.map(normalized_label))
    )
    conflicting = categories[categories.map(len) > 1]
    if not conflicting.empty:
        raise ValidationError(f"Conflicting duplicate identities: {conflicting.index[:5].tolist()}")
    frame = frame.sort_values(["hgvs_id", "VariationID"])
    return frame.drop_duplicates("hgvs_id").reset_index(drop=True)


def read_cohort(snapshot: Path, out_dir: Path, source_url=None, chunksize=200_000):
    exclusions = []
    pieces, stages, categories = (
        [],
        {"raw": 0, "grch38": 0, "review_2plus": 0, "snv": 0, "eligible_label_or_vus": 0},
        {},
    )
    for chunk in pd.read_csv(
        snapshot, sep="\t", usecols=COLUMNS, dtype=str, chunksize=chunksize, keep_default_na=False
    ):
        stages["raw"] += len(chunk)
        chunk = chunk[chunk.Assembly == "GRCh38"]
        stages["grch38"] += len(chunk)
        chunk = chunk[chunk.ReviewStatus.isin(GOOD_REVIEW)]
        stages["review_2plus"] += len(chunk)
        chunk = chunk[chunk.Type == "single nucleotide variant"]
        stages["snv"] += len(chunk)
        for key, count in chunk.ClinicalSignificance.value_counts().items():
            categories[key] = categories.get(key, 0) + int(count)
        special = chunk.ClinicalSignificance.str.contains(
            "low penetrance|risk factor|risk allele", case=False, regex=True
        )
        exclusions.append(add_identities(chunk.loc[special]))
        chunk["label"] = chunk.ClinicalSignificance.map(map_label)
        chunk["is_vus"] = chunk.ClinicalSignificance.map(normalized_label).eq(
            "uncertain significance"
        )
        chunk = chunk[chunk.label.notna() | chunk.is_vus].copy()
        stages["eligible_label_or_vus"] += len(chunk)
        pieces.append(add_identities(chunk))
    if not pieces:
        raise ValidationError("No eligible records in snapshot.")
    combined = pd.concat(pieces, ignore_index=True).rename(columns={"#AlleleID": "AlleleID"})
    before = len(combined)
    combined = deduplicate(combined)
    stages["valid_identities_before_dedup"] = before
    stages["unique_identities"] = len(combined)
    manifest = {
        "snapshot_sha256": sha256(snapshot),
        "snapshot_filename": Path(snapshot).name,
        "source_url": source_url,
        "created_at": utc_now(),
        "assembly": "GRCh38",
        "label_policy": LABEL_POLICY,
        "filters": stages,
        "label_categories_before_policy": categories,
        "counts": {"labeled": int(combined.label.notna().sum()), "vus": int(combined.is_vus.sum())},
        "scope": "2+ review-status GRCh38 SNVs; special penetrance/risk classifications excluded",
    }
    out_dir = Path(out_dir)
    write_parquet(out_dir / "cohort.parquet", combined)
    excluded = pd.concat(exclusions, ignore_index=True)[
        ["hgvs_id", "ClinicalSignificance"]
    ].drop_duplicates("hgvs_id")
    write_parquet(out_dir / "scope_exclusions.parquet", excluded)
    manifest["known_scope_exclusions"] = len(excluded)
    write_json(out_dir / "cohort_manifest.json", manifest)
    return combined, manifest


def merge_annotations(cohort: pd.DataFrame, annotations: pd.DataFrame):
    if cohort.hgvs_id.duplicated().any() or annotations.hgvs_id.duplicated().any():
        raise ValidationError("Cohort and annotations must have unique variant identities.")
    if set(cohort.hgvs_id) != set(annotations.hgvs_id):
        raise ValidationError("All cohort identities must have an annotation status.")
    joined = cohort.merge(annotations, on="hgvs_id", validate="one_to_one", how="left")
    status = joined.annotation_status.value_counts().to_dict()
    usable = joined[joined.annotation_status == "ok"].copy()
    counts = {
        str(label): {
            "before": int((joined.label == label).sum()),
            "after": int((usable.label == label).sum()),
        }
        for label in [0, 1]
    }
    return usable.reset_index(drop=True), {
        "annotation_status": status,
        "class_retention": counts,
        "vus_before": int(joined.is_vus.sum()),
        "vus_after": int(usable.is_vus.sum()),
    }

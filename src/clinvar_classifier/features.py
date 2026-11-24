"""One feature contract for training, manual entry and live annotation."""

import numpy as np
import pandas as pd

from .schemas import ValidationError, finite_number

FEATURES_V2 = ["cadd_phred", "phylop_100way", "gnomad_af_log", "gnomad_af_missing"]
FEATURES_LEGACY = FEATURES_V2[:3]
CONTRACT_V2 = "snv-af-missing-v2"

# v3 (S4): missense-only, no population frequency. gnomAD AF is excluded because ACMG/AMP
# frequency criteria (BA1/BS1/PM2) are part of how ClinVar labels are assigned; see
# experiments/CIRCULARITY_RESULTS.md. AF is applied as its own criterion, not in this score.
CONTRACT_V3 = "missense-noaf-v3"
FEATURES_V3 = [
    "esm1b",
    "sift4g",
    "provean",
    "gerp_91way",
    "phylop_100way",
    "phylop_470way",
    "phastcons_100way",
    "alphamissense",
    "cadd_phred",
]
V3_REQUIRED = {"cadd_phred", "phylop_100way"}
V3_BOUNDS = {
    "esm1b": (0, 1),
    "sift4g": (0, 1),
    "provean": (0, 1),
    "gerp_91way": (-20, 10),
    "phylop_100way": (-20, 15),
    "phylop_470way": (-20, 15),
    "phastcons_100way": (0, 1),
    "alphamissense": (0, 1),
    "cadd_phred": (0, 99),
}
FEATURE_CONTRACT = CONTRACT_V3
FEATURES = {CONTRACT_V2: FEATURES_V2, CONTRACT_V3: FEATURES_V3}


def _bounded(series: pd.Series, field: str, minimum, maximum) -> pd.Series:
    return series.map(
        lambda v: None if pd.isna(v) else finite_number(v, field, minimum=minimum, maximum=maximum)
    ).astype(float)


def in_v3_scope(frame: pd.DataFrame) -> pd.Series:
    """Rows a v3 model may score: dbNSFP missense with the required CADD/phyloP values."""
    if "consequence" not in frame:
        return pd.Series(False, index=frame.index)
    return frame.consequence.eq("missense") & frame[list(V3_REQUIRED)].notna().all(axis=1)


def _make_v3(frame: pd.DataFrame) -> pd.DataFrame:
    missing_required = V3_REQUIRED - set(frame.columns)
    if missing_required:
        raise ValidationError(f"Missing input columns: {sorted(missing_required)}")
    if "consequence" in frame:
        outside = ~frame.consequence.eq("missense")
        if outside.any():
            raise ValidationError(
                "This model scores dbNSFP missense SNVs only; "
                f"consequence {sorted(frame.loc[outside, 'consequence'].astype(str).unique())} "
                "is outside the v3 scope. No score produced."
            )
    output = pd.DataFrame(index=frame.index)
    for field in FEATURES_V3:
        low, high = V3_BOUNDS[field]
        column = frame[field] if field in frame else pd.Series(np.nan, index=frame.index)
        output[field] = _bounded(column, field, low, high)
    if output[list(V3_REQUIRED)].isna().any().any():
        raise ValidationError(
            "CADD and phyloP 100way are required; missing annotations do not establish consequence."
        )
    return output[FEATURES_V3]


def make_features(frame: pd.DataFrame, contract: str = FEATURE_CONTRACT) -> pd.DataFrame:
    if contract == CONTRACT_V3:
        return _make_v3(frame)
    required = {"cadd_phred", "phylop_100way", "gnomad_af"}
    if not required.issubset(frame.columns):
        raise ValidationError(f"Missing input columns: {sorted(required - set(frame.columns))}")
    values = {}
    for field, minimum, maximum in [
        ("cadd_phred", 0, 99),
        ("phylop_100way", -20, 15),
        ("gnomad_af", 0, 1),
    ]:
        values[field] = _bounded(frame[field], field, minimum, maximum)
    if values["cadd_phred"].isna().any() or values["phylop_100way"].isna().any():
        raise ValidationError(
            "CADD and phyloP are required; missing annotations do not establish consequence."
        )
    af = values["gnomad_af"]
    if "af_available" in frame:
        if not frame["af_available"].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValidationError("af_available must contain booleans, not text or numeric flags.")
        available = frame["af_available"].fillna(False).astype(bool)
        if (available & af.isna()).any():
            raise ValidationError("AF marked available but no numeric frequency was supplied.")
        af = af.where(available)
    output = pd.DataFrame(
        {
            "cadd_phred": values["cadd_phred"],
            "phylop_100way": values["phylop_100way"],
            "gnomad_af_log": np.log10(af + 1e-9),
        },
        index=frame.index,
    )
    if contract == "legacy-zero-af-v1":
        if af.isna().any():
            raise ValidationError(
                "The legacy model cannot distinguish unknown AF; prediction is unavailable."
            )
        return output[FEATURES_LEGACY]
    if contract != CONTRACT_V2:
        raise ValidationError(f"Unknown feature contract: {contract}")
    output["gnomad_af_missing"] = af.isna().astype(float)
    return output[FEATURES_V2]


def score_direction(score: float, low=0.1, high=0.9) -> str:
    score = finite_number(score, "model_score", minimum=0, maximum=1)
    if score is None:
        raise ValidationError("A model score is required.")
    if score < low:
        return "Benign direction"
    if score > high:
        return "Pathogenic direction"
    return "Indeterminate"

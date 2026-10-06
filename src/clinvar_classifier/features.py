"""One feature contract for training, manual entry and live annotation."""

import numpy as np
import pandas as pd

from .schemas import ValidationError, finite_number

FEATURES_V2 = ["cadd_phred", "phylop_100way", "gnomad_af_log", "gnomad_af_missing"]
FEATURES_LEGACY = FEATURES_V2[:3]
FEATURE_CONTRACT = "snv-af-missing-v2"


def make_features(frame: pd.DataFrame, contract: str = FEATURE_CONTRACT) -> pd.DataFrame:
    required = {"cadd_phred", "phylop_100way", "gnomad_af"}
    if not required.issubset(frame.columns):
        raise ValidationError(f"Missing input columns: {sorted(required - set(frame.columns))}")
    values = {}
    for field, minimum, maximum in [
        ("cadd_phred", 0, 99),
        ("phylop_100way", -20, 15),
        ("gnomad_af", 0, 1),
    ]:
        values[field] = (
            frame[field]
            .map(
                lambda v: (
                    None
                    if pd.isna(v)
                    else finite_number(v, field, minimum=minimum, maximum=maximum)
                )
            )
            .astype(float)
        )
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
    if contract != FEATURE_CONTRACT:
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

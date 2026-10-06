"""Verified native XGBoost bundle loading and shared score semantics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import make_features, score_direction
from .io import sha256
from .schemas import ValidationError, normalize_hgvs


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1 / (1 + np.exp(-np.clip(values, -700, 700)))


class Predictor:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        for path, expected in self.manifest["files"].items():
            candidate = (self.directory / path).resolve()
            if self.directory not in candidate.parents:
                raise ValidationError("Artifact path escapes bundle directory.")
            if not candidate.exists() or sha256(candidate) != expected:
                raise ValidationError(f"Missing or mismatched artifact: {path}")
        self.model = xgb.Booster()
        self.model.load_model(self.directory / "model.ubj")
        if self.model.feature_names != self.manifest["features"]:
            raise ValidationError("Model feature order differs from bundle manifest.")
        if self.manifest.get("assembly") != "GRCh38":
            raise ValidationError("Only GRCh38 bundles are supported.")

    def check_variant_scope(self, hgvs_id):
        identity = normalize_hgvs(hgvs_id)
        reason = self.manifest.get("known_scope_exclusions", {}).get(identity)
        if reason:
            raise ValidationError(
                f"This variant is outside the binary model scope: {reason}. No score produced."
            )
        return identity

    def score_features(self, features):
        if list(features.columns) != self.manifest["features"]:
            raise ValidationError("Feature order mismatch.")
        matrix = xgb.DMatrix(features, feature_names=self.manifest["features"])
        margin = self.model.predict(matrix, output_margin=True)
        calibration = self.manifest.get("calibration")
        if calibration:
            return sigmoid(calibration["slope"] * margin + calibration["intercept"])
        return sigmoid(margin)

    def predict(self, raw: pd.DataFrame):
        if "hgvs_id" in raw:
            for value in raw.hgvs_id:
                self.check_variant_scope(value)
        features = make_features(raw, self.manifest["feature_contract"])
        scores = self.score_features(features)
        return pd.DataFrame(
            {"model_score": scores, "direction": [score_direction(p) for p in scores]},
            index=raw.index,
        )

    def explain(self, raw):
        features = make_features(raw, self.manifest["feature_contract"])
        matrix = xgb.DMatrix(features, feature_names=self.manifest["features"])
        contributions = self.model.predict(matrix, pred_contribs=True)
        margin = self.model.predict(matrix, output_margin=True)
        np.testing.assert_allclose(contributions.sum(axis=1), margin, rtol=1e-5, atol=1e-5)
        return features, contributions[:, :-1], contributions[:, -1]

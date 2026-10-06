"""One-time, restricted conversion of the repository's trusted legacy pickle.

The default application never loads pickle. Historical CSV counts retain their
original limitations and are not relabeled as corrected results.
"""

import io
import json
import pickle
from pathlib import Path

import pandas as pd

from clinvar_classifier.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("xgboost.sklearn", "XGBClassifier"),
            ("xgboost.core", "Booster"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy", "dtype"),
            ("builtins", "bytearray"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unapproved class {module}.{name}")
        return super().find_class(module, name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "clinvar_app")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "legacy")
    args = parser.parse_args()
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    model = LegacyUnpickler(io.BytesIO((source / "xgb_model.pkl").read_bytes())).load()
    model.save_model(output / "model.ubj")
    tests = pd.read_csv(source / "test_predictions.csv")
    from clinvar_classifier.evaluate import metrics

    evaluation = {
        "status": "historical; test used during original model selection",
        "test": metrics(tests.y_test, tests.y_proba),
        "limitations": [
            "No training snapshot or variant identities in original test CSV",
            "AF missingness was merged with zero",
            "No independent VUS clinical validation",
        ],
    }
    write_json(output / "historical_metrics.json", evaluation)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 2,
            "run_id": "legacy-05511a0",
            "assembly": "GRCh38",
            "feature_contract": "legacy-zero-af-v1",
            "features": model.get_booster().feature_names,
            "validation_status": "historical comparison only; never default production bundle",
            "clinical_probability_validated": False,
            "source_pickle_sha256": sha256(source / "xgb_model.pkl"),
            "files": {"model.ubj": sha256(output / "model.ubj")},
            "optional_files": {
                "historical_metrics.json": sha256(output / "historical_metrics.json")
            },
        },
    )
    print(json.dumps({"output": str(output), "roc_auc": evaluation["test"]["roc_auc"]}))


if __name__ == "__main__":
    main()

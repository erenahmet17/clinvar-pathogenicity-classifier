"""Verify real bundle outputs, split/fold identity and default Streamlit flows."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from clinvar_classifier.inference import Predictor
from clinvar_classifier.io import sha256, write_json
from clinvar_classifier.split import assert_disjoint

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "research-v2"


def main():
    predictor = Predictor(BUNDLE)
    for name, digest in predictor.manifest["optional_files"].items():
        if sha256(BUNDLE / name) != digest:
            raise ValueError(f"Optional artifact checksum mismatch: {name}")
    predictions = pd.read_parquet(BUNDLE / "test_predictions.parquet")
    reproduced = predictor.predict(predictions).model_score.to_numpy()
    np.testing.assert_allclose(reproduced, predictions.model_score, rtol=1e-6, atol=1e-7)
    features, values, base = predictor.explain(predictions.iloc[:100])
    split = pd.read_parquet(BUNDLE / "split_manifest.parquet")
    assert_disjoint(
        *(split[split.split == key] for key in ["fit", "calibration", "variant_test", "gene_test"])
    )
    assert_disjoint(
        split[split.split == "gene_test"], split[split.split != "gene_test"], gene_disjoint=True
    )
    fit = split[split.split == "fit"]
    for fold in sorted(fit.cv_validation_fold.unique()):
        assert_disjoint(
            fit[fit.cv_validation_fold == fold],
            fit[fit.cv_validation_fold != fold],
            gene_disjoint=True,
        )
    app = AppTest.from_file(str(ROOT / "clinvar_app" / "app.py"), default_timeout=30).run()
    if app.exception:
        raise AssertionError(app.exception)
    app.button[0].click().run()
    if app.exception:
        raise AssertionError(app.exception)
    app.radio[0].set_value("VUS explorer").run()
    app.text_input[0].set_value("[").run()
    if app.exception:
        raise AssertionError(app.exception)
    app.radio[0].set_value("Performance").run()
    if app.exception:
        raise AssertionError(app.exception)
    app.radio[0].set_value("Variant search").run()
    app.text_input[0].set_value("chr6:g.26092913G>A")
    app.button[0].click().run()
    if app.exception or not any(
        "outside the binary model scope" in error.value for error in app.error
    ):
        raise AssertionError("Expected HFE scope rejection before a network request.")
    write_json(
        ROOT / "validation_logs" / "bundle_verification.json",
        {
            "default_app_pages": "passed",
            "manual_unknown_af": "passed",
            "literal_gene_filter": "passed",
            "hfe_scope_rejection": "passed",
            "mandatory_and_optional_checksums": "passed",
            "variant_gene_and_cv_fold_disjointness": "passed",
            "test_prediction_rows": len(predictions),
            "max_saved_prediction_difference": float(
                np.max(np.abs(reproduced - predictions.model_score.to_numpy()))
            ),
            "native_shap_additivity": "passed on 100 test rows",
            "features": features.columns.tolist(),
            "shap_rows": len(values),
            "shap_base_rows": len(base),
        },
    )
    print("Real bundle and default four-page Streamlit checks passed.")


if __name__ == "__main__":
    main()

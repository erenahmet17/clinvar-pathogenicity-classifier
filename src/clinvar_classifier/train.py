"""Development-only selection; independent calibration and two untouched test partitions."""

import json
import logging
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics, single_feature_baselines, subgroup_report
from .features import (
    CONTRACT_V2,
    CONTRACT_V3,
    FEATURE_CONTRACT,
    FEATURES,
    in_v3_scope,
    make_features,
)
from .inference import sigmoid
from .io import sha256, utc_now, write_json, write_parquet
from .plots import development_eda, evaluation_figures, shap_summary
from .schemas import ValidationError
from .split import make_splits

LOGGER = logging.getLogger(__name__)


def environment():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "requests", "pyarrow"]
        },
    }


V3_CLEAN = [
    "esm1b",
    "sift4g",
    "provean",
    "gerp_91way",
    "phylop_100way",
    "phylop_470way",
    "phastcons_100way",
]
ABLATIONS = {
    CONTRACT_V2: {
        "XGBoost CADD only": ["cadd_phred"],
        "XGBoost AF only (explicit missingness)": ["gnomad_af_log", "gnomad_af_missing"],
        "XGBoost CADD + AF": ["cadd_phred", "gnomad_af_log", "gnomad_af_missing"],
    },
    CONTRACT_V3: {
        "XGBoost CADD only": ["cadd_phred"],
        "XGBoost CADD + phyloP (no AF)": ["cadd_phred", "phylop_100way"],
        "XGBoost label-free scores only (S2)": V3_CLEAN,
        "XGBoost S2 + AlphaMissense": V3_CLEAN + ["alphamissense"],
    },
}
SCOPE = {
    CONTRACT_V2: "2+ review-status GRCh38 SNVs with CADD/phyloP; special penetrance/risk labels excluded; no inferred clinical classification",
    CONTRACT_V3: "2+ review-status GRCh38 dbNSFP missense SNVs with CADD/phyloP; population frequency not used as a feature; special penetrance/risk labels excluded; no inferred clinical classification",
}


def train_run(
    dataset: pd.DataFrame,
    output: Path,
    provenance: dict,
    seed=42,
    iterations=12,
    folds=5,
    contract=FEATURE_CONTRACT,
):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValidationError("Completed runs are immutable; choose a new output directory.")
    if contract not in FEATURES:
        raise ValidationError(f"Unknown feature contract: {contract}")
    labeled = dataset[dataset.label.notna()].copy()
    scope_counts = {"labeled_rows": len(labeled)}
    if contract == CONTRACT_V3:
        labeled = labeled[in_v3_scope(labeled)].copy()
        scope_counts["in_scope_missense_rows"] = len(labeled)
    partitioned = make_splits(labeled, seed)
    write_parquet(
        output / "split_manifest.parquet",
        partitioned[
            ["hgvs_id", "AlleleID", "VariationID", "GeneSymbol", "gene_group", "label", "split"]
        ],
    )
    write_json(
        output / "protocol.json",
        {
            "seed": seed,
            "cv_folds": folds,
            "search_iterations": iterations,
            "model_family": "predeclared XGBoost",
            "feature_contract": contract,
            "selection_data": "fit partition only",
            "calibration_data": "calibration partition only",
            "test_partitions": ["variant_test", "gene_test"],
            "test_opened_at": None,
        },
    )
    fit = partitioned[partitioned.split == "fit"].copy()
    calibration = partitioned[partitioned.split == "calibration"].copy()
    development_eda(fit, output, contract)
    X, y = make_features(fit, contract), fit.label.astype(int)
    cv = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            X, y, groups=fit.gene_group
        )
    )
    for training, validation in cv:
        if set(fit.iloc[training].gene_group) & set(fit.iloc[validation].gene_group):
            raise ValidationError("Group overlap in selection folds.")
        if fit.iloc[validation].label.nunique() != 2:
            raise ValidationError("A development fold has only one class.")
    partitioned["cv_validation_fold"] = -1
    for fold_id, (_, validation) in enumerate(cv, 1):
        partitioned.loc[fit.iloc[validation].index, "cv_validation_fold"] = fold_id
    write_parquet(
        output / "split_manifest.parquet",
        partitioned[
            [
                "hgvs_id",
                "AlleleID",
                "VariationID",
                "GeneSymbol",
                "gene_group",
                "label",
                "split",
                "cv_validation_fold",
            ]
        ],
    )
    estimator = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=2,
    )
    candidates = {
        "Logistic regression (fold-local imputation/scaling)": make_pipeline(
            SimpleImputer(add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=1500, random_state=seed),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(add_indicator=True),
            RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=2),
        ),
        "XGBoost baseline": estimator,
    }
    records = []
    for name, candidate in candidates.items():
        LOGGER.info("Development CV: %s", name)
        scores = cross_val_score(
            candidate, X, y, cv=cv, scoring="roc_auc", n_jobs=2, error_score="raise"
        )
        records.extend(
            {"model": name, "fold": i + 1, "roc_auc": float(score)}
            for i, score in enumerate(scores)
        )
    ablation_frame = X
    if contract == CONTRACT_V3:
        # Reference only: shows the circular frequency contribution; never selectable.
        af = make_features(fit, CONTRACT_V2)[["gnomad_af_log", "gnomad_af_missing"]]
        ablation_frame = pd.concat([X, af], axis=1)
    ablations = dict(ABLATIONS[contract])
    if contract == CONTRACT_V3:
        ablations["Reference only, not selectable: + gnomAD AF (circular)"] = list(X.columns) + [
            "gnomad_af_log",
            "gnomad_af_missing",
        ]
    for name, columns in ablations.items():
        LOGGER.info("Development ablation: %s", name)
        scores = cross_val_score(
            clone(estimator),
            ablation_frame[columns],
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=2,
            error_score="raise",
        )
        records.extend(
            {"model": name, "fold": i + 1, "roc_auc": float(score)}
            for i, score in enumerate(scores)
        )
    search = RandomizedSearchCV(
        estimator,
        {
            "n_estimators": [150, 250, 400],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.75, 1.0],
        },
        n_iter=iterations,
        cv=cv,
        scoring="roc_auc",
        n_jobs=2,
        random_state=seed,
        error_score="raise",
        refit=True,
    )
    LOGGER.info("Development-only randomized search: %s iterations", iterations)
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(output / "search_results.csv", index=False)
    model = search.best_estimator_
    for i in range(folds):
        records.append(
            {
                "model": "XGBoost selected parameters",
                "fold": i + 1,
                "roc_auc": float(search.cv_results_[f"split{i}_test_score"][search.best_index_]),
            }
        )
    write_parquet(output / "cv_scores.parquet", pd.DataFrame(records))
    model.save_model(output / "model.ubj")
    # Calibration uses independent rows, no class weighting and no test-set decisions.
    margins = model.predict(make_features(calibration, contract), output_margin=True)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=seed)
    calibrator.fit(margins.reshape(-1, 1), calibration.label.astype(int))
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    if slope <= 0:
        raise ValidationError("Nonpositive calibration slope; do not publish this model.")
    calibration_spec = {
        "method": "sigmoid-on-independent-calibration-rows",
        "slope": slope,
        "intercept": intercept,
        "clinical_probability_validated": False,
    }
    report = {
        "development": {
            "best_cv_roc_auc": float(search.best_score_),
            "best_params": search.best_params_,
            "fold_scores": records,
            "comparison_limitation": "Shared grouped development folds; no formal statistical equivalence claim; selected CV score is subject to selection optimism.",
        },
        "partition_counts": {
            key: {
                "n": len(group),
                "n_benign": int((group.label == 0).sum()),
                "n_pathogenic": int((group.label == 1).sum()),
                "n_gene_groups": int(group.gene_group.nunique()),
            }
            for key, group in partitioned.groupby("split")
        },
        "test": {},
        "calibration": calibration_spec,
        "feature_contract": contract,
        "scope_counts": scope_counts,
    }
    predictions = []
    # Tests are first accessed for evaluation after all fitting/selection/calibration completes.
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    protocol["test_opened_at"] = utc_now()
    write_json(output / "protocol.json", protocol)
    for name in ["variant_test", "gene_test"]:
        test = partitioned[partitioned.split == name].copy()
        test_features = make_features(test, contract)
        raw = model.predict_proba(test_features)[:, 1]
        score = sigmoid(slope * model.predict(test_features, output_margin=True) + intercept)
        report["test"][name] = {
            "calibrated_score": metrics(test.label, score, bootstrap=500, seed=seed),
            "raw_score": metrics(test.label, raw),
            "subgroups": subgroup_report(test, score),
            "single_feature_baselines": single_feature_baselines(test),
        }
        test["model_score"], test["raw_model_score"] = score, raw
        identity = ["hgvs_id", "AlleleID", "VariationID", "GeneSymbol", "gene_group", "Name"]
        inputs = [c for c in FEATURES[contract] if c in test] + ["gnomad_af", "af_available"]
        if "consequence" in test:
            inputs.append("consequence")
        inputs = list(dict.fromkeys(c for c in inputs if c in test))
        predictions.append(
            test[identity + ["label", "split"] + inputs + ["model_score", "raw_model_score"]]
        )
    write_parquet(output / "test_predictions.parquet", pd.concat(predictions, ignore_index=True))
    write_json(output / "metrics.json", report)
    evaluation_figures(output)
    shap_summary(model, fit, output, seed, contract)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    protocol["test_evaluation_completed_at"] = utc_now()
    write_json(output / "protocol.json", protocol)
    code_root = Path(__file__).parent
    try:
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        base_commit = None
    manifest = {
        "schema_version": 2,
        "run_id": output.name,
        "created_at": utc_now(),
        "assembly": "GRCh38",
        "feature_contract": contract,
        "features": FEATURES[contract],
        "calibration": calibration_spec,
        "thresholds": {
            "low": 0.1,
            "high": 0.9,
            "semantics": "display-only prioritization bands; not ACMG classes",
        },
        "scope": SCOPE[contract],
        "scope_counts": scope_counts,
        "validation_status": "internal variant and connected-gene holdouts; external validation reported separately",
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "provenance": provenance,
        "known_scope_exclusions": provenance.get("known_scope_exclusions", {}),
        "environment": environment(),
        "base_commit": base_commit,
        "code_sha256": {p.name: sha256(p) for p in code_root.glob("*.py")},
        "files": {"model.ubj": sha256(output / "model.ubj")},
        "optional_files": {
            name: sha256(output / name)
            for name in [
                "metrics.json",
                "test_predictions.parquet",
                "cv_scores.parquet",
                "split_manifest.parquet",
                "search_results.csv",
                "protocol.json",
                "development_eda.json",
                "development_eda.png",
                "evaluation.png",
                "shap_summary.json",
                "shap_summary.png",
            ]
        },
    }
    write_json(output / "manifest.json", manifest)
    return report

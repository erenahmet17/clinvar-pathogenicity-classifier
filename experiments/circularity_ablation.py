"""Circularity ablation: retrain the research-v2 protocol with feature subsets.

Hypothesis: gnomAD allele frequency is circular with ClinVar labels, because ACMG/AMP
frequency criteria (BA1 >5% stand-alone benign, BS1 "too common for the disorder",
PM2 "absent/rare in population databases") are part of how submitters assign the
P/LP and B/LB labels the model is trained on.

The protocol mirrors ``clinvar_classifier.train.train_run`` exactly (same seed, same
fit/calibration/variant_test/gene_test partitions, same gene-grouped CV folds, same
randomized search grid, same sigmoid calibration), but restricts the feature columns.
The temporal external cohort (Oct 2026 newly labeled variants, July VUS reclassified)
is scored the same way as ``cli evaluate-external``.

Usage (from repo root):
    python experiments/circularity_ablation.py data/processed/2026-07/dataset.parquet \
        data/processed/2026-10/dataset.parquet artifacts/experiments/circularity_ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from clinvar_classifier.evaluate import external_cohort, metrics, subgroup_report
from clinvar_classifier.features import CONTRACT_V2, make_features
from clinvar_classifier.inference import sigmoid
from clinvar_classifier.split import make_splits

FEATURE_SETS = {
    "full (CADD+phyloP+AF+AF_missing)": [
        "cadd_phred",
        "phylop_100way",
        "gnomad_af_log",
        "gnomad_af_missing",
    ],
    "no_AF (CADD+phyloP)": ["cadd_phred", "phylop_100way"],
    "no_CADD (phyloP+AF+AF_missing)": ["phylop_100way", "gnomad_af_log", "gnomad_af_missing"],
    "AF_only (AF+AF_missing)": ["gnomad_af_log", "gnomad_af_missing"],
    "phyloP_only": ["phylop_100way"],
}
GRID = {
    "n_estimators": [150, 250, 400],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.75, 1.0],
}


def bootstrap_delta(y, a, b, n=500, seed=42):
    """Paired, class-stratified bootstrap CI of AUC(a) - AUC(b) on the same rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def run(earlier_path: Path, later_path: Path, output: Path, seed=42, iterations=12, folds=5):
    output.mkdir(parents=True, exist_ok=True)
    earlier = pd.read_parquet(earlier_path)
    later = pd.read_parquet(later_path)
    labeled = earlier[earlier.label.notna()].copy()
    part = make_splits(labeled, seed)
    fit = part[part.split == "fit"]
    cal = part[part.split == "calibration"]
    X_fit_all, y_fit = make_features(fit, CONTRACT_V2), fit.label.astype(int)
    cv = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            X_fit_all, y_fit, groups=fit.gene_group
        )
    )
    new, reclassified = external_cohort(labeled, later, earlier[earlier.is_vus])
    evals = {
        "variant_test": part[part.split == "variant_test"],
        "gene_test": part[part.split == "gene_test"],
        "ext_new_labeled": new.reset_index(drop=True),
        "ext_reclassified_vus": reclassified.reset_index(drop=True),
    }
    feats = {name: make_features(frame, CONTRACT_V2) for name, frame in evals.items()}
    X_cal_all = make_features(cal, CONTRACT_V2)

    report, scores = {}, {}
    for set_name, cols in FEATURE_SETS.items():
        t0 = time.time()
        estimator = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=2,
        )
        search = RandomizedSearchCV(
            estimator,
            GRID,
            n_iter=iterations,
            cv=cv,
            scoring="roc_auc",
            n_jobs=2,
            random_state=seed,
            error_score="raise",
            refit=True,
        )
        search.fit(X_fit_all[cols], y_fit)
        model = search.best_estimator_
        margins = model.predict(X_cal_all[cols], output_margin=True)
        calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed)
        calibrator.fit(margins.reshape(-1, 1), cal.label.astype(int))
        slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
        entry = {
            "features": cols,
            "cv_best_roc_auc": float(search.best_score_),
            "cv_fold_scores": [
                float(search.cv_results_[f"split{i}_test_score"][search.best_index_])
                for i in range(folds)
            ],
            "best_params": search.best_params_,
            "calibration": {"slope": slope, "intercept": intercept},
            "eval": {},
        }
        scores[set_name] = {}
        for name, frame in evals.items():
            score = sigmoid(
                slope * model.predict(feats[name][cols], output_margin=True) + intercept
            )
            scores[set_name][name] = score
            m = metrics(frame.label.astype(int), score, bootstrap=500, seed=seed)
            entry["eval"][name] = {
                **m,
                "subgroups": subgroup_report(frame, score) if len(frame) > 50 else None,
            }
        entry["seconds"] = round(time.time() - t0, 1)
        report[set_name] = entry
        print(
            f"{set_name}: CV={entry['cv_best_roc_auc']:.4f} "
            + " ".join(f"{k}={v['roc_auc']:.4f}" for k, v in entry["eval"].items()),
            flush=True,
        )

    full, no_af = list(FEATURE_SETS)[:2]
    report["paired_delta_full_minus_noAF"] = {
        name: {
            "delta_auc": float(
                roc_auc_score(frame.label.astype(int), scores[full][name])
                - roc_auc_score(frame.label.astype(int), scores[no_af][name])
            ),
            "ci95": bootstrap_delta(
                frame.label.astype(int), scores[full][name], scores[no_af][name], seed=seed
            ),
        }
        for name, frame in evals.items()
    }
    pred = []
    for name, frame in evals.items():
        out = frame[["hgvs_id", "GeneSymbol", "label", "gnomad_af", "af_available"]].copy()
        out["evaluation_set"] = name
        for set_name in FEATURE_SETS:
            out[set_name] = scores[set_name][name]
        pred.append(out)
    pd.concat(pred, ignore_index=True).to_parquet(output / "ablation_predictions.parquet")
    (output / "ablation_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run(args.earlier, args.later, args.output, iterations=args.iterations, folds=args.folds)
    sys.exit(0)

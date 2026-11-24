"""Step 1: no-AF model evaluated where it will be used — rare missense variants.

Same research-v2 partitions (seed 42, make_splits on the full labeled July cohort), so
test rows never enter fitting. Training scope and features vary:

  all-variants      : train on every fit row (as research-v2)
  missense          : train only on missense fit rows
  rare-missense     : train only on rare missense fit rows

Missense is the Name-based proxy used in evaluate.subgroup_report (p.XxxNNNYyy, not Ter).
Rare = gnomAD genome AF <= 1e-4 or AF not available. Evaluation sets are the rare
missense rows of variant_test, gene_test and the Oct-2026 newly labeled cohort, plus the
"AF observed" part of those rows (where the AF-missing shortcut cannot operate).

Usage (repo root):
    python experiments/rare_missense_eval.py data/processed/2026-07/dataset.parquet \
        data/processed/2026-10/dataset.parquet artifacts/experiments/rare_missense
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from clinvar_classifier.evaluate import external_cohort
from clinvar_classifier.features import make_features
from clinvar_classifier.inference import sigmoid
from clinvar_classifier.split import gene_groups, make_splits

FULL = ["cadd_phred", "phylop_100way", "gnomad_af_log", "gnomad_af_missing"]
NO_AF = ["cadd_phred", "phylop_100way"]
GRID = {
    "n_estimators": [150, 250, 400],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.75, 1.0],
}


def scope_masks(frame):
    name = frame.Name.fillna("").astype(str)
    stop = name.str.contains(r"p\.[^)]*(?:Ter|\*)", regex=True)
    missense = name.str.contains(r"p\.[A-Z][a-z]{2}[0-9]+[A-Z][a-z]{2}", regex=True) & ~stop
    af = frame.gnomad_af.where(frame.af_available.fillna(False).astype(bool))
    rare = af.isna() | (af <= 1e-4)
    return missense, rare, af.notna()


def auc_ci(y, s, n=500, seed=42):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_delta(y, a, b, n=500, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    ]


def fit_calibrated(fit, cal, cols, seed=42, iterations=12, folds=5):
    X, y = make_features(fit)[cols], fit.label.astype(int)
    cv = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            X, y, groups=fit.gene_group
        )
    )
    search = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2),
        GRID,
        n_iter=iterations,
        cv=cv,
        scoring="roc_auc",
        n_jobs=2,
        random_state=seed,
        error_score="raise",
        refit=True,
    )
    search.fit(X, y)
    model = search.best_estimator_
    margins = model.predict(make_features(cal)[cols], output_margin=True)
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(
        margins.reshape(-1, 1), cal.label.astype(int)
    )
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])

    def predict(frame):
        return sigmoid(
            slope * model.predict(make_features(frame)[cols], output_margin=True) + intercept
        )

    return predict, float(search.best_score_), search.best_params_


def run(earlier_path, later_path, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    earlier, later = pd.read_parquet(earlier_path), pd.read_parquet(later_path)
    labeled = earlier[earlier.label.notna()].copy()
    part = make_splits(labeled, seed)
    part["missense"], part["rare"], part["af_observed"] = scope_masks(part)
    new, _ = external_cohort(labeled, later, earlier[earlier.is_vus])
    new = new.reset_index(drop=True)
    new["gene_group"] = gene_groups(new)
    new["missense"], new["rare"], new["af_observed"] = scope_masks(new)

    evals = {}
    for name, frame in [
        ("variant_test", part[part.split == "variant_test"]),
        ("gene_test", part[part.split == "gene_test"]),
        ("ext_new_labeled", new),
    ]:
        rm = frame[frame.missense & frame.rare]
        evals[f"{name} | rare missense"] = rm
        evals[f"{name} | rare missense, AF observed"] = rm[rm.af_observed]

    fit_all, cal_all = part[part.split == "fit"], part[part.split == "calibration"]
    scopes = {
        "all-variants": (fit_all, cal_all),
        "missense": (fit_all[fit_all.missense], cal_all[cal_all.missense]),
        "rare-missense": (
            fit_all[fit_all.missense & fit_all.rare],
            cal_all[cal_all.missense & cal_all.rare],
        ),
    }
    report = {"eval_sets": {}, "models": {}, "baselines": {}, "paired_full_minus_noAF": {}}
    for name, frame in evals.items():
        report["eval_sets"][name] = {
            "n": len(frame),
            "n_benign": int((frame.label == 0).sum()),
            "n_pathogenic": int((frame.label == 1).sum()),
        }
        y = frame.label.astype(int)
        report["baselines"][name] = {
            "CADD alone": float(roc_auc_score(y, frame.cadd_phred)),
            "phyloP alone": float(roc_auc_score(y, frame.phylop_100way)),
        }
    scores = {}
    for scope, (fit, cal) in scopes.items():
        for label, cols in [("full", FULL), ("no_AF", NO_AF)]:
            key = f"{scope} / {label}"
            predict, cv_auc, params = fit_calibrated(fit, cal, cols, seed)
            entry = {
                "n_fit": len(fit),
                "n_calibration": len(cal),
                "fit_class_counts": fit.label.value_counts().to_dict(),
                "cv_best_roc_auc": cv_auc,
                "best_params": params,
                "eval": {},
            }
            scores[key] = {}
            for name, frame in evals.items():
                s = predict(frame)
                scores[key][name] = s
                y = frame.label.astype(int)
                entry["eval"][name] = {
                    "roc_auc": float(roc_auc_score(y, s)),
                    "roc_auc_ci95": auc_ci(y, s, seed=seed),
                    "average_precision": float(average_precision_score(y, s)),
                }
            report["models"][key] = entry
            print(
                key,
                f"CV={cv_auc:.4f}",
                " ".join(f"[{n}]={v['roc_auc']:.4f}" for n, v in entry["eval"].items()),
                flush=True,
            )
        report["paired_full_minus_noAF"][scope] = {
            name: dict(
                zip(
                    ["delta_auc", "ci95"],
                    paired_delta(
                        frame.label.astype(int),
                        scores[f"{scope} / full"][name],
                        scores[f"{scope} / no_AF"][name],
                        seed=seed,
                    ),
                )
            )
            for name, frame in evals.items()
        }
    (output / "rare_missense_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("earlier", type=Path)
    p.add_argument("later", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.earlier, a.later, a.output)

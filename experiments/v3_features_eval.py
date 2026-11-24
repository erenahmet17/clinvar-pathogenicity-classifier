"""Step 2b: retrain the no-AF model with low-circularity dbNSFP predictors.

Same research-v2 partitions (seed 42). Training on missense fit rows (step 1 showed the
training scope does not change rare-missense performance). Evaluation on rare missense
rows of variant_test, gene_test and the Oct-2026 newly labeled cohort, plus their
"AF observed" part. Single-score baselines include REVEL as a comparator only.

    python experiments/v3_features_eval.py data/processed/2026-07/dataset.parquet \
        data/processed/2026-10/dataset.parquet data/processed/v3_scores.parquet \
        artifacts/experiments/v3_features
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from rare_missense_eval import GRID, auc_ci, paired_delta, scope_masks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from clinvar_classifier.evaluate import external_cohort
from clinvar_classifier.features import make_features
from clinvar_classifier.inference import sigmoid
from clinvar_classifier.split import gene_groups, make_splits

CLEAN = [
    "esm1b",
    "sift4g",
    "provean",
    "gerp_91way",
    "phylop_100way",
    "phylop_470way",
    "phastcons_100way",
]
SETS = {
    "S1 CADD+phyloP (adim 1)": ["cadd_phred", "phylop_100way"],
    "S2 temiz (ESM1b+SIFT4G+PROVEAN+korunmusluk)": CLEAN,
    "S3 temiz + AlphaMissense": CLEAN + ["alphamissense"],
    "S4 temiz + AlphaMissense + CADD": CLEAN + ["alphamissense", "cadd_phred"],
    "S5 referans: S4 + AF (dongusel)": CLEAN
    + ["alphamissense", "cadd_phred", "gnomad_af_log", "gnomad_af_missing"],
}
SINGLE = [
    "cadd_phred",
    "esm1b",
    "alphamissense",
    "sift4g",
    "provean",
    "gerp_91way",
    "phylop_100way",
    "phylop_470way",
    "revel",
]


def matrix(frame):
    af = make_features(frame)[["gnomad_af_log", "gnomad_af_missing"]]
    return pd.concat([frame.drop(columns=af.columns, errors="ignore"), af], axis=1)


def run(july, october, scores_path, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    earlier, later = pd.read_parquet(july), pd.read_parquet(october)
    v3 = pd.read_parquet(scores_path)
    labeled = earlier[earlier.label.notna()].copy()
    part = make_splits(labeled, seed)
    new, _ = external_cohort(labeled, later, earlier[earlier.is_vus])
    new = new.reset_index(drop=True)
    new["gene_group"] = gene_groups(new)
    frames = {}
    for name, frame in [("part", part), ("new", new)]:
        frame = frame.merge(v3, on="hgvs_id", how="left", validate="one_to_one")
        frame["missense"], frame["rare"], frame["af_observed"] = scope_masks(frame)
        frames[name] = matrix(frame)
    part, new = frames["part"], frames["new"]
    checks = part.dropna(subset=["cadd_check"])
    cadd_agreement = float((checks.cadd_check - checks.cadd_phred).abs().le(0.05).mean())

    evals = {}
    for name, frame in [
        ("variant_test", part[part.split == "variant_test"]),
        ("gene_test", part[part.split == "gene_test"]),
        ("ext_new_labeled", new),
    ]:
        rm = frame[frame.missense & frame.rare]
        evals[f"{name} | rare missense"] = rm
        evals[f"{name} | rare missense, AF observed"] = rm[rm.af_observed]
    fit = part[(part.split == "fit") & part.missense]
    cal = part[(part.split == "calibration") & part.missense]

    report = {
        "cadd_identity_agreement_abs_le_0.05": cadd_agreement,
        "coverage_missense_fit": fit[SINGLE].notna().mean().round(4).to_dict(),
        "eval_sets": {},
        "single_score_auc": {},
        "models": {},
        "paired_vs_S1": {},
    }
    for name, frame in evals.items():
        y = frame.label.astype(int)
        report["eval_sets"][name] = {
            "n": len(frame),
            "n_benign": int((y == 0).sum()),
            "n_pathogenic": int((y == 1).sum()),
        }
        singles = {}
        for col in SINGLE:
            ok = frame[col].notna()
            if y[ok].nunique() == 2:
                singles[col] = {
                    "auc": float(roc_auc_score(y[ok], frame.loc[ok, col])),
                    "coverage": float(ok.mean()),
                }
        report["single_score_auc"][name] = singles

    X_fit, y_fit = fit, fit.label.astype(int)
    cv = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            X_fit, y_fit, groups=fit.gene_group
        )
    )
    scores = {}
    for key, cols in SETS.items():
        search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2),
            GRID,
            n_iter=12,
            cv=cv,
            scoring="roc_auc",
            n_jobs=2,
            random_state=seed,
            error_score="raise",
            refit=True,
        )
        search.fit(X_fit[cols], y_fit)
        model = search.best_estimator_
        margins = model.predict(cal[cols], output_margin=True)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(
            margins.reshape(-1, 1), cal.label.astype(int)
        )
        entry = {
            "features": cols,
            "cv_best_roc_auc": float(search.best_score_),
            "best_params": search.best_params_,
            "eval": {},
        }
        scores[key] = {}
        for name, frame in evals.items():
            s = sigmoid(
                lr.coef_[0, 0] * model.predict(frame[cols], output_margin=True)
                + lr.intercept_[0]
            )
            scores[key][name] = s
            y = frame.label.astype(int)
            entry["eval"][name] = {
                "roc_auc": float(roc_auc_score(y, s)),
                "roc_auc_ci95": auc_ci(y, s, seed=seed),
                "average_precision": float(average_precision_score(y, s)),
            }
        gain = model.get_booster().get_score(importance_type="gain")
        entry["gain_importance"] = {c: float(gain.get(c, 0.0)) for c in cols}
        report["models"][key] = entry
        print(
            key,
            f"CV={entry['cv_best_roc_auc']:.4f}",
            " ".join(f"[{n}]={v['roc_auc']:.4f}" for n, v in entry["eval"].items()),
            flush=True,
        )
    base = list(SETS)[0]
    for key in list(SETS)[1:]:
        report["paired_vs_S1"][key] = {
            name: dict(
                zip(
                    ["delta_auc", "ci95"],
                    paired_delta(
                        frame.label.astype(int), scores[key][name], scores[base][name], seed=seed
                    ),
                )
            )
            for name, frame in evals.items()
        }
    (output / "v3_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ["july", "october", "scores", "output"]:
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    run(a.july, a.october, a.scores, a.output)

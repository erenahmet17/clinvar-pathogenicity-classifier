"""Static scientific figures generated once per research run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve

from .features import make_features
from .io import write_json


def development_eda(frame: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fields = ["cadd_phred", "phylop_100way", "gnomad_af"]
    for ax, field in zip(axes.flat, fields):
        for label, color, name in [(0, "#4477aa", "B/LB"), (1, "#bb5544", "P/LP")]:
            values = frame.loc[frame.label == label, field].dropna()
            if field == "gnomad_af":
                values = np.log10(values + 1e-9)
            ax.hist(values, bins=40, density=True, alpha=0.55, label=name, color=color)
        ax.set_title(field + (" (log10 + 1e-9)" if field == "gnomad_af" else ""))
        ax.set_ylabel("Density among available observations")
        ax.legend()
    missing = [
        float(frame.loc[frame.label == label, "gnomad_af"].isna().mean()) for label in [0, 1]
    ]
    axes[1, 1].bar(["B/LB", "P/LP"], missing, color=["#4477aa", "#bb5544"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Unknown AF fraction — separate from observed zero")
    fig.suptitle("Development fit partition only • no test rows", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "development_eda.png", dpi=150)
    plt.close(fig)
    stats = {
        "partition": "fit only",
        "n": len(frame),
        "unknown_af_fraction_by_label": dict(zip(["B/LB", "P/LP"], missing)),
        "correlation_pairwise_observed": frame[fields].corr().replace({np.nan: None}).to_dict(),
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
    }
    write_json(output / "development_eda.json", stats)


def evaluation_figures(output: Path):
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(output / "test_predictions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    folds = pd.DataFrame(report["development"]["fold_scores"])
    summary = folds.groupby("model").roc_auc.agg(["mean", "std"]).sort_values("mean")
    axes[0].barh(
        summary.index.str.replace(" (fold-local imputation/scaling)", "", regex=False),
        summary["mean"],
        xerr=summary["std"],
        color="#4477aa",
    )
    axes[0].set_xlim(0.5, 1.01)
    axes[0].set_title("Grouped development CV / ablation")
    axes[0].set_xlabel("ROC-AUC mean ± fold SD")
    for (name, frame), color in zip(predictions.groupby("split"), ["#4477aa", "#bb5544"]):
        fpr, tpr, _ = roc_curve(frame.label, frame.model_score)
        auc = report["test"][name]["calibrated_score"]["roc_auc"]
        axes[1].plot(fpr, tpr, color=color, label=f"{name}: {auc:.4f} (n={len(frame):,})")
        bins = report["test"][name]["calibrated_score"]["calibration_bins"]
        axes[2].plot(
            [b["mean_score"] for b in bins],
            [b["positive_fraction"] for b in bins],
            "o-",
            color=color,
            label=name,
        )
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("Untouched test partitions")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].legend(fontsize=8)
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set_title("Test calibration — counts in metrics.json")
    axes[2].set_xlabel("Mean calibrated score")
    axes[2].set_ylabel("Observed P/LP fraction")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def shap_summary(model, fit: pd.DataFrame, output: Path, seed=42):
    sample = fit.sample(min(1000, len(fit)), random_state=seed)
    features = make_features(sample)
    matrix = xgb.DMatrix(features)
    contribution = model.get_booster().predict(matrix, pred_contribs=True)
    margin = model.predict(features, output_margin=True)
    np.testing.assert_allclose(contribution.sum(axis=1), margin, atol=1e-5, rtol=1e-5)
    importance = pd.Series(
        np.mean(np.abs(contribution[:, :-1]), axis=0), index=features.columns
    ).sort_values()
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.barh(importance.index, importance.values, color="#4477aa")
    axis.set_xlabel("Mean absolute Tree SHAP contribution (raw log-odds)")
    axis.set_title(f"Fit-partition sample (n={len(sample):,}); descriptive, not causal")
    fig.tight_layout()
    fig.savefig(output / "shap_summary.png", dpi=150)
    plt.close(fig)
    write_json(
        output / "shap_summary.json",
        {
            "n_sample": len(sample),
            "seed": seed,
            "sample_hgvs_ids": sample.hgvs_id.tolist(),
            "mean_abs_raw_margin_shap": importance.to_dict(),
            "additivity_verified": True,
        },
    )

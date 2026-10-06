"""Descriptive metrics, calibration, predeclared subgroups and external identity checks."""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from .schemas import ValidationError


def metrics(y, score, threshold=0.5, bootstrap=0, seed=42):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if len(y) == 0 or len(y) != len(score):
        raise ValidationError("Empty or mismatched evaluation data.")
    if (
        not set(np.unique(y)).issubset({0, 1})
        or not np.isfinite(score).all()
        or ((score < 0) | (score > 1)).any()
    ):
        raise ValidationError("Evaluation needs binary labels and finite scores in [0,1].")
    pred = score > threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    calibration = []
    bins = np.minimum((score * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if mask.any():
            calibration.append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "n": int(mask.sum()),
                    "mean_score": float(score[mask].mean()),
                    "positive_fraction": float(y[mask].mean()),
                }
            )

    def ratio(a, b):
        return float(a / b) if b else None

    result = {
        "n": len(y),
        "n_benign": int((y == 0).sum()),
        "n_pathogenic": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, score)) if both else None,
        "average_precision": float(average_precision_score(y, score)) if both else None,
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "accuracy": float((pred == y).mean()),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "calibration_bins": calibration,
        "ece": float(
            sum(
                c["n"] / len(y) * abs(c["mean_score"] - c["positive_fraction"]) for c in calibration
            )
        ),
    }
    if bootstrap and both:
        rng = np.random.default_rng(seed)
        indices = [np.flatnonzero(y == value) for value in [0, 1]]
        values = []
        for _ in range(bootstrap):
            sample = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in indices])
            values.append(roc_auc_score(y[sample], score[sample]))
        result["roc_auc_ci_95_row_bootstrap"] = np.quantile(values, [0.025, 0.975]).tolist()
        result["bootstrap_limitation"] = (
            "Conditional row resampling; does not account for gene clustering or domain shift."
        )
    return result


def subgroup_report(frame: pd.DataFrame, score):
    af = frame.gnomad_af
    name = frame.Name.fillna("").astype(str)
    masks = {
        "af_unknown": af.isna(),
        "af_observed_zero": af.eq(0),
        "af_observed_rare_le_1e_4": af.gt(0) & af.le(1e-4),
        "af_observed_gt_1e_3": af.gt(1e-3),
        "stop_gain_name_proxy": name.str.contains(r"p\.[^)]*(?:Ter|\*)", regex=True),
        "splice_name_proxy": name.str.contains(r"c\.[0-9]+[+-][0-9]", regex=True),
    }
    result = {}
    score = np.asarray(score)
    for key, mask in masks.items():
        if mask.any():
            result[key] = metrics(frame.loc[mask, "label"], score[mask.to_numpy()])
    missense_proxy = (
        name.str.contains(r"p\.[A-Z][a-z]{2}[0-9]+[A-Z][a-z]{2}", regex=True)
        & ~masks["stop_gain_name_proxy"]
    )
    if missense_proxy.any():
        result["missense_name_proxy"] = metrics(
            frame.loc[missense_proxy, "label"], score[missense_proxy.to_numpy()]
        )
    if "ReviewStatus" in frame:
        reviewed = frame.ReviewStatus.isin(["reviewed by expert panel", "practice guideline"])
        if reviewed.any():
            result["expert_or_guideline_review"] = metrics(
                frame.loc[reviewed, "label"], score[reviewed.to_numpy()]
            )
    result["consequence_limitation"] = (
        "Name-based stop/splice proxies are not VEP consequences; no definitive missense-only claim."
    )
    return result


def single_feature_baselines(frame: pd.DataFrame):
    result = {}
    for name, values in [
        ("cadd", frame.cadd_phred),
        ("phylop", frame.phylop_100way),
        ("inverted_observed_af", -frame.gnomad_af),
    ]:
        valid = values.notna() & frame.label.notna()
        y = frame.loc[valid, "label"]
        result[name] = {
            "n": int(valid.sum()),
            "roc_auc": float(roc_auc_score(y, values[valid])) if y.nunique() == 2 else None,
        }
    return result


def external_cohort(training: pd.DataFrame, later: pd.DataFrame, earlier_vus: pd.DataFrame):
    if training.hgvs_id.duplicated().any() or later.hgvs_id.duplicated().any():
        raise ValidationError("External evaluation requires unique identities.")
    labeled = later[later.label.notna()].copy()
    new = labeled[~labeled.hgvs_id.isin(training.hgvs_id)].copy()
    reclassified = new[new.hgvs_id.isin(earlier_vus.hgvs_id)].copy()
    if set(new.hgvs_id) & set(training.hgvs_id):
        raise ValidationError("External variants overlap the training source cohort.")
    return new, reclassified


def validate_temporal_sources(earlier, later, training_provenance, early_hash, late_hash):
    if early_hash != earlier.get("dataset_sha256") or late_hash != later.get("dataset_sha256"):
        raise ValidationError("External dataset bytes differ from their manifests.")
    if early_hash != training_provenance.get("dataset_sha256"):
        raise ValidationError("Earlier cohort is not the source dataset used to train this model.")
    namespaces = [value.get("annotation", {}).get("namespace") for value in [earlier, later]]
    if not namespaces[0] or namespaces[0] != namespaces[1]:
        raise ValidationError(
            "Temporal label evaluation requires the same frozen annotation source namespace."
        )
    months = []
    for value in [earlier, later]:
        match = re.search(
            r"(20[0-9]{2}-(?:0[1-9]|1[0-2]))\.txt\.gz$",
            value.get("cohort", {}).get("snapshot_filename", ""),
        )
        if not match:
            raise ValidationError("A dated archive snapshot is required for temporal evaluation.")
        months.append(match[1])
    if months[0] >= months[1]:
        raise ValidationError(
            "The external label snapshot must be later than the training snapshot."
        )
    return months

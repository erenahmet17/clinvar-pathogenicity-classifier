"""Step 3d: external validation on ClinGen erepo (evidence-code strata) and BAP1 SGE.

Models S1-S5 from v3_features_eval.py are retrained on July-2026 missense fit rows with
the external variants removed, under two exclusion designs:
  variant-disjoint : drop fit/calibration rows whose identity is in erepo or BAP1 SGE
  gene-disjoint    : additionally drop every row in a gene group that contains an
                     external gene (131 evaluated erepo missense genes + BAP1 = 132)
Evaluation:
  erepo missense P/LP vs B/LB, all / no computational code (PP3, BP4, BP7) /
  no computational and no frequency code (BA1, BS1, PM2); each also restricted to rare
  (gnomAD AF <= 1e-4 or absent).
  BAP1 SGE missense: abnormal vs normal (AUC) and Spearman rho of model score vs -score.

    python experiments/external_validation.py artifacts/experiments/external_validation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rare_missense_eval import GRID, auc_ci, scope_masks
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from v3_features_eval import SETS, SINGLE, matrix

from clinvar_classifier.inference import sigmoid
from clinvar_classifier.split import make_splits

ROOT = Path(__file__).resolve().parents[1]


def load_training(seed=42):
    earlier = pd.read_parquet(ROOT / "data/processed/2026-07/dataset.parquet")
    v3 = pd.read_parquet(ROOT / "data/processed/v3_scores.parquet")
    part = make_splits(earlier[earlier.label.notna()].copy(), seed)
    part = part.merge(v3, on="hgvs_id", how="left", validate="one_to_one")
    part["missense"], part["rare"], part["af_observed"] = scope_masks(part)
    return matrix(part)


def load_external():
    e = pd.read_parquet(ROOT / "data/external/erepo/erepo_snv_annotated.parquet")
    e = e[e.ext_status == "ok"].copy()
    e["missense"], e["rare"], e["af_observed"] = scope_masks(e)
    e = matrix(e[e.missense].copy())
    b = pd.read_parquet(ROOT / "data/external/mave/bap1_snv_annotated.parquet")
    b = b[(b.ext_status == "ok") & (b.aa_alt != "X") & (b.aa_ref != b.aa_alt)].copy()
    b = b[~b.aa_alt.str.contains(";")]
    b["rare"] = b.gnomad_af.isna() | (b.gnomad_af <= 1e-4)
    b = matrix(b)
    return e, b


def fit_model(fit, cal, cols, seed=42):
    cv = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            fit, fit.label.astype(int), groups=fit.gene_group
        )
    )
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
    search.fit(fit[cols], fit.label.astype(int))
    model = search.best_estimator_
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(
        model.predict(cal[cols], output_margin=True).reshape(-1, 1), cal.label.astype(int)
    )
    return lambda f: sigmoid(
        lr.coef_[0, 0] * model.predict(f[cols], output_margin=True) + lr.intercept_[0]
    )


def auc_entry(y, s):
    y, s = np.asarray(y).astype(int), np.asarray(s)
    if len(set(y)) < 2:
        return None
    return {
        "n": int(len(y)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "roc_auc_ci95": auc_ci(y, s),
    }


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    part = load_training()
    erepo, bap1 = load_external()
    ext_ids = set(erepo.hgvs_id) | set(bap1.hgvs_id)
    ext_genes = {g.upper() for g in erepo.GeneSymbol.dropna()} | {"BAP1"}
    row_genes = part.GeneSymbol.fillna("").str.upper().str.split(r"[;,|]")
    in_ext_gene = row_genes.map(lambda gs: bool({g.strip() for g in gs} & ext_genes))
    bad_groups = set(part.loc[in_ext_gene, "gene_group"])

    strata = {
        "erepo | all": erepo,
        "erepo | no computational code": erepo[~erepo.uses_computational],
        "erepo | no computational, no frequency code": erepo[
            ~erepo.uses_computational & ~erepo.uses_frequency
        ],
    }
    for name in list(strata):
        strata[name + " | rare"] = strata[name][strata[name].rare]
    labeled_bap1 = bap1[bap1.label.notna()]
    strata["BAP1 SGE | missense"] = labeled_bap1
    strata["BAP1 SGE | missense | rare"] = labeled_bap1[labeled_bap1.rare]

    report = {"eval_sets": {}, "single_score": {}, "designs": {}}
    for name, frame in strata.items():
        y = frame.label.astype(int)
        report["eval_sets"][name] = {
            "n": len(frame),
            "n_benign_or_normal": int((y == 0).sum()),
            "n_pathogenic_or_abnormal": int((y == 1).sum()),
        }
        report["single_score"][name] = {
            c: auc_entry(y[frame[c].notna()], frame.loc[frame[c].notna(), c])
            for c in SINGLE
            if frame[c].notna().sum() > 20
        }

    base = part[part.missense & ~part.hgvs_id.isin(ext_ids)]
    designs = {
        "variant-disjoint": base,
        "gene-disjoint": base[~base.gene_group.isin(bad_groups)],
    }
    for design, pool in designs.items():
        fit, cal = pool[pool.split == "fit"], pool[pool.split == "calibration"]
        entry = {"n_fit": len(fit), "n_calibration": len(cal), "models": {}}
        for key, cols in SETS.items():
            predict = fit_model(fit, cal, cols)
            res = {name: auc_entry(f.label, predict(f)) for name, f in strata.items()}
            scores = predict(bap1)
            rho = spearmanr(scores, -bap1.mave_score)
            res["BAP1 SGE | spearman(model, -score), all missense"] = {
                "n": int(len(bap1)),
                "rho": float(rho.statistic),
            }
            entry["models"][key] = res
            print(
                design,
                key,
                " ".join(
                    f"[{n}]={v['roc_auc']:.4f}" for n, v in res.items() if v and "roc_auc" in v
                ),
                f"rho={rho.statistic:.3f}",
                flush=True,
            )
        report["designs"][design] = entry
    (output / "external_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1])

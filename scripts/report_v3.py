"""Write docs/RUN_RESULTS_V3.md from a completed research-v3 bundle (measured values only)."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def ci(value):
    interval = value.get("roc_auc_ci_95_row_bootstrap")
    return "—" if not interval else f"{interval[0]:.4f}–{interval[1]:.4f}"


def main(bundle):
    bundle = Path(bundle)
    load = lambda name: json.loads((bundle / name).read_text(encoding="utf-8"))  # noqa: E731
    manifest, metrics = load("manifest.json"), load("metrics.json")
    external = (
        load("external_metrics.json") if (bundle / "external_metrics.json").exists() else None
    )
    lines = [
        f"# Koşu sonuçları — `{manifest['run_id']}` (S4, `{manifest['feature_contract']}`)",
        "",
        f"Oluşturma: {manifest['created_at'][:10]}. Bütün değerler bu bundle'daki dosyalardan okunur.",
        "",
        f"Kapsam: {manifest['scope']}.",
        "",
        f"Özellikler: `{'`, `'.join(manifest['features'])}`. **gnomAD AF modele girmez**; "
        "gerekçe ve kanıt: [experiments/CIRCULARITY_RESULTS.md](../experiments/CIRCULARITY_RESULTS.md).",
        "",
        f"Kapsam sayımları: {json.dumps(manifest.get('scope_counts', {}))}. "
        f"Fit: **{manifest['n_fit']:,}**, kalibrasyon: **{manifest['n_calibration']:,}**.",
        "",
        "## İç test (kalibre skor)",
        "",
        "| Partition | N | B/LB | P/LP | ROC-AUC | %95 GA (satır bootstrap) | AP | Brier |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name, result in metrics["test"].items():
        v = result["calibrated_score"]
        lines.append(
            f"| {name} | {v['n']:,} | {v['n_benign']:,} | {v['n_pathogenic']:,} | "
            f"{fmt(v['roc_auc'])} | {ci(v)} | {fmt(v['average_precision'])} | {fmt(v['brier'])} |"
        )
    lines += [
        "",
        "## Nadir ve AF gözlenen alt gruplar",
        "",
        "| Partition | Alt grup | N | B/LB | P/LP | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in metrics["test"].items():
        for key in [
            "rare_af_le_1e_4_or_unknown",
            "af_observed_le_1e_4",
            "af_unknown",
            "af_observed_gt_1e_3",
        ]:
            v = result["subgroups"].get(key)
            if v:
                lines.append(
                    f"| {name} | {key} | {v['n']:,} | {v['n_benign']:,} | {v['n_pathogenic']:,} | {fmt(v['roc_auc'])} |"
                )
    lines += [
        "",
        "## Tek skor karşılaştırması (test satırları)",
        "",
        "| Partition | "
        + " | ".join(metrics["test"]["variant_test"]["single_feature_baselines"])
        + " |",
        "|---|" + "---:|" * len(metrics["test"]["variant_test"]["single_feature_baselines"]),
    ]
    for name, result in metrics["test"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(fmt(v["roc_auc"]) for v in result["single_feature_baselines"].values())
            + " |"
        )
    folds = pd.DataFrame(metrics["development"]["fold_scores"])
    summary = folds.groupby("model").roc_auc.agg(["mean", "std"]).sort_values("mean")
    lines += [
        "",
        "## Geliştirme CV'si (yalnızca fit satırları, gen gruplu 5 kat)",
        "",
        "| Model / ablasyon | ROC-AUC ortalama | SD |",
        "|---|---:|---:|",
        *[f"| {i} | {r['mean']:.4f} | {r['std']:.4f} |" for i, r in summary.iterrows()],
        "",
        '"Reference only" satırı AF eklenmiş halidir; seçilemez, döngüsel katkının büyüklüğünü gösterir. '
        f"Seçilen parametreler: `{json.dumps(metrics['development']['best_params'])}`.",
    ]
    if external:
        lines += [
            "",
            "## Zaman bazlı dış kontrol (Temmuz → Ekim 2026)",
            "",
            "| Grup | N | B/LB | P/LP | ROC-AUC | %95 GA | Brier | Kapsam dışı (non-missense) |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for name, v in external["groups"].items():
            lines.append(
                f"| {name} | {v.get('n', 0):,} | {v.get('n_benign', 0):,} | {v.get('n_pathogenic', 0):,} | "
                f"{fmt(v.get('roc_auc'))} | {ci(v)} | {fmt(v.get('brier'))} | {v.get('outside_feature_contract_not_scored', 0):,} |"
            )
    if manifest.get("vus_counts"):
        lines += [
            "",
            "## VUS skorları",
            "",
            "```json",
            json.dumps(manifest["vus_counts"], indent=2),
            "```",
        ]
    lines += [
        "",
        "## Kaynak",
        "",
        f"Eğitim snapshot SHA-256: `{manifest['provenance']['cohort']['snapshot_sha256']}`",
        "",
        f"Dataset SHA-256: `{manifest['provenance']['dataset_sha256']}`",
        "",
        f"Anotasyon namespace: `{manifest['provenance']['annotation']['namespace']}`",
        "",
        "Bootstrap aralıkları gen kümelenmesini modellemez. Skorlar ClinVar sınıflarını değiştirmez; "
        "0.1/0.9 bantları ACMG/AMP kriteri değildir.",
    ]
    out = ROOT / "docs" / "RUN_RESULTS_V3.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "artifacts" / "research-v3")

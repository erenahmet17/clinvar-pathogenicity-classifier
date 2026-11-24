"""Thin Streamlit presentation; shared data contracts live in clinvar_classifier."""
# ruff: noqa: E402

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from clinvar_classifier.annotation import AnnotationClient, AnnotationError
from clinvar_classifier.features import CONTRACT_V3, FEATURES_V3, V3_BOUNDS, V3_REQUIRED
from clinvar_classifier.inference import Predictor
from clinvar_classifier.io import sha256
from clinvar_classifier.schemas import ValidationError

BUNDLE = Path(os.environ.get("CLINVAR_BUNDLE_DIR", str(ROOT / "artifacts" / "research-v3")))
st.set_page_config(page_title="ClinVar SNV Research", layout="wide")
st.title("🧬 ClinVar SNV Research & Prioritization")
st.info(
    "Research prioritization scores. These scores do not reclassify ClinVar VUS, estimate penetrance, or provide patient disease risk."
)


@st.cache_resource
def load_predictor(path, manifest_mtime):
    return Predictor(Path(path))


@st.cache_data
def read_artifact(path, expected_hash, columns=None):
    path = Path(path)
    if not path.exists() or sha256(path) != expected_hash:
        raise ValidationError(f"Artifact unavailable or checksum mismatch: {path.name}")
    if path.suffix == ".png":
        return path.read_bytes()
    return (
        pd.read_parquet(path, columns=columns)
        if path.suffix == ".parquet"
        else json.loads(path.read_text(encoding="utf-8"))
    )


def optional_artifact(name, columns=None):
    expected = predictor.manifest.get("optional_files", {}).get(name)
    if not expected:
        st.warning(f"This model bundle does not include {name}.")
        return None
    try:
        return read_artifact(str(BUNDLE / name), expected, columns)
    except (OSError, ValueError, ValidationError) as exc:
        st.warning(str(exc))
        return None


try:
    predictor = load_predictor(str(BUNDLE.resolve()), (BUNDLE / "manifest.json").stat().st_mtime_ns)
except (OSError, ValueError, ValidationError) as exc:
    st.error(f"Model bundle unavailable: {exc}")
    st.code("python -m clinvar_classifier.cli train DATASET MANIFEST artifacts/research-v3")
    st.stop()

manifest = predictor.manifest
V3 = manifest.get("feature_contract") == CONTRACT_V3
LABELS = {
    "esm1b": "ESM1b rankscore",
    "sift4g": "SIFT4G converted rankscore",
    "provean": "PROVEAN converted rankscore",
    "gerp_91way": "GERP 91 mammals",
    "phylop_100way": "phyloP 100way vertebrate",
    "phylop_470way": "phyloP 470way mammalian",
    "phastcons_100way": "phastCons 100way vertebrate",
    "alphamissense": "AlphaMissense rankscore",
    "cadd_phred": "CADD Phred",
}
with st.sidebar:
    st.header("Model scope")
    if V3:
        st.write("GRCh38 dbNSFP **missense** SNVs with required CADD/phyloP annotations.")
        st.write(
            "Population frequency is **not** a model input: ACMG/AMP frequency criteria "
            "(BA1/BS1/PM2) are part of how ClinVar labels are assigned, so it is shown "
            "separately as its own evidence line."
        )
    else:
        st.write("GRCh38 genomic SNVs with required dbNSFP CADD/phyloP annotations.")
        st.caption("AF unavailable ≠ AF zero. Zero in a source is not proof of adequate coverage.")
    st.write("Special penetrance and risk classifications require separate evidence.")
    st.metric("Model fitting rows", manifest.get("n_fit", "Unknown"))
    st.caption(f"Run: {manifest['run_id']}")
    st.write("[Repository](https://github.com/erenahmet17/clinvar-pathogenicity-classifier)")
    page = st.radio("Page", ["Manual input", "Variant search", "VUS explorer", "Performance"])


def render_prediction(raw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        result = predictor.predict(raw).iloc[0]
        st.metric("Model prioritization score", f"{result.model_score:.3f}")
        st.write(f"**Direction:** {result.direction}")
        st.caption(
            "Display bands: <0.1 benign direction; >0.9 pathogenic direction; intermediate scores remain indeterminate. Bands are not ACMG/AMP classes."
        )
        st.progress(float(result.model_score))
        features, contributions, base = predictor.explain(raw)
        st.subheader("Model explanation")
        st.caption(
            "Tree SHAP explains the uncalibrated raw log-odds margin. The displayed score may also apply independent sigmoid calibration. SHAP is not independent clinical evidence."
        )
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.barh(
            features.columns,
            contributions[0],
            color=["#bb5544" if v > 0 else "#4477aa" for v in contributions[0]],
        )
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.set_xlabel("Tree SHAP contribution to raw log-odds")
        axis.set_title(
            f"Base margin: {base[0]:.3f}; explained margin: {base[0] + contributions[0].sum():.3f}"
        )
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    except (ValidationError, ValueError) as exc:
        st.warning(str(exc))


def manual_v3():
    st.caption(
        "Enter dbNSFP 4.8a values for a missense SNV. Leave optional scores blank when "
        "unavailable; CADD and phyloP 100way are required. Population frequency is not used."
    )
    with st.form("manual_input"):
        values = {}
        for field in FEATURES_V3:
            low, high = V3_BOUNDS[field]
            default = {"cadd_phred": 25.0, "phylop_100way": 2.0}.get(field)
            values[field] = st.number_input(
                LABELS[field] + ("" if field in V3_REQUIRED else " (optional)"),
                min_value=float(low),
                max_value=float(high),
                value=default,
                format="%.5f",
            )
        submitted = st.form_submit_button("Score")
    if submitted:
        render_prediction(pd.DataFrame([values]))


if page == "Manual input" and V3:
    st.header("Manual feature input")
    manual_v3()

elif page == "Manual input":
    st.header("Manual feature input")
    st.caption(
        "Use matched GRCh38 sources within the SNV scope. Unknown AF is preserved explicitly."
    )
    with st.form("manual_input"):
        cadd = st.number_input("CADD Phred", min_value=0.0, max_value=99.0, value=25.0)
        phylop = st.number_input(
            "phyloP 100way vertebrate", min_value=-20.0, max_value=15.0, value=2.0
        )
        known = st.checkbox("A numeric gnomAD AF is available", value=False)
        af = st.number_input(
            "gnomAD AF (fraction, not percent)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.000001,
            format="%.8f",
        )
        submitted = st.form_submit_button("Score")
    if submitted:
        render_prediction(
            pd.DataFrame(
                [
                    {
                        "cadd_phred": cadd,
                        "phylop_100way": phylop,
                        "gnomad_af": af if known else None,
                        "af_available": known,
                    }
                ]
            )
        )

elif page == "Variant search":
    st.header("Variant search — GRCh38 genomic SNV")
    st.caption(
        "Example: chr11:g.126275389C>T. Transcript c./p. identifiers, indels and other assemblies are outside this input contract."
    )
    with st.form("variant_search"):
        hgvs = st.text_input("GRCh38 HGVS ID", placeholder="chr11:g.126275389C>T", key="hgvs_query")
        submitted = st.form_submit_button("Fetch and score")
    with st.expander("Documented limitation: HFE p.Cys282Tyr"):
        st.write(
            "chr6:g.26092913G>A is a pathogenic/low-penetrance HFE example, not a known benign control. Frequent pathogenic alleles, inheritance and penetrance need separate clinical evidence."
        )
        st.write("[ClinVar Variation 9](https://www.ncbi.nlm.nih.gov/clinvar/variation/9/)")
    if submitted:
        try:
            hgvs = predictor.check_variant_scope(hgvs)
            with st.spinner("Reading MyVariant hg38 annotations…"), AnnotationClient() as client:
                current = client.pin_metadata()
                expected = manifest.get("provenance", {}).get("annotation", {}).get("namespace")
                if expected and current["namespace"] != expected:
                    raise ValidationError(
                        "Live source metadata differs from this training bundle. Use matched sources or retrain; no score was produced."
                    )
                annotation = client.get(hgvs)
            if annotation.annotation_status != "ok":
                st.warning(
                    f"Required annotations unavailable: {annotation.annotation_status}. This does not establish a non-coding consequence."
                )
            elif V3:
                record = annotation.to_dict()
                st.json(
                    {LABELS[f]: record[f] for f in FEATURES_V3}
                    | {"consequence": record["consequence"]}
                )
                st.write("**Population frequency (separate evidence, not a model input)**")
                st.json(
                    {
                        "gnomAD genome AF": annotation.gnomad_af,
                        "AF available": annotation.af_available,
                        "AC": annotation.allele_count,
                        "AN": annotation.allele_number,
                    }
                )
                if annotation.consequence != "missense":
                    st.warning(
                        f"dbNSFP consequence is '{annotation.consequence}'. This model scores "
                        "missense SNVs only; no score produced."
                    )
                else:
                    render_prediction(pd.DataFrame([record]))
            else:
                st.json(
                    {
                        "CADD": annotation.cadd_phred,
                        "phyloP": annotation.phylop_100way,
                        "AF": annotation.gnomad_af,
                        "AF_available": annotation.af_available,
                        "AC": annotation.allele_count,
                        "AN": annotation.allele_number,
                    }
                )
                render_prediction(pd.DataFrame([annotation.to_dict()]))
        except (AnnotationError, ValidationError) as exc:
            st.error(str(exc))

elif page == "VUS explorer":
    st.header("VUS research scores")
    columns = [
        "hgvs_id",
        "GeneSymbol",
        "Name",
        "model_score",
        "direction",
        "gnomad_af",
        "af_available",
    ]
    vus = optional_artifact("vus_scores.parquet", columns)
    if vus is not None:
        gene = st.text_input("Gene text (literal substring)")
        band = st.selectbox(
            "Direction", ["All", "Pathogenic direction", "Benign direction", "Indeterminate"]
        )
        order = st.selectbox("Order", ["Highest score", "Lowest score"])
        mask = pd.Series(True, index=vus.index)
        if gene.strip():
            mask &= vus.GeneSymbol.str.contains(gene.strip(), case=False, na=False, regex=False)
        if band != "All":
            mask &= vus.direction.eq(band)
        filtered = vus.loc[mask]
        visible = (
            filtered.nlargest(50, "model_score")
            if order == "Highest score"
            else filtered.nsmallest(50, "model_score")
        )
        st.dataframe(visible, hide_index=True, width="stretch")
        st.caption(
            f"Showing {len(visible):,} of {len(filtered):,} matching variants; {len(vus):,} total scores. ClinVar VUS labels remain unchanged."
        )

else:
    st.header("Evaluation and source provenance")
    report = optional_artifact("metrics.json")
    if report is not None:
        if "evaluation.png" in manifest.get("optional_files", {}):
            figure = optional_artifact("evaluation.png")
            if figure is not None:
                st.image(figure, width="stretch")
        st.caption(
            "Selection used grouped fit-only CV; calibration had separate rows; variant and gene holdouts opened after selection. Row bootstrap intervals do not address gene clustering."
        )
        if V3:
            st.caption(
                "The development table includes a reference row with gnomAD AF added. It is "
                "never selectable: its gain reflects frequency criteria inside the labels. "
                "Independent checks (ClinGen erepo strata, BAP1 SGE) are in experiments/CIRCULARITY_RESULTS.md."
            )
        for name, result in report["test"].items():
            st.subheader(name.replace("_", " ").title())
            value = result["calibrated_score"]
            st.metric(
                "ROC-AUC",
                f"{value['roc_auc']:.4f}" if value["roc_auc"] is not None else "Unavailable",
            )
            st.dataframe(
                pd.DataFrame(
                    value["confusion_matrix"],
                    index=["Actual benign", "Actual pathogenic"],
                    columns=["Predicted benign", "Predicted pathogenic"],
                ),
                width="stretch",
            )
            calibration = pd.DataFrame(value["calibration_bins"])
            st.dataframe(calibration, hide_index=True, width="stretch")
            st.line_chart(calibration.set_index("mean_score")[["positive_fraction"]])
            st.json(
                {
                    "brier": value["brier"],
                    "log_loss": value["log_loss"],
                    "auc_ci": value.get("roc_auc_ci_95_row_bootstrap"),
                    "subgroups": result["subgroups"],
                    "single_feature_baselines": result["single_feature_baselines"],
                }
            )
        folds = pd.DataFrame(report["development"]["fold_scores"])
        st.subheader("Grouped development CV and feature ablation")
        st.dataframe(folds.groupby("model").roc_auc.agg(["mean", "std"]), width="stretch")
        st.caption(
            "Selected-parameter CV is subject to search optimism. Similar averages do not establish statistical equivalence."
        )
    external = optional_artifact("external_metrics.json")
    if external is not None:
        st.subheader("Temporal external evaluation")
        st.json(external)
    with st.expander("Source and model manifest"):
        st.json(manifest)

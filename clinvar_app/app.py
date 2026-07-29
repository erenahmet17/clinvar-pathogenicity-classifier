import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import pickle

st.set_page_config(page_title="ClinVar Variant Classifier", layout="wide")
st.title("🧬 ClinVar Variant Pathogenicity Classifier")
with st.sidebar:
    st.header("📌 About This Project")
    st.write("""
    This app uses an XGBoost model trained on ClinVar variant data
    to predict the pathogenicity of genetic variants.
    """)
    st.metric("Test ROC-AUC", "0.996")
    st.metric("Training Data", "79,586 variants")
    st.divider()
    st.write("**Features Used:**")
    st.write("- CADD Phred Score")
    st.write("- phyloP Conservation Score")
    st.write("- gnomAD Population Frequency")
    st.divider()
    st.write("[GitHub Repo](https://github.com/erenahmet17)")
st.write("Predicts the pathogenicity (harmfulness) of genetic variants using an XGBoost model "
         "based on CADD, phyloP, and gnomAD frequency data.")


@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")


model = load_model()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Manual Feature Input", "🔍 Search Variant (HGVS ID)",
                                  "🧬 VUS Explorer", "📈 Model Performance"])

with tab1:
    st.header("Manual Feature Input")
    st.write("Enter CADD, phyloP, and gnomAD frequency values manually to get a prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        cadd_input = st.number_input("CADD Phred Score", min_value=0.0, max_value=99.0, value=25.0, step=0.1,
                                     help="Higher value = predicted to be more harmful")
    with col2:
        phylop_input = st.number_input("phyloP (100way vertebrate)", min_value=-20.0, max_value=15.0, value=2.0,
                                       step=0.1,
                                       help="Higher value = more evolutionarily conserved region")
    with col3:
        gnomad_input = st.number_input("gnomAD Population Frequency", min_value=0.0, max_value=1.0, value=0.0,
                                       step=0.0001, format="%.6f",
                                       help="0 = variant never observed in the population")

    if st.button("Predict", key="manual_predict"):
        gnomad_log = np.log10(gnomad_input + 1e-9)
        X_input = pd.DataFrame([[cadd_input, phylop_input, gnomad_log]],
                               columns=["cadd_phred", "phylop_100way", "gnomad_af_log"])

        proba = model.predict_proba(X_input)[0, 1]
        label = "Pathogenic" if proba > 0.5 else "Benign"

        st.subheader("Result")
        if proba > 0.5:
            st.error(f"⚠️ Prediction: **{label}** (Pathogenic probability: {proba:.1%})")
        else:
            st.success(f"✅ Prediction: **{label}** (Pathogenic probability: {proba:.1%})")


        def render_gauge(proba):
            color = "#ff4b4b" if proba > 0.5 else "#00cc66"
            st.markdown(f"""
            <div style="background-color:#333; border-radius:10px; height:30px; width:100%;">
                <div style="background-color:{color}; width:{proba * 100}%; height:100%; 
                            border-radius:10px; text-align:center; color:white; line-height:30px;">
                    {proba:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)


        render_gauge(proba)

        st.subheader("Why this prediction (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(8, 3))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=X_input.columns.tolist()
        ), show=False)
        st.pyplot(fig)

with tab2:
    st.header("Variant Search (HGVS ID)")
    st.write("Enter an HGVS ID, or try one of the examples below.")

    if "hgvs_query" not in st.session_state:
        st.session_state.hgvs_query = ""

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("🔴 Known Pathogenic example"):
            st.session_state.hgvs_query = "chr11:g.126275389C>T"
    with ex_col2:
        if st.button("🟢 Known Benign example"):
            st.session_state.hgvs_query = "chr6:g.26092913G>A"
    with ex_col3:
        if st.button("🧹 Clear"):
            st.session_state.hgvs_query = ""

    hgvs_query = st.text_input("Enter HGVS ID", placeholder="chr11:g.126275389C>T", key="hgvs_query")

    if st.button("Search and Predict", key="hgvs_predict"):
        if not hgvs_query:
            st.warning("Please enter an HGVS ID.")
        else:
            with st.spinner("Fetching data from myvariant.info..."):
                try:
                    r = requests.get(
                        f"https://myvariant.info/v1/variant/{hgvs_query}",
                        params={"assembly": "hg38",
                                "fields": "dbnsfp.cadd.phred,gnomad_genome.af.af,dbnsfp.phylop.100way_vertebrate"}
                    )
                    r.raise_for_status()
                    data = r.json()

                    if isinstance(data, list) or data.get("notfound"):
                        st.error("This variant was not found on myvariant.info.")
                    else:
                        dbnsfp = data.get("dbnsfp", {})
                        cadd = dbnsfp.get("cadd", {}).get("phred")
                        phylop_data = dbnsfp.get("phylop", {}).get("100way_vertebrate", {})
                        phylop = phylop_data.get("score") if isinstance(phylop_data, dict) else None
                        gnomad_af = data.get("gnomad_genome", {}).get("af", {}).get("af", 0)

                        if cadd is None or phylop is None:
                            st.warning("CADD/phyloP data is not available for this variant "
                                       "(likely a non-coding region variant).")
                        else:
                            gnomad_log = np.log10((gnomad_af or 0) + 1e-9)
                            X_input = pd.DataFrame([[cadd, phylop, gnomad_log]],
                                                   columns=["cadd_phred", "phylop_100way", "gnomad_af_log"])

                            proba = model.predict_proba(X_input)[0, 1]
                            label = "Pathogenic" if proba > 0.5 else "Benign"

                            st.subheader("Retrieved Values")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("CADD Phred", f"{cadd:.2f}")
                            c2.metric("phyloP", f"{phylop:.3f}")
                            c3.metric("gnomAD AF", f"{gnomad_af:.6f}" if gnomad_af else "0 (not observed)")

                            st.subheader("Result")
                            if proba > 0.5:
                                st.error(f"⚠️ Prediction: **{label}** (Pathogenic probability: {proba:.1%})")
                            else:
                                st.success(f"✅ Prediction: **{label}** (Pathogenic probability: {proba:.1%})")


                            def render_gauge(proba):
                                color = "#ff4b4b" if proba > 0.5 else "#00cc66"
                                st.markdown(f"""
                                <div style="background-color:#333; border-radius:10px; height:30px; width:100%;">
                                    <div style="background-color:{color}; width:{proba * 100}%; height:100%; 
                                                border-radius:10px; text-align:center; color:white; line-height:30px;">
                                        {proba:.1%}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)


                            render_gauge(proba)

                            st.subheader("Why this prediction (SHAP)")
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_input)

                            fig, ax = plt.subplots(figsize=(8, 3))
                            shap.plots.waterfall(shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=X_input.iloc[0],
                                feature_names=X_input.columns.tolist()
                            ), show=False)
                            st.pyplot(fig)

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")

with tab3:
    st.header("VUS Explorer")
    st.write("Variants labeled 'Uncertain Significance' in ClinVar, "
             "for which the model offers a strong directional prediction.")


    @st.cache_data
    def load_vus():
        return pd.read_csv("vus_final_checkpoint.csv")


    vus_display = load_vus()

    col1, col2 = st.columns([2, 1])
    with col1:
        gene_filter = st.text_input("Filter by gene name (optional)", "")
    with col2:
        direction = st.selectbox("Show:", ["Highest Pathogenic probability", "Highest Benign probability"])

    filtered = vus_display.copy()
    if gene_filter:
        filtered = filtered[filtered["GeneSymbol"].str.contains(gene_filter, case=False, na=False)]

    ascending = direction == "Highest Benign probability"
    filtered = filtered.sort_values("predicted_proba_pathogenic", ascending=ascending)

    st.dataframe(
        filtered[["hgvs_id", "GeneSymbol", "Name", "predicted_proba_pathogenic"]].head(50).style.format({
            "predicted_proba_pathogenic": "{:.6f}"
        }),
        use_container_width=True
    )
    st.caption(f"Showing {len(filtered)} variants (filtered).")

with tab4:
    st.header("Model Performance")

    st.subheader("Algorithm Comparison (5-fold CV)")
    comparison_df = pd.read_csv("model_comparison.csv", index_col=0)
    comparison_df.columns = ["Mean ROC-AUC"]
    comparison_df = comparison_df.sort_values("Mean ROC-AUC", ascending=True)
    st.bar_chart(comparison_df)

    st.divider()

    test_results = pd.read_csv("test_predictions.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(test_results["y_test"], test_results["y_pred"])
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Benign", "Pathogenic"], y=["Benign", "Pathogenic"],
            text=cm, texttemplate="%{text}", colorscale="Blues"
        ))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(test_results["y_test"], test_results["y_proba"])
        roc_auc = auc(fpr, tpr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.4f})"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash")))
        fig2.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Overall Feature Importance (SHAP)")
    with open("shap_data.pkl", "rb") as f:
        shap_data = pickle.load(f)

    fig3, ax = plt.subplots()
    shap.summary_plot(shap_data["shap_values"], shap_data["X_test"], show=False)
    st.pyplot(fig3)

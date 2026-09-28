"""Additive DWT result and saved-dataset views for the Streamlit application."""
from html import escape

import pandas as pd
import streamlit as st

from FRONTEND.utils import get_feature_table


def render_dwt_result(result):
    st.markdown("## DWT Classification Result")
    meta = result.get("metadata", {})
    models = [("dwt_svm", "Support Vector Machine (SVM)", "Class Probability"),
              ("dwt_knn", "K-Nearest Neighbors (KNN)", "Posterior Class Probability")]
    cards = []
    for key, label, score_label in models:
        data = result.get(key)
        if data:
            prediction = escape(data["prediction"].upper())
            class_name = data["prediction"].lower()
            pill = f"class-pill-{class_name}" if class_name in ("glioma", "meningioma", "pituitary") else "verif-badge"
            cards.append(
                f'<div class="result-box"><div class="model-label">{escape(data["model_name"])}</div>'
                f'<div><span class="{pill}">{prediction}</span></div>'
                f'<div class="confidence-score">{data["score"]:.2f}%</div>'
                f'<div style="font-size: 0.8rem; color: #64748b;">{score_label}</div></div>')
        else:
            cards.append(f'<div class="result-box"><div class="model-label">{label}</div>'
                         '<div>Result unavailable</div></div>')
    st.markdown(
        '<div class="classification-container">'
        '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">'
        + "".join(cards) + '</div><div class="verification-row">'
        f'<span class="verif-badge">DWT Features: {len(meta.get("dwt_feature_values", []))}</span>'
        f'<span class="verif-badge">PCA Components: {len(meta.get("dwt_pca_components", []))}</span>'
        '</div></div>', unsafe_allow_html=True)
    if not all(result.get(key) for key, _, _ in models):
        st.warning(meta.get("dwt_error") or "DWT results are unavailable from this backend. Run analysis again after restarting the backend.")
    for column, (key, label, _) in zip(st.columns(2), models):
        with column:
            st.markdown(f"**DWT {label} Class Probabilities**")
            data = result.get(key)
            if data and data.get("probabilities"):
                st.bar_chart(pd.DataFrame({"Class": list(data["probabilities"]),
                                          "Probability (%)": list(data["probabilities"].values())}).set_index("Class"),
                             height=180)
    st.caption("Features below belong to the current scan's extracted tumor ROI.")
    with st.expander("DWT Feature Analysis", expanded=True):
        st.markdown("**DWT Feature Table**")
        if meta.get("dwt_feature_values"):
            st.dataframe(pd.DataFrame({"Feature": meta["dwt_feature_names"],
                                       "Value": meta["dwt_feature_values"]}),
                         use_container_width=True, height=320)
        else:
            st.info("No DWT features are available for this scan.")
        st.markdown("**DWT + PCA Table**")
        if meta.get("dwt_pca_components"):
            st.dataframe(pd.DataFrame({"Principal Component": meta["dwt_pca_feature_names"],
                                       "Score Value": meta["dwt_pca_components"]}),
                         use_container_width=True)
        else:
            st.info("DWT PCA coordinates are unavailable.")


def render_reference_tables():
    with st.expander("Saved Feature Datasets (Train / Test)", expanded=False):
        st.caption("Reference dataset rows, not predictions or features for the uploaded scan.")
        split = st.selectbox("Dataset split", ["test", "train"])
        page = int(st.number_input("Dataset page (25 rows)", min_value=1, value=1, step=1))
        if st.checkbox("Load saved feature tables"):
            for label, key in (("GLCM", "glcm"), ("GLCM + PCA", "glcm_pca"),
                               ("DWT", "dwt"), ("DWT + PCA", "dwt_pca")):
                st.markdown(f"**{label}**")
                try:
                    with st.spinner(f"Loading {label} table..."):
                        table = get_feature_table(key, split, (page - 1) * 25)
                    st.caption(f'{table["source"]} — {table["total"]} rows')
                    st.dataframe(pd.DataFrame(table["rows"], columns=table["columns"]),
                                 use_container_width=True)
                except Exception as error:
                    st.warning(f"Could not load {label} table: {error}")

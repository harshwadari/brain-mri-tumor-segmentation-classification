"""Final result appended after all existing MRI analysis sections."""
from html import escape

import pandas as pd
import streamlit as st


def render_ensemble_result(result):
    st.markdown("## Final Ensemble Prediction")
    ensemble = result.get("ensemble")
    if not ensemble:
        st.info("Run Analyze Image again with the updated backend to obtain the ensemble prediction.")
        return
    st.caption(f"Voting method: {ensemble['voting_method']}")
    if ensemble.get("status") != "success":
        st.warning(ensemble.get("error") or "All four model results are required for the ensemble.")
        return
    st.markdown(
        '<div class="classification-container"><div class="result-box">'
        f'<div class="model-label">Final predicted class: {escape(ensemble["prediction"].title())}</div>'
        f'<div class="confidence-score">{ensemble["score"]:.2f}%</div>'
        '<div>Final confidence</div></div></div>', unsafe_allow_html=True)
    for column, label in zip(st.columns(3), ("glioma", "meningioma", "pituitary")):
        column.metric(f"{label.title()} probability", f'{ensemble["probabilities"][label]:.2f}%')
    names = {"svm": "GLCM + SVM", "knn": "GLCM + KNN",
             "dwt_svm": "DWT + SVM", "dwt_knn": "DWT + KNN"}
    st.dataframe(pd.DataFrame([
        {"Model": label, "Prediction": ensemble["model_predictions"][key].title(),
         "Weight (%)": ensemble["weights"][key] * 100}
        for key, label in names.items()
    ]), use_container_width=True, hide_index=True)
    st.caption(ensemble["weight_source"])

import sys
import time
from pathlib import Path
from io import BytesIO
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to sys.path so imports work seamlessly regardless of launch directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from FRONTEND.utils import (
        API_BASE_URL,
        check_backend_health,
        get_backend_info,
        get_sample_images,
        get_sample_image_bytes,
        predict_image,
        base64_to_pil,
    )
except ImportError:
    from utils import (
        API_BASE_URL,
        check_backend_health,
        get_backend_info,
        get_sample_images,
        get_sample_image_bytes,
        predict_image,
        base64_to_pil,
    )

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Analysis | AI Medical Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Dashboard CSS
st.markdown(
    """
    <style>
    /* Global Typography and Layout */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px -2px rgba(2, 132, 199, 0.25);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
        text-align: center;
    }
    .header-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.92;
        letter-spacing: 0.2px;
    }

    /* Pipeline Status Flow Cards */
    .pipeline-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem 1.7rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .pipeline-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #475569;
        letter-spacing: 0.3px;
    }
    .pipeline-stage-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 500;
        margin: 0.2rem 0.25rem;
    }
    .pipeline-line {
        width: 65%;
        height: 4px;
        margin: 1rem auto 0.8rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #3b82f6 var(--progress), #e2e8f0 var(--progress));
    }
    .pipeline-status {
        font-size: 0.82rem;
        color: #64748b;
    }
    .stage-completed {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .stage-active {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }gf
    .stage-pending {
        background: #f8fafc;
        color: #94a3b8;
        border: 1px solid #e2e8f0;
    }

    /* Clinical Classification Card */
    .classification-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .result-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .model-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .class-pill-glioma {
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.25rem;
        border: 1px solid #fca5a5;
    }
    .class-pill-meningioma {
        background: #fef3c7;
        color: #92400e;
        font-weight: 700;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.25rem;
        border: 1px solid #fcd34d;
    }
    .class-pill-pituitary {
        background: #e0e7ff;
        color: #3730a3;
        font-weight: 700;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.25rem;
        border: 1px solid #c7d2fe;
    }
    .confidence-score {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.4rem;
    }

    /* Verification Badge Row */
    .verification-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    .verif-badge {
        background: #f1f5f9;
        color: #334155;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.83rem;
        font-weight: 500;
        border: 1px solid #cbd5e1;
    }

    /* Image Grid Cards */
    .image-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    .image-caption {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 0.4rem;
    }
    .image-subtext {
        font-size: 0.76rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: System Status & Demo Controls
with st.sidebar:
    st.title("System Control")
    
    # Check backend connectivity
    is_online = check_backend_health(API_BASE_URL)
    if is_online:
        st.success("● FastAPI Backend: Connected (Port 8000)")
    else:
        st.error("● FastAPI Backend: Offline")
        st.caption(f"Run `uvicorn BACKEND.main:app --port 8000` to start backend.")

    st.divider()

    # Load Model Metadata
    info = get_backend_info(API_BASE_URL) if is_online else None
    if info:
        st.markdown("**Deployed Pipeline Specification**")
        st.caption(f"• **Segmentation**: U-Net (256×256)")
        st.caption(f"• **Texture Analysis**: GLCM (96 features)")
        st.caption(f"• **Reduction**: PCA (5 components)")
        st.caption(f"• **Classifiers**: SVM + KNN (k=5)")
        st.caption(f"• **Target Classes**: `glioma`, `meningioma`, `pituitary`")
    
    st.divider()

    # Sample Image Quick-Demo Picker
    st.markdown("### Quick-Demo Samples")
    st.caption("Select a validated test scan from the official evaluation dataset:")

    selected_sample_bytes = None
    selected_sample_name = None

    if is_online:
        sample_dict = get_sample_images(API_BASE_URL)
        if sample_dict:
            chosen_class = st.selectbox("Select Tumor Class", options=list(sample_dict.keys()))
            chosen_file = st.selectbox("Select MRI Scan", options=sample_dict[chosen_class])
            
            if st.button("Load Selected Sample MRI", use_container_width=True):
                with st.spinner("Fetching test scan..."):
                    img_bytes = get_sample_image_bytes(chosen_class, chosen_file, API_BASE_URL)
                    if img_bytes:
                        st.session_state["loaded_bytes"] = img_bytes
                        st.session_state["loaded_filename"] = chosen_file
                        st.session_state["analysis_result"] = None
                        st.success(f"Loaded: {chosen_file}")
                    else:
                        st.error("Failed to load sample.")
    else:
        st.caption("Connect backend to view test dataset samples.")

    st.divider()
    st.caption("B.E. Major Project Deployment")

# Main Header
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">
            Brain Tumor Analysis System
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload Section
col_upload, col_preview = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown("### 1. Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image.",
    )

    if uploaded_file is not None:
        # User uploaded a new file
        st.session_state["loaded_bytes"] = uploaded_file.read()
        st.session_state["loaded_filename"] = uploaded_file.name
        # Invalidate previous result if file changes
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state["analysis_result"] = None
            st.session_state["last_uploaded_name"] = uploaded_file.name

with col_preview:
    st.markdown("### 2. Selected Scan Preview")
    current_bytes = st.session_state.get("loaded_bytes")
    current_filename = st.session_state.get("loaded_filename", "No image selected")

    if current_bytes:
        try:
            pil_img = Image.open(BytesIO(current_bytes))
            st.image(pil_img, caption=f"Active: {current_filename}", width=220)
        except Exception:
            st.warning("Could not render image preview.")
    else:
        st.info("Upload an MRI scan or choose a Quick-Demo sample from the sidebar.")

# Analyze Trigger Button
st.markdown("---")
analyze_col, info_col = st.columns([2, 4])

with analyze_col:
    analyze_clicked = st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=(current_bytes is None or not is_online),
    )

with info_col:
    if not is_online:
        st.warning("FastAPI backend is offline. Please launch the backend server.")
    elif current_bytes is None:
        st.info("Please select or upload an image to begin automated analysis.")

# Execution and Dynamic Progress Animation
PIPELINE_STAGES = [
    ("Image Validation & Decode", "Validating image headers, dimension checks, and decoding raw stream"),
    ("MRI Preprocessing", "Converting to grayscale, resizing to 256×256, 3×3 Gaussian denoising & Z-score normalization"),
    ("U-Net Deep Learning Segmentation", "Generating pixel-wise tumor probability map using trained U-Net (threshold 0.5)"),
    ("Connected Component QC & Morphology", "Removing small disconnected artifacts (<50 px) & applying 3×3 elliptical closing"),
    ("Tumor ROI Extraction & Visual Blend", "Extracting tumor region and generating semi-transparent overlay with yellow boundary"),
    ("GLCM Texture Feature Extraction", "Cropping bounding box, quantizing to 32 gray levels & extracting 96 Haralick features"),
    ("Standardization & PCA Projection", "Standardizing features and projecting to 5 principal components (95% variance)"),
    ("Dual-Model SVM + KNN Classification", "Running tuned Support Vector Machine and K-Nearest Neighbors inference"),
]

if analyze_clicked and current_bytes:
    # Render Step-by-Step Progress Pipeline visualizer
    progress_placeholder = st.empty()
    status_text = st.empty()

    # Animate stages to give visual clarity matching the requested reference design
    for idx, (stage_name, stage_desc) in enumerate(PIPELINE_STAGES):
        pct = int(((idx + 1) / len(PIPELINE_STAGES)) * 100)

        # Build visual badges
        badges_html = "<div class='pipeline-container'><div class='pipeline-title'>ANALYZING IMAGE</div>"
        for j, (name, _) in enumerate(PIPELINE_STAGES):
            if j < idx:
                badges_html += f"<span class='pipeline-stage-badge stage-completed'>✓ {name}</span>"
            elif j == idx:
                badges_html += f"<span class='pipeline-stage-badge stage-active'>● {name} (Running)</span>"
            else:
                badges_html += f"<span class='pipeline-stage-badge stage-pending'>○ {name}</span>"
        badges_html += f"<div class='pipeline-line' style='--progress: {pct}%;'></div>"
        badges_html += f"<div class='pipeline-status'>{stage_desc}...</div></div>"
        progress_placeholder.markdown(badges_html, unsafe_allow_html=True)
        time.sleep(0.12)  # Smooth animation step

    # Actual backend call
    try:
        with st.spinner("Completing deep learning forward pass and feature extraction..."):
            result = predict_image(
                current_bytes,
                filename=current_filename,
                base_url=API_BASE_URL,
            )
            st.session_state["analysis_result"] = result
            status_text.empty()

            # Final completed badges
            badges_html = "<div class='pipeline-container'><div class='pipeline-title'>ANALYSIS COMPLETED</div>"
            for name, _ in PIPELINE_STAGES:
                badges_html += f"<span class='pipeline-stage-badge stage-completed'>✓ {name}</span>"
            badges_html += "<div class='pipeline-line' style='--progress: 100%;'></div>"
            badges_html += "<div class='pipeline-status'>All stages completed successfully.</div></div>"
            progress_placeholder.markdown(badges_html, unsafe_allow_html=True)

    except Exception as err:
        status_text.error(f"Analysis failed: {str(err)}")
        st.session_state["analysis_result"] = None

# Results Section
result = st.session_state.get("analysis_result")
if result:
    st.markdown("## Classification Result")

    svm_data = result["svm"]
    knn_data = result["knn"]
    meta = result["metadata"]
    images = result["images"]

    # Helper function for class badge styling
    def get_class_pill(class_name: str):
        c = class_name.lower()
        if "glioma" in c:
            return f'<span class="class-pill-glioma">GLIOMA</span>'
        elif "meningioma" in c:
            return f'<span class="class-pill-meningioma">MENINGIOMA</span>'
        elif "pituitary" in c:
            return f'<span class="class-pill-pituitary">PITUITARY</span>'
        else:
            return f'<span class="verif-badge">{class_name.upper()}</span>'

    # 1. Dual Classification Result Card
    st.markdown(
        f"""
        <div class="classification-container">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div class="result-box">
                    <div class="model-label">Support Vector Machine (SVM)</div>
                    <div>{get_class_pill(svm_data['prediction'])}</div>
                    <div class="confidence-score">{svm_data['score']:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.2rem;">Class Probability</div>
                </div>
                <div class="result-box">
                    <div class="model-label">K-Nearest Neighbors (KNN, k=5)</div>
                    <div>{get_class_pill(knn_data['prediction'])}</div>
                    <div class="confidence-score">{knn_data['score']:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.2rem;">Posterior Class Probability</div>
                </div>
            </div>
            <div class="verification-row">
                <span class="verif-badge">✓ GLCM Features Extracted: Yes (96)</span>
                <span class="verif-badge">✓ PCA Applied: Yes (5 Components)</span>
                <span class="verif-badge">✓ Segmentation: U-Net Deep Learning (256×256)</span>
                <span class="verif-badge">✓ Tumor Pixels: {meta['tumor_pixel_count']:,} ({meta['tumor_area_percentage']}%)</span>
                <span class="verif-badge">✓ Latency: {meta['execution_time_ms']} ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Probability Distribution Breakdown
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.markdown("**SVM Class Probabilities**")
        if svm_data.get("probabilities"):
            svm_df = pd.DataFrame({
                "Class": list(svm_data["probabilities"].keys()),
                "Probability (%)": list(svm_data["probabilities"].values()),
            }).set_index("Class")
            st.bar_chart(svm_df, height=180)
        else:
            st.info("The saved SVM does not expose probability values.")

    with prob_col2:
        st.markdown("**KNN Posterior Class Probabilities**")
        knn_df = pd.DataFrame({
            "Class": list(knn_data["probabilities"].keys()),
            "Probability (%)": list(knn_data["probabilities"].values()),
        }).set_index("Class")
        st.bar_chart(knn_df, height=180)

    # 3. 5-Image Pipeline Visualization Grid
    st.markdown("### 5-Stage Image Processing Pipeline")
    st.caption("Visual verification of intermediate transformations directly generated by the inference backend:")

    img_col1, img_col2, img_col3, img_col4, img_col5 = st.columns(5)

    with img_col1:
        st.markdown(
            """
            <div class="image-card">
                <div class="image-caption">Image 1 — Original</div>
                <div class="image-subtext">Input MRI Slice</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(base64_to_pil(images["original"]), use_container_width=True)

    with img_col2:
        st.markdown(
            """
            <div class="image-card">
                <div class="image-caption">Image 2 — Preprocessed</div>
                <div class="image-subtext">Gaussian + Z-Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(base64_to_pil(images["preprocessed"]), use_container_width=True)

    with img_col3:
        st.markdown(
            """
            <div class="image-card">
                <div class="image-caption">Image 3 — Segmentation</div>
                <div class="image-subtext">U-Net Predicted Mask</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(base64_to_pil(images["segmentation_mask"]), use_container_width=True)

    with img_col4:
        st.markdown(
            """
            <div class="image-card">
                <div class="image-caption">Image 4 — Extracted ROI</div>
                <div class="image-subtext">Tumor Intensity Masking</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(base64_to_pil(images["extracted_roi"]), use_container_width=True)

    with img_col5:
        st.markdown(
            """
            <div class="image-card">
                <div class="image-caption">Image 5 — ROI Overlay</div>
                <div class="image-subtext">Red Blend + Yellow Contour</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(base64_to_pil(images["roi_overlay"]), use_container_width=True)

    # 4. Detailed Diagnostic & Feature Inspection (Expandable)
    with st.expander("🔍 Detailed Diagnostic & Feature Analysis (Click to Expand)", expanded=False):
        f_col1, f_col2 = st.columns([1, 1])

        with f_col1:
            st.markdown("**Principal Component Analysis (PCA) Coordinates**")
            pca_df = pd.DataFrame({
                "Principal Component": [f"PC{i+1}" for i in range(len(meta["pca_components"]))],
                "Score Value": meta["pca_components"],
            })
            st.dataframe(pca_df, use_container_width=True)

        with f_col2:
            st.markdown("**Segmentation Morphometrics**")
            st.write({
                "Total Slice Resolution": "256 × 256 (65,536 px)",
                "Detected Tumor Pixels": f"{meta['tumor_pixel_count']} px",
                "Tumor Area Percentage": f"{meta['tumor_area_percentage']} %",
                "Morphological Closing": "3×3 Elliptical Kernel",
                "Component Filtering": "Minimum Area ≥ 50 px",
                "U-Net Threshold": "0.5 (Sigmoid output)",
            })

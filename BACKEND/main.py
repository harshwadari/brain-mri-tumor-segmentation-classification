import time
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from BACKEND.config import (
    CLASS_NAMES,
    DATASET_DIR,
    IMG_SIZE,
)
from BACKEND.preprocessing import preprocess_mri, resize_image, convert_to_grayscale
from BACKEND.segmentation import extract_tumor_roi, create_roi_overlay
from BACKEND.features import extract_glcm_features
from BACKEND.models import model_manager
from BACKEND.schemas import (
    PredictionResponse,
    PipelineImages,
    ClassificationModelResult,
    PipelineMetadata,
    PipelineStage,
    SystemInfoResponse,
)
from BACKEND.utils import decode_image_bytes, encode_image_to_base64


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models once into memory
    print("==================================================")
    print("INITIALIZING BRAIN TUMOR ANALYSIS BACKEND SERVICE")
    print("==================================================")
    try:
        model_manager.load_all_models()
    except Exception as e:
        print(f"Error during model loading: {e}")
    yield
    print("Shutting down Brain Tumor Analysis Service.")


app = FastAPI(
    title="Brain Tumor Analysis API",
    description="FastAPI Backend for MRI Segmentation (U-Net) and Classification (GLCM + PCA + SVM / KNN)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Brain Tumor Analysis API",
        "models_ready": model_manager.is_loaded,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded,
    }


@app.get("/api/info", response_model=SystemInfoResponse, tags=["Metadata"])
async def get_system_info():
    return SystemInfoResponse(
        status="online",
        service="Brain Tumor Analysis API",
        version="1.0.0",
        target_classes=model_manager.classes,
        input_resolution=f"{IMG_SIZE}x{IMG_SIZE}",
        models_loaded=model_manager.is_loaded,
        classifier_notes={
            "svm": "Saved SVC supports predict_proba; scores are class probabilities.",
            "knn": "Saved KNN supports predict_proba; scores are posterior class probabilities.",
        },
    )


@app.get("/api/sample-images", tags=["Samples"])
async def get_sample_images():
    """
    Returns available test images from the real dataset for quick demonstration.
    """
    test_dir = DATASET_DIR / "Classification" / "test"
    samples = {}

    if test_dir.exists():
        for cls in CLASS_NAMES:
            cls_dir = test_dir / cls
            if cls_dir.exists():
                files = sorted([f.name for f in cls_dir.glob("*.jpg")])[:10]
                samples[cls] = files
    return samples


@app.get("/api/sample-image/{class_name}/{filename}", tags=["Samples"])
async def get_sample_image_file(class_name: str, filename: str):
    """
    Retrieve raw image bytes of a specific test sample from the dataset.
    """
    target_path = DATASET_DIR / "Classification" / "test" / class_name / filename
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    with open(target_path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_brain_mri(file: UploadFile = File(...)):
    """
    Execute the end-to-end brain MRI analysis pipeline:
    1. Decode and validate MRI
    2. Exact preprocessing (Resize, Gaussian Denoising, Z-score normalization)
    3. U-Net segmentation (threshold 0.5)
    4. Post-processing (Connected Component QC + Morphological Closing)
    5. Tumor ROI extraction & red-blend overlay with yellow contour
    6. 96 GLCM feature extraction from cropped ROI
    7. Standard scaling & PCA projection (5 components)
    8. Dual classification via SVM & KNN
    """
    start_time = time.perf_counter()

    # 1. Read and validate uploaded file
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        raw_image = decode_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    stages_executed = []
    stages_executed.append(PipelineStage(
        name="Image Validation & Loading",
        description=f"Decoded {raw_image.shape} MRI image successfully."
    ))

    # 2. MRI Preprocessing
    # Keep a resized original (256x256) for visual comparison
    orig_gray = convert_to_grayscale(raw_image)
    orig_resized = resize_image(orig_gray, (IMG_SIZE, IMG_SIZE))
    preprocessed = preprocess_mri(raw_image)

    stages_executed.append(PipelineStage(
        name="MRI Preprocessing",
        description="Applied Grayscale -> 256x256 Resize -> Gaussian Denoising (3x3) -> Z-Score Normalization."
    ))

    # 3. U-Net Segmentation & Post-processing
    try:
        prob_map, raw_mask, cleaned_mask, refined_mask = model_manager.segment_image(preprocessed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Required model artifact is missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"U-Net segmentation failed: {str(e)}")
    tumor_pixel_count = int(np.sum(refined_mask))
    total_pixels = IMG_SIZE * IMG_SIZE
    tumor_area_pct = round(float((tumor_pixel_count / total_pixels) * 100.0), 2)
    has_tumor = tumor_pixel_count > 0

    stages_executed.append(PipelineStage(
        name="U-Net Deep Learning Segmentation",
        description=f"Generated probability map (threshold 0.5). Detected {tumor_pixel_count} tumor pixels ({tumor_area_pct}% of area)."
    ))

    stages_executed.append(PipelineStage(
        name="Connected Component QC & Morphology",
        description="Removed small artifacts (<50 px) and applied morphological closing with 3x3 elliptical kernel."
    ))

    # 4. Extract Tumor ROI and generate Overlay
    roi = extract_tumor_roi(preprocessed, refined_mask)
    roi_overlay = create_roi_overlay(orig_resized, refined_mask)

    stages_executed.append(PipelineStage(
        name="Tumor ROI Extraction & Overlay",
        description="Extracted tumor pixel intensities and created semi-transparent red blend with yellow contour boundary."
    ))

    # 5. GLCM Feature Extraction
    glcm_feat = extract_glcm_features(roi)
    if glcm_feat is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "The U-Net mask did not produce a usable non-empty ROI, so GLCM, PCA, "
                "SVM, and KNN classification were not run."
            ),
        )

    stages_executed.append(PipelineStage(
        name="GLCM Texture Feature Extraction",
        description="Computed 96 Haralick texture features across 4 distances, 4 angles, and 6 statistical properties."
    ))

    # 6. StandardScaler, PCA & Dual Classification (SVM + KNN)
    try:
        pca_values, svm_data, knn_data = model_manager.classify_features(glcm_feat)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Required classification artifact is missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    stages_executed.append(PipelineStage(
        name="Standardization & PCA Dimensionality Reduction",
        description="Standardized 96 GLCM features and projected to 5 principal components (capturing 95% variance)."
    ))

    stages_executed.append(PipelineStage(
        name="Dual-Model Machine Learning Classification",
        description=f"SVM Predicted: {svm_data['prediction']} ({svm_data['score']}%) | KNN Predicted: {knn_data['prediction']} ({knn_data['score']}%)."
    ))

    execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    # 7. Encode pipeline images as Base64 data URLs
    images_payload = PipelineImages(
        original=encode_image_to_base64(raw_image),
        preprocessed=encode_image_to_base64(preprocessed),
        segmentation_mask=encode_image_to_base64(refined_mask),
        extracted_roi=encode_image_to_base64(roi),
        roi_overlay=encode_image_to_base64(roi_overlay),
    )

    metadata_payload = PipelineMetadata(
        glcm_features_extracted=True,
        glcm_features_count=96,
        pca_applied=True,
        pca_components_count=5,
        pca_components=pca_values,
        segmentation_model="U-Net",
        tumor_detected=has_tumor,
        tumor_pixel_count=tumor_pixel_count,
        tumor_area_percentage=tumor_area_pct,
        execution_time_ms=execution_time_ms,
    )

    svm_result = ClassificationModelResult(
        model_name="Support Vector Machine (SVM)",
        prediction=svm_data["prediction"],
        score=svm_data["score"],
        score_type=svm_data["score_type"],
        score_unit=svm_data["score_unit"],
        probabilities=svm_data["probabilities"],
        decision_scores=svm_data["decision_scores"],
    )

    knn_result = ClassificationModelResult(
        model_name="K-Nearest Neighbors (KNN)",
        prediction=knn_data["prediction"],
        score=knn_data["score"],
        score_type=knn_data["score_type"],
        score_unit=knn_data["score_unit"],
        probabilities=knn_data["probabilities"],
    )

    return PredictionResponse(
        status="success",
        stages=stages_executed,
        images=images_payload,
        svm=svm_result,
        knn=knn_result,
        metadata=metadata_payload,
    )

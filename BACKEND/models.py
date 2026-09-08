import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from BACKEND.config import (
    UNET_MODEL_PATH,
    SCALER_PATH,
    PCA_PATH,
    SVM_MODEL_PATH,
    KNN_MODEL_PATH,
    UNET_THRESHOLD,
    CLASS_NAMES,
)
from BACKEND.segmentation import clean_prediction_mask, apply_morphological_closing


class ModelManager:
    """
    Singleton class managing the lifecycle and inference of all machine learning
    and deep learning artifacts. Loaded once during application startup.
    """
    def __init__(self):
        self.unet = None
        self.scaler = None
        self.pca = None
        self.svm = None
        self.knn = None
        self.classes = CLASS_NAMES
        self.pca_feature_names = ["PC1", "PC2", "PC3", "PC4", "PC5"]
        self.is_loaded = False

    def load_all_models(self):
        """Load all saved models and pipelines."""
        if self.is_loaded:
            return

        print("--> Loading U-Net Segmentation Model...")
        if not os.path.exists(UNET_MODEL_PATH):
            raise FileNotFoundError(f"U-Net model file not found at: {UNET_MODEL_PATH}")
        self.unet = tf.keras.models.load_model(str(UNET_MODEL_PATH), compile=False)

        print("--> Loading Feature Scaler...")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
        self.scaler = joblib.load(str(SCALER_PATH))

        print("--> Loading PCA Pipeline...")
        if not os.path.exists(PCA_PATH):
            raise FileNotFoundError(f"PCA file not found at: {PCA_PATH}")
        self.pca = joblib.load(str(PCA_PATH))

        print("--> Loading SVM Classifier...")
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError(f"SVM model file not found at: {SVM_MODEL_PATH}")
        self.svm = joblib.load(str(SVM_MODEL_PATH))

        print("--> Loading KNN Classifier...")
        if not os.path.exists(KNN_MODEL_PATH):
            raise FileNotFoundError(f"KNN model file not found at: {KNN_MODEL_PATH}")
        self.knn = joblib.load(str(KNN_MODEL_PATH))

        # Confirm classes and feature names from models
        if hasattr(self.svm, "classes_"):
            self.classes = list(self.svm.classes_)
        if hasattr(self.svm, "feature_names_in_"):
            self.pca_feature_names = list(self.svm.feature_names_in_)

        self.is_loaded = True
        print(" All models loaded successfully into memory.")

    def segment_image(self, preprocessed_image: np.ndarray):
        """
        Run U-Net segmentation on preprocessed image (256x256 uint8).
        Returns:
            probability_map: (256, 256) float32
            raw_mask: (256, 256) uint8 [0, 1]
            cleaned_mask: (256, 256) uint8 [0, 1]
            refined_mask: (256, 256) uint8 [0, 1]
        """
        if not self.is_loaded:
            self.load_all_models()

        # Normalize to [0, 1] as in Training_U-Net & Tumor_ROI_Generation
        image_norm = preprocessed_image.astype(np.float32) / 255.0
        model_input = np.expand_dims(image_norm, axis=(0, -1))

        # Predict probability map
        pred = self.unet.predict(model_input, verbose=0)
        probability_map = pred[0, :, :, 0]

        # Thresholding
        raw_mask = (probability_map >= UNET_THRESHOLD).astype(np.uint8)

        # Connected component filtering (removes components < 50 px)
        cleaned_mask = clean_prediction_mask(raw_mask)

        # Morphological closing (3x3 ellipse kernel, 1 iteration)
        refined_mask = apply_morphological_closing(cleaned_mask)

        return probability_map, raw_mask, cleaned_mask, refined_mask

    def classify_features(self, glcm_features: np.ndarray):
        """
        Apply Scaler -> PCA -> SVM & KNN classifiers.
        Returns:
            pca_values: list of 5 float values
            svm_result: dict with prediction, score, class_probabilities
            knn_result: dict with prediction, score, class_probabilities
        """
        if not self.is_loaded:
            self.load_all_models()

        # 1. Standardization (StandardScaler fitted on 96 features)
        scaled_feat = self.scaler.transform([glcm_features])

        # 2. PCA Transformation (Project to 5 principal components)
        pca_feat_raw = self.pca.transform(scaled_feat)
        pca_values = pca_feat_raw[0].tolist()

        # Use DataFrame with feature names to match training dataset columns
        pca_df = pd.DataFrame(pca_feat_raw, columns=self.pca_feature_names)

        # 3. SVM Classification
        svm_pred = self.svm.predict(pca_df)[0]
        svm_classes = list(self.svm.classes_)
        svm_probs = self.svm.predict_proba(pca_df)[0]
        svm_pred_idx = svm_classes.index(svm_pred)
        svm_conf = float(svm_probs[svm_pred_idx] * 100.0)

        svm_result = {
            "prediction": str(svm_pred),
            "score": round(svm_conf, 2),
            "score_type": "SVM predict_proba probability",
            "score_unit": "%",
            "decision_scores": None,
            "probabilities": {cls: round(float(svm_probs[i] * 100.0), 2) for i, cls in enumerate(svm_classes)},
        }

        # 4. KNN Classification
        knn_pred = self.knn.predict(pca_df)[0]
        knn_probs = self.knn.predict_proba(pca_df)[0]
        knn_classes = list(self.knn.classes_)
        knn_pred_idx = knn_classes.index(knn_pred)
        knn_conf = float(knn_probs[knn_pred_idx] * 100.0)

        knn_result = {
            "prediction": str(knn_pred),
            "score": round(knn_conf, 2),
            "score_type": "KNN predict_proba posterior probability",
            "score_unit": "%",
            "probabilities": {cls: round(float(knn_probs[i] * 100.0), 2) for i, cls in enumerate(knn_classes)},
        }

        return pca_values, svm_result, knn_result


# Singleton instance
model_manager = ModelManager()

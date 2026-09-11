import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"
DATASET_DIR = PROJECT_ROOT / "DATASET"
FEATURES_DIR = PROJECT_ROOT / "Features"

# Model Weights and Artifact Paths
UNET_MODEL_PATH = MODELS_DIR / "best_unet.keras"
SCALER_PATH = FEATURES_DIR / "scaler.pkl"
PCA_PATH = FEATURES_DIR / "pca.pkl"
SVM_MODEL_PATH = MODELS_DIR / "final_svm.pkl"
KNN_MODEL_PATH = MODELS_DIR / "pca_knn_glcm.joblib"

# Image and Pipeline Parameters
IMG_SIZE = 256
UNET_THRESHOLD = 0.5
MIN_COMPONENT_AREA = 50
MORPH_KERNEL_SIZE = (3, 3)
MORPH_ITERATIONS = 1

# GLCM Parameters
GLCM_DISTANCES = [1, 2, 3, 4]
GLCM_ANGLES = [0, 3.141592653589793 / 4, 3.141592653589793 / 2, 3 * 3.141592653589793 / 4]
GLCM_LEVELS = 32
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# Target Classes
CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

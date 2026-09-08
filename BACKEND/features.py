import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from BACKEND.config import (
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
)


def crop_tumor_roi(roi: np.ndarray):
    """
    Crop the saved tumor ROI to the bounding box containing non-zero tumor pixels.
    Exact reproduction from Feauture_Extraction_1.ipynb:
        y_coords, x_coords = np.where(roi > 0)
        cropped_roi = roi[y_min:y_max + 1, x_min:x_max + 1]
    """
    y_coords, x_coords = np.where(roi > 0)
    if len(x_coords) == 0:
        return None

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    cropped_roi = roi[y_min:y_max + 1, x_min:x_max + 1]
    return cropped_roi


def prepare_for_glcm(roi: np.ndarray):
    """
    Convert ROI intensities into 32 gray levels for GLCM computation.
    Exact reproduction from Feauture_Extraction_1.ipynb:
        roi_normalized = cv2.normalize(roi, None, 0, GLCM_LEVELS - 1, cv2.NORM_MINMAX)
        roi_quantized = roi_normalized.astype(np.uint8)
    """
    roi_uint8 = roi.astype(np.uint8)
    if np.max(roi_uint8) == 0:
        return None

    roi_normalized = cv2.normalize(
        roi_uint8,
        None,
        0,
        GLCM_LEVELS - 1,
        cv2.NORM_MINMAX,
    )
    return roi_normalized.astype(np.uint8)


def extract_glcm_features(roi: np.ndarray):
    """
    Extract GLCM texture features from the cropped tumor ROI.
    Exact reproduction from Feauture_Extraction_1.ipynb:
        Calculates contrast, dissimilarity, homogeneity, energy, correlation, ASM
        across 4 distances [1, 2, 3, 4] and 4 angles [0, pi/4, pi/2, 3pi/4].
        Total features: 6 * 4 * 4 = 96.
    """
    cropped_roi = crop_tumor_roi(roi)
    if cropped_roi is None:
        return None

    roi_quantized = prepare_for_glcm(cropped_roi)
    if roi_quantized is None:
        return None

    glcm = graycomatrix(
        roi_quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for property_name in GLCM_PROPERTIES:
        property_values = graycoprops(glcm, property_name)
        for distance_index in range(len(GLCM_DISTANCES)):
            for angle_index in range(len(GLCM_ANGLES)):
                val = property_values[distance_index, angle_index]
                features.append(val)

    return np.array(features, dtype=np.float32)


def create_feature_names() -> list:
    """
    Generate the 96 feature column names matching Feauture_Extraction_1.ipynb.
    """
    feature_names = []
    angle_labels = ["0", "45", "90", "135"]
    for property_name in GLCM_PROPERTIES:
        for distance in GLCM_DISTANCES:
            for angle_name in angle_labels:
                feature_names.append(f"{property_name}_d{distance}_a{angle_name}")
    return feature_names

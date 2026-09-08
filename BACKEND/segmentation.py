import cv2
import numpy as np
from BACKEND.config import (
    MIN_COMPONENT_AREA,
    MORPH_KERNEL_SIZE,
    MORPH_ITERATIONS,
)


def clean_prediction_mask(binary_mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    """
    Remove small disconnected components from binary U-Net prediction mask.
    Exact reproduction from Tumor_ROI_Generation.ipynb:
        Components smaller than min_area are removed.
    """
    mask_uint8 = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8
    )

    cleaned_mask = np.zeros_like(mask_uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == label] = 1

    return cleaned_mask


def apply_morphological_closing(
    binary_mask: np.ndarray,
    kernel_size: tuple = MORPH_KERNEL_SIZE,
    iterations: int = MORPH_ITERATIONS,
) -> np.ndarray:
    """
    Apply morphological closing using an elliptical structuring element.
    Exact reproduction from Tumor_ROI_Generation.ipynb:
        Helps fill small holes and gaps in the predicted tumor region.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    refined_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )
    return refined_mask


def extract_tumor_roi(image_normalized_or_uint8: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """
    Extract tumor ROI as done in Tumor_ROI_Generation.ipynb:
        roi = (image_batch[j][:, :, 0] * refined_mask) * 255 -> uint8
    """
    if image_normalized_or_uint8.dtype == np.uint8:
        image_float = image_normalized_or_uint8.astype(np.float32) / 255.0
    else:
        image_float = image_normalized_or_uint8.astype(np.float32)

    roi_float = image_float * refined_mask
    roi_uint8 = (roi_float * 255).astype(np.uint8)
    return roi_uint8


def create_roi_overlay(image_256: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Create tumor ROI visual overlay matching U-Net_Evaluation.ipynb (Cell 13):
        1. Convert grayscale image to RGB.
        2. Create red highlight layer for tumor region (alpha=0.5).
        3. Draw yellow contour borders around the tumor (thickness=2).
    """
    if len(image_256.shape) == 2:
        overlay = cv2.cvtColor(image_256, cv2.COLOR_GRAY2RGB)
    else:
        overlay = cv2.cvtColor(cv2.cvtColor(image_256, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

    tumor_region = np.zeros_like(overlay)
    tumor_region[:, :, 0] = 255  # Red channel

    mask_bool = binary_mask.astype(bool)
    if np.any(mask_bool):
        # Blend only where tumor exists
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool],
            0.5,
            tumor_region[mask_bool],
            0.5,
            0,
        )

        # Find external contours and draw yellow contour border
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    return overlay

import cv2
import numpy as np
from BACKEND.config import IMG_SIZE


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to single-channel grayscale image.
    Exact reproduction of preprocessing.ipynb:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, size: tuple = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """
    Resize MRI image to fixed spatial resolution (256, 256) using cv2.INTER_AREA.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply mild Gaussian filtering (3x3 kernel, sigma=0) to reduce noise.
    """
    return cv2.GaussianBlur(image, (3, 3), 0)


def z_score_normalization(image: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization:
        Z = (X - mean) / standard_deviation
    Rescaled to 0-255 using cv2.NORM_MINMAX and converted back to uint8.
    """
    image_float = image.astype(np.float32)
    mean = np.mean(image_float)
    std = np.std(image_float)

    if std == 0:
        normalized = np.zeros_like(image_float)
    else:
        normalized = (image_float - mean) / std

    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def preprocess_mri(image: np.ndarray) -> np.ndarray:
    """
    Execute full preprocessing pipeline as defined in notebooks:
    RGB/BGR -> Grayscale -> Resize (256x256) -> Gaussian Denoising -> Z-Score Normalization -> uint8 (0-255)
    """
    gray = convert_to_grayscale(image)
    resized = resize_image(gray, (IMG_SIZE, IMG_SIZE))
    denoised = denoise_image(resized)
    normalized = z_score_normalization(denoised)
    return normalized

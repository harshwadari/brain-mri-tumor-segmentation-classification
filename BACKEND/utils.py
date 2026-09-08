import base64
import cv2
import numpy as np


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV image (numpy array).
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Invalid or corrupted image format.")
    return image


def encode_image_to_base64(image: np.ndarray, format: str = ".png") -> str:
    """
    Convert an OpenCV numpy image into a Base64 data URL string.
    """
    # If binary mask [0, 1], scale to [0, 255]
    if image.dtype == bool or (image.dtype == np.uint8 and np.max(image) <= 1):
        display_img = (image.astype(np.uint8) * 255)
    else:
        display_img = image

    success, buffer = cv2.imencode(format, display_img)
    if not success:
        raise RuntimeError("Failed to encode image to base64.")

    encoded = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format.endswith("png") else "image/jpeg"
    return f"data:{mime};base64,{encoded}"

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional
import requests
from PIL import Image

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def check_backend_health(base_url: str = API_BASE_URL) -> bool:
    """Check if FastAPI backend is reachable and responsive."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_backend_info(base_url: str = API_BASE_URL) -> Optional[Dict[str, Any]]:
    """Retrieve backend metadata, classes, and loaded status."""
    try:
        resp = requests.get(f"{base_url}/api/info", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_sample_images(base_url: str = API_BASE_URL) -> Dict[str, list]:
    """Retrieve list of sample test images grouped by class."""
    try:
        resp = requests.get(f"{base_url}/api/sample-images", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def get_sample_image_bytes(class_name: str, filename: str, base_url: str = API_BASE_URL) -> Optional[bytes]:
    """Download bytes of a specific test sample from the backend."""
    try:
        url = f"{base_url}/api/sample-image/{class_name}/{filename}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def predict_image(image_bytes: bytes, filename: str = "mri.jpg", base_url: str = API_BASE_URL) -> Dict[str, Any]:
    """
    Send MRI image bytes to FastAPI /api/predict endpoint.
    Returns parsed PredictionResponse JSON or raises RuntimeError.
    """
    files = {"file": (filename, image_bytes, "image/jpeg")}
    try:
        resp = requests.post(f"{base_url}/api/predict", files=files, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Backend returned error ({resp.status_code}): {detail}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Unable to connect to the FastAPI backend at " + base_url)
    except Exception as e:
        raise RuntimeError(f"Inference request failed: {str(e)}")


def base64_to_pil(b64_data_url: str) -> Image.Image:
    """Convert base64 data URL string to PIL Image."""
    if "," in b64_data_url:
        b64_str = b64_data_url.split(",")[1]
    else:
        b64_str = b64_data_url
    image_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_bytes))

# 🧠 Brain Tumor MRI Segmentation & Classification

An end-to-end deep learning and machine learning pipeline for **brain tumor segmentation** and **multi-class classification** from T1-weighted MRI scans. The system segments tumor regions using a **U-Net** deep learning model, extracts **GLCM texture features** from the segmented ROI, reduces dimensionality with **PCA**, and classifies tumors into three categories — **Glioma**, **Meningioma**, and **Pituitary** — using dual **SVM** and **KNN** classifiers. A full-stack web application (**FastAPI** backend + **Streamlit** frontend) provides an interactive medical diagnostic interface.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Complete Pipeline Workflow](#-complete-pipeline-workflow)
- [Methodology](#-methodology)
- [Models and Algorithms](#-models-and-algorithms)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run the Project](#-how-to-run-the-project)
- [Notebook Documentation](#-notebook-documentation)
- [Model Files](#-model-files)
- [Dataset Setup](#-dataset-setup)
- [Results / Evaluation](#-results--evaluation)
- [Technologies Used](#-technologies-used)
- [Limitations / Notes](#-limitations--notes)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔬 Project Overview

This project addresses the problem of **automated brain tumor detection and classification** from MRI scans. Manual analysis of brain MRI images by radiologists is time-consuming and prone to variability. This system automates the process through a multi-stage pipeline:

1. **Preprocessing** the raw MRI image
2. **Segmenting** the tumor region using a trained U-Net model
3. **Extracting** texture features from the segmented tumor ROI
4. **Classifying** the tumor type using dual machine learning classifiers (SVM + KNN)

The project is implemented as a deployable web application with a FastAPI REST backend and a Streamlit-based frontend dashboard.

**Target Classes:** `Glioma` · `Meningioma` · `Pituitary`

---

## ✨ Key Features

- **U-Net Deep Learning Segmentation** — Pixel-wise tumor mask prediction with post-processing (connected component filtering + morphological closing)
- **GLCM Texture Feature Extraction** — 96 Haralick features across 4 distances, 4 angles, and 6 statistical properties
- **PCA Dimensionality Reduction** — Projects 96 features to 5 principal components
- **Dual-Model Classification** — Parallel SVM and KNN inference with probability scores
- **Interactive Web Dashboard** — Streamlit UI with animated pipeline progress, probability bar charts, and 5-stage image visualization
- **FastAPI REST API** — Production-ready backend with health checks, sample image endpoints, and structured JSON responses
- **End-to-End Pipeline Visualization** — Displays Original → Preprocessed → Segmentation Mask → Extracted ROI → ROI Overlay for visual verification
- **Sample Image Demo** — Load test images directly from the dataset for quick demonstration

---

## 🔄 Complete Pipeline Workflow

```
┌──────────────┐     ┌───────────────────────┐     ┌──────────────────────────┐
│  Input MRI   │────▶│    Preprocessing      │────▶│  U-Net Segmentation      │
│  (JPG/PNG)   │     │  Grayscale → Resize   │     │  Probability Map → Mask  │
│              │     │  → Denoise → Z-Score   │     │  (Threshold = 0.5)       │
└──────────────┘     └───────────────────────┘     └──────────┬───────────────┘
                                                              │
                     ┌───────────────────────┐     ┌──────────▼───────────────┐
                     │  Post-Processing      │◀────│  Connected Component QC  │
                     │  Morphological Closing │     │  Remove artifacts <50 px │
                     │  (3×3 Elliptical)      │     └──────────────────────────┘
                     └──────────┬────────────┘
                                │
                     ┌──────────▼────────────┐     ┌──────────────────────────┐
                     │  Tumor ROI Extraction  │────▶│  GLCM Feature Extraction │
                     │  + Overlay Generation  │     │  96 Texture Features     │
                     └────────────────────────┘     └──────────┬───────────────┘
                                                               │
                     ┌──────────────────────────┐   ┌──────────▼───────────────┐
                     │  Dual Classification     │◀──│  StandardScaler + PCA    │
                     │  SVM + KNN Inference     │   │  96 → 5 Components       │
                     └──────────┬───────────────┘   └──────────────────────────┘
                                │
                     ┌──────────▼───────────────┐
                     │  Final Output            │
                     │  Class + Probability (%)  │
                     │  Pipeline Images (Base64) │
                     └──────────────────────────┘
```

---

## 🧪 Methodology

### 1. Preprocessing

Implemented in [`BACKEND/preprocessing.py`](BACKEND/preprocessing.py) and [`NOTEBOOKS/preprocessing.ipynb`](NOTEBOOKS/preprocessing.ipynb):

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Grayscale Conversion | `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` |
| 2 | Resize | `256 × 256` using `cv2.INTER_AREA` interpolation |
| 3 | Gaussian Denoising | `3 × 3` kernel, sigma = 0 |
| 4 | Z-Score Normalization | `Z = (X - μ) / σ`, rescaled to `[0, 255]` via `cv2.NORM_MINMAX` |

### 2. U-Net Segmentation

Implemented in [`NOTEBOOKS/Training_U-Net.ipynb`](NOTEBOOKS/Training_U-Net.ipynb):

- **Architecture:** U-Net (encoder-decoder with skip connections)
- **Input:** `256 × 256 × 1` grayscale MRI (normalized to `[0, 1]`)
- **Output:** Pixel-wise probability map (sigmoid activation)
- **Loss Function:** Binary Cross-Entropy + Dice Loss (`BCE + Dice Loss`)
- **Optimizer:** Adam (learning rate = `1e-4`)
- **Callbacks:** ModelCheckpoint (best `val_dice_coefficient`), EarlyStopping (patience = 10), ReduceLROnPlateau (factor = 0.5, patience = 5)
- **Training:** Up to 50 epochs with early stopping on validation loss
- **Threshold:** `0.5` on sigmoid probability map to produce binary mask

### 3. Mask Post-Processing & ROI Generation

Implemented in [`BACKEND/segmentation.py`](BACKEND/segmentation.py), matching [`NOTEBOOKS/Tumor_ROI_Generation.ipynb`](NOTEBOOKS/Tumor_ROI_Generation.ipynb):

1. **Connected Component Filtering** — Removes small disconnected artifacts smaller than 50 pixels (8-connectivity)
2. **Morphological Closing** — Fills small holes using a `3 × 3` elliptical structuring element (1 iteration)
3. **Tumor ROI Extraction** — Masks the preprocessed image with the refined binary mask
4. **ROI Overlay** — Creates a semi-transparent red blend with yellow contour boundary for visualization

### 4. Feature Extraction (GLCM)

Implemented in [`BACKEND/features.py`](BACKEND/features.py), matching [`NOTEBOOKS/Feauture_Extraction_1.ipynb`](NOTEBOOKS/Feauture_Extraction_1.ipynb):

1. **Bounding Box Crop** — Crops ROI to the tightest bounding box around non-zero tumor pixels
2. **Intensity Quantization** — Normalizes and quantizes ROI to **32 gray levels**
3. **GLCM Computation** — Computes the Gray-Level Co-occurrence Matrix:
   - **Distances:** `[1, 2, 3, 4]`
   - **Angles:** `[0°, 45°, 90°, 135°]`
   - Symmetric and normalized
4. **Property Extraction** — Extracts 6 Haralick properties per distance-angle pair:
   - `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`
   - **Total Features:** `6 × 4 × 4 = 96`

### 5. Dimensionality Reduction (PCA)

- **StandardScaler** applied to the 96 GLCM features (fitted on training data)
- **PCA** reduces 96 features → **5 principal components**
- Saved artifacts: `scaler.pkl` and `pca.pkl` in `DATASET/Features(1)/`

### 6. Classification (SVM + KNN)

**SVM (Support Vector Machine)** — [`NOTEBOOKS/Training_SVM.ipynb`](NOTEBOOKS/Training_SVM.ipynb):
- Hyperparameter tuning via `GridSearchCV` with 3-fold stratified cross-validation
- Best parameters: `C = 10`, `kernel = linear`
- `probability=True` enabled for `predict_proba()` support
- Trained on PCA-transformed features (50 PCA components during training; 5 used in deployment)

**KNN (K-Nearest Neighbors)** — [`NOTEBOOKS/Training_KNN_Final.ipynb`](NOTEBOOKS/Training_KNN_Final.ipynb):
- Hyperparameter tuning via `GridSearchCV` with 3-fold cross-validation
- Best parameters: `n_neighbors = 7`, `weights = distance`, `metric = manhattan`
- Trained on PCA-transformed features

---

## 🤖 Models and Algorithms

| Model | Role | File | Details |
|-------|------|------|---------|
| **U-Net** | Tumor Segmentation | `MODELS/best_unet.keras` | Deep learning encoder-decoder; predicts pixel-wise tumor masks |
| **SVM** | Tumor Classification | `MODELS/final_svm.pkl` | Linear kernel SVM (`C=10`); classifies GLCM+PCA features into tumor types |
| **KNN** | Tumor Classification | `MODELS/pca_knn_glcm.joblib` | KNN (`k=7`, manhattan distance, distance-weighted); provides secondary classification |
| **StandardScaler** | Feature Scaling | `DATASET/Features(1)/scaler.pkl` | Z-score standardization of 96 GLCM features |
| **PCA** | Dimensionality Reduction | `DATASET/Features(1)/pca.pkl` | Reduces 96 features to 5 principal components |

---

## 📊 Datasets

### Primary Dataset: BRISC 2025

**BRISC** (BRain tumor Image Segmentation & Classification) — a curated, expert-annotated T1-weighted MRI dataset.

- **Source:** [ArXiv preprint (Fateh et al., 2025)](https://arxiv.org/abs/2506.14318)
- **Total samples:** 6,000 T1-weighted MRI slices (5,000 train / 1,000 test)
- **Classes:** 4 — Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Anatomical Planes:** Axial, Coronal, Sagittal
- **Annotations:** Pixel-wise segmentation masks reviewed by radiologists

The dataset is organized under the `brisc2025/` directory with:
- `classification_task/` — Image-level classification labels organized by class folders
- `segmentation_task/` — Paired MRI images and binary segmentation masks
- `no_tumor/` — MRI slices without tumors

### Processed Datasets (under `DATASET/`)

| Subdirectory | Description |
|-------------|-------------|
| `Classification/` | Classification images split into `train/` and `test/` with subfolders per class (`glioma`, `meningioma`, `pituitary`) |
| `Classification_Preprocessed/` | Preprocessed versions of classification images (train/test) |
| `Segmentation/` | Segmentation images and masks split into `train/` and `test/` with `images/` and `masks/` subfolders |
| `Preprocessed_segmented/` | Preprocessed segmentation images and masks (train/test) |
| `Tumor_ROI/` | Extracted tumor ROI images per class (train/test) |
| `Tumor_ROI_Morphology/` | ROI images after morphological post-processing (train/test) |
| `Features(1)/` | Extracted GLCM features (`glcm_train.csv`, `glcm_test.csv`), PCA-transformed features (`pca_train.csv`, `pca_test.csv`), and fitted `scaler.pkl` / `pca.pkl` |

---

## 📁 Project Structure

```
BE_Major_Project/
│
├── app.py                          # Root launcher — runs Streamlit frontend via runpy
├── run_backend.py                  # Launches FastAPI backend via uvicorn
├── LICENSE                         # MIT License
├── .gitignore                      # Excludes MODELS/, DATASET/, brisc2025/, venvs, caches
│
├── BACKEND/                        # FastAPI REST API backend
│   ├── __init__.py
│   ├── main.py                     # FastAPI app definition, CORS, lifespan, API endpoints
│   ├── config.py                   # Paths, hyperparameters, GLCM config, class names
│   ├── models.py                   # ModelManager singleton — loads & runs U-Net, SVM, KNN
│   ├── preprocessing.py            # MRI preprocessing pipeline (grayscale, resize, denoise, z-score)
│   ├── segmentation.py             # Mask cleaning, morphological closing, ROI extraction, overlay
│   ├── features.py                 # GLCM feature extraction (crop, quantize, compute 96 features)
│   ├── schemas.py                  # Pydantic response models (PredictionResponse, etc.)
│   └── utils.py                    # Image decode/encode utilities (bytes ↔ base64)
│
├── FRONTEND/                       # Streamlit web dashboard
│   ├── __init__.py
│   ├── app.py                      # Full Streamlit UI — upload, pipeline animation, results display
│   └── utils.py                    # HTTP client utilities — health check, predict, sample images
│
├── NOTEBOOKS/                      # Jupyter notebooks (training, evaluation, data analysis)
│   ├── Data_Analysis.ipynb
│   ├── DATA_ANALYSIS_SEGMENTATION.ipynb
│   ├── preprocessing.ipynb
│   ├── Preprocessing_Segmented_Dataset.ipynb
│   ├── Training_U-Net.ipynb
│   ├── U-Net_Evaluation.ipynb
│   ├── Tumor_ROI_Generation.ipynb
│   ├── Feauture_Extraction_1.ipynb
│   ├── Training_SVM.ipynb
│   ├── Training_KNN_Final.ipynb
│   ├── Untitled.ipynb              # Experimental/scratch KNN notebook
│   ├── classification_dataset_distribution.png
│   └── segmentation_dataset_distribution.png
│
├── MODELS/                         # Trained model artifacts (git-ignored)
│   ├── best_unet.keras             # Best U-Net weights (saved by ModelCheckpoint)
│   ├── final_unet.keras            # Final U-Net weights (after full training)
│   ├── final_svm.pkl               # Trained SVM classifier (joblib)
│   ├── pca_knn_glcm.joblib         # Trained KNN classifier (joblib)
│   ├── pca_knn_glcm_final.joblib   # Alternative KNN model
│   ├── svm_confusion_matrix.png    # SVM confusion matrix visualization
│   └── svm_roc_curve.png           # SVM ROC curve visualization
│
├── DATASET/                        # Processed datasets (git-ignored)
│   ├── Classification/             # Classification split (train/test per class)
│   ├── Classification_Preprocessed/
│   ├── Segmentation/               # Segmentation split (images + masks)
│   ├── Preprocessed_segmented/
│   ├── Tumor_ROI/
│   ├── Tumor_ROI_Morphology/
│   └── Features(1)/               # Extracted features, scaler, PCA
│       ├── glcm_train.csv
│       ├── glcm_test.csv
│       ├── pca_train.csv
│       ├── pca_test.csv
│       ├── scaler.pkl
│       └── pca.pkl
│
├── brisc2025/                      # Raw BRISC 2025 dataset (git-ignored)
│   ├── README.md
│   ├── classification_task/
│   ├── segmentation_task/
│   ├── no_tumor/
│   ├── manifest.csv
│   └── manifest.json
│
└── Images/                         # Miscellaneous project images
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.x** (developed with Anaconda distribution)
- **pip** or **conda** for package management
- GPU support recommended for U-Net inference (NVIDIA CUDA + cuDNN)

### Dependencies

The project uses the following core libraries:

| Category | Libraries |
|----------|-----------|
| **Deep Learning** | `tensorflow` (Keras API) |
| **Machine Learning** | `scikit-learn`, `joblib` |
| **Image Processing** | `opencv-python` (`cv2`), `scikit-image` |
| **Backend** | `fastapi`, `uvicorn`, `pydantic` |
| **Frontend** | `streamlit`, `requests` |
| **Data Science** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `Pillow` (`PIL`) |

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/brain-mri-tumor-segmentation-classification.git
cd brain-mri-tumor-segmentation-classification

# 2. Create a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install tensorflow opencv-python scikit-learn scikit-image \
            fastapi uvicorn pydantic streamlit requests \
            numpy pandas matplotlib Pillow joblib
```

> **Note:** A `requirements.txt` file is not present in the repository. Install the dependencies listed above manually.

---

## 🚀 How to Run the Project

The application runs as two separate processes: a **FastAPI backend** and a **Streamlit frontend**.

### Step 1: Start the Backend (FastAPI)

```bash
# From the project root directory
python run_backend.py
```

This launches the FastAPI server on `http://127.0.0.1:8000` and loads all models into memory during startup.

Alternatively:

```bash
uvicorn BACKEND.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Start the Frontend (Streamlit)

In a separate terminal:

```bash
# From the project root directory
python app.py
```

Or directly:

```bash
streamlit run FRONTEND/app.py
```

The Streamlit dashboard will open in your browser (typically at `http://localhost:8501`).

### Using the Application

1. **Check Connectivity** — The sidebar shows the backend connection status
2. **Upload an MRI Scan** — Use the file uploader (supports JPG, JPEG, PNG)
3. **Or Select a Sample** — Choose a Quick-Demo sample from the sidebar dropdown
4. **Click "Analyze Image"** — Watch the animated 8-stage pipeline progress
5. **View Results** — See dual SVM/KNN classification with probability charts, 5-stage image pipeline, and detailed diagnostic data

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — returns service status |
| `/health` | GET | Backend health status |
| `/api/info` | GET | System metadata (classes, resolution, models loaded) |
| `/api/sample-images` | GET | Available test images grouped by class |
| `/api/sample-image/{class}/{file}` | GET | Retrieve specific test image |
| `/api/predict` | POST | **Main inference endpoint** — runs full pipeline on uploaded MRI |

---

## 📓 Notebook Documentation

| Notebook | Purpose |
|----------|---------|
| [`Data_Analysis.ipynb`](NOTEBOOKS/Data_Analysis.ipynb) | Exploratory data analysis on the classification dataset |
| [`DATA_ANALYSIS_SEGMENTATION.ipynb`](NOTEBOOKS/DATA_ANALYSIS_SEGMENTATION.ipynb) | Exploratory data analysis on the segmentation dataset |
| [`preprocessing.ipynb`](NOTEBOOKS/preprocessing.ipynb) | Develops and visualizes the MRI preprocessing pipeline (grayscale, resize, denoise, z-score) |
| [`Preprocessing_Segmented_Dataset.ipynb`](NOTEBOOKS/Preprocessing_Segmented_Dataset.ipynb) | Applies preprocessing to the segmentation dataset images and masks |
| [`Training_U-Net.ipynb`](NOTEBOOKS/Training_U-Net.ipynb) | Builds, trains, and evaluates the U-Net segmentation model (BCE + Dice Loss, Adam, 50 epochs) |
| [`U-Net_Evaluation.ipynb`](NOTEBOOKS/U-Net_Evaluation.ipynb) | Comprehensive U-Net evaluation on the test set — computes Dice, IoU, Precision, Recall, Specificity, F1, Pixel Accuracy |
| [`Tumor_ROI_Generation.ipynb`](NOTEBOOKS/Tumor_ROI_Generation.ipynb) | Generates tumor ROI images using U-Net predictions — includes connected component filtering and morphological closing |
| [`Feauture_Extraction_1.ipynb`](NOTEBOOKS/Feauture_Extraction_1.ipynb) | Extracts 96 GLCM texture features from tumor ROIs; fits StandardScaler and PCA |
| [`Training_SVM.ipynb`](NOTEBOOKS/Training_SVM.ipynb) | Trains and evaluates the SVM classifier with GridSearchCV hyperparameter tuning |
| [`Training_KNN_Final.ipynb`](NOTEBOOKS/Training_KNN_Final.ipynb) | Trains and evaluates the KNN classifier with GridSearchCV hyperparameter tuning |
| [`Untitled.ipynb`](NOTEBOOKS/Untitled.ipynb) | Experimental/scratch notebook with preliminary KNN exploration |

---

## 🗃 Model Files

Trained model files are **excluded from Git** (via `.gitignore`) due to their large size. The following files must be present in the `MODELS/` directory for the application to run:

| File | Size (approx.) | Description |
|------|-----------------|-------------|
| `best_unet.keras` | ~355 MB | Best U-Net model weights (saved during training by ModelCheckpoint) |
| `final_svm.pkl` | ~714 KB | Trained SVM classifier (linear kernel, C=10, probability=True) |
| `pca_knn_glcm.joblib` | ~790 KB | Trained KNN classifier (k=7, manhattan, distance-weighted) |

Additionally, the following files must be present in `DATASET/Features(1)/`:

| File | Description |
|------|-------------|
| `scaler.pkl` | Fitted StandardScaler for 96 GLCM features |
| `pca.pkl` | Fitted PCA transformer |

> **Note:** These models must be generated by running the training notebooks in order, or obtained from the project contributors. The `final_unet.keras` and `pca_knn_glcm_final.joblib` files are alternative model versions also present in `MODELS/`.

---

## 📂 Dataset Setup

Large dataset files are **excluded from the Git repository** (via `.gitignore`). To run the project:

1. **BRISC 2025 Dataset** — Place the BRISC 2025 dataset in the `brisc2025/` directory at the project root. The dataset can be obtained from the [BRISC 2025 publication](https://arxiv.org/abs/2506.14318).

2. **Processed Datasets** — The `DATASET/` directory must contain the processed data generated by running the preprocessing and feature extraction notebooks. The expected structure is:

   ```
   DATASET/
   ├── Classification/train/{glioma,meningioma,pituitary}/
   ├── Classification/test/{glioma,meningioma,pituitary}/
   ├── Segmentation/train/{images,masks}/
   ├── Segmentation/test/{images,masks}/
   ├── Preprocessed_segmented/train/{images,masks}/
   ├── Preprocessed_segmented/test/{images,masks}/
   └── Features(1)/{scaler.pkl, pca.pkl, glcm_*.csv, pca_*.csv}
   ```

3. **For inference only** — The web application requires `DATASET/Classification/test/` for the sample image demo feature, and the `scaler.pkl` / `pca.pkl` files in `DATASET/Features(1)/`.

---

## 📈 Results / Evaluation

### U-Net Segmentation (Test Set)

*Evaluated in [`U-Net_Evaluation.ipynb`](NOTEBOOKS/U-Net_Evaluation.ipynb)*

| Metric | Score | Percentage |
|--------|-------|------------|
| **Dice Coefficient** | 0.8569 | 85.69% |
| **IoU (Intersection over Union)** | 0.7496 | 74.96% |
| **Precision** | 0.8902 | 89.02% |
| **Recall / Sensitivity** | 0.8260 | 82.60% |
| **Specificity** | 0.9978 | 99.78% |
| **F1 Score** | 0.8569 | 85.69% |
| **Pixel Accuracy** | 0.9942 | 99.42% |

### SVM Classification (Test Set — 830 samples)

*Evaluated in [`Training_SVM.ipynb`](NOTEBOOKS/Training_SVM.ipynb)*

| Metric | Score |
|--------|-------|
| **Accuracy** | 83.86% |
| **Precision** (weighted) | 84.14% |
| **Recall** (weighted) | 83.73% |
| **F1-Score** (weighted) | 83.90% |

**Per-Class Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.87 | 0.82 | 0.85 | 234 |
| Meningioma | 0.85 | 0.88 | 0.87 | 301 |
| Pituitary | 0.80 | 0.81 | 0.80 | 295 |

### KNN Classification (Test Set — 830 samples)

*Evaluated in [`Training_KNN_Final.ipynb`](NOTEBOOKS/Training_KNN_Final.ipynb)*

| Metric | Score |
|--------|-------|
| **Accuracy** | 75.78% |
| **Precision** (weighted) | 76.98% |
| **Recall** (weighted) | 75.78% |
| **F1-Score** (weighted) | 75.67% |
| **Macro AUC** | 0.9107 |
| **Weighted AUC** | 0.9113 |

**Per-Class Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.81 | 0.60 | 0.69 | 234 |
| Meningioma | 0.84 | 0.82 | 0.83 | 301 |
| Pituitary | 0.67 | 0.82 | 0.73 | 295 |

**Best Hyperparameters Found:**

| Model | Parameters |
|-------|------------|
| SVM | `C = 10`, `kernel = linear` |
| KNN | `n_neighbors = 7`, `weights = distance`, `metric = manhattan` |

---

## 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow / Keras (U-Net architecture) |
| **Machine Learning** | scikit-learn (SVM, KNN, PCA, StandardScaler, GridSearchCV) |
| **Image Processing** | OpenCV (`cv2`), scikit-image (`graycomatrix`, `graycoprops`) |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic |
| **Frontend Framework** | Streamlit |
| **Data Handling** | NumPy, Pandas |
| **Serialization** | joblib, Keras `.keras` format |
| **Visualization** | Matplotlib, Pillow (PIL) |
| **HTTP Client** | Requests |
| **Development** | Jupyter Notebooks |

---

## ⚠️ Limitations / Notes

- **GPU Recommended** — U-Net inference benefits significantly from GPU acceleration. CPU inference is functional but slower.
- **No "No Tumor" Class at Inference** — The deployed classification pipeline targets only three tumor types (`glioma`, `meningioma`, `pituitary`). It does not classify "no tumor" cases, even though the BRISC 2025 dataset includes a no-tumor class (used for segmentation training).
- **Large Model Files** — The U-Net `.keras` files are approximately 355 MB each. They are excluded from version control and must be obtained separately.
- **Fixed Input Resolution** — All input images are resized to `256 × 256` pixels during preprocessing.
- **Dataset Dependency** — Running the full training pipeline requires the BRISC 2025 dataset. For inference only, the trained model artifacts suffice.
- **No `requirements.txt`** — Dependencies are not listed in a lockfile; they must be installed manually based on the libraries documented above.
- **The KNN classifier (75.78%) underperforms the SVM classifier (83.86%)**. The dual-model approach allows comparison but should be interpreted accordingly.

---

## 🚀 Future Improvements

> *These are suggested enhancements and are **not currently implemented** in the project.*

- **Add a `requirements.txt` or `pyproject.toml`** for reproducible dependency management
- **Support for "No Tumor" classification** — Extend the classification pipeline to detect the absence of tumors
- **3D MRI volume support** — Process full 3D MRI volumes instead of individual 2D slices
- **Model ensembling** — Combine SVM and KNN predictions via weighted voting for improved accuracy
- **Advanced segmentation architectures** — Evaluate Attention U-Net, U-Net++, or transformer-based models (e.g., Swin-UNETR)
- **Data augmentation** — Apply on-the-fly augmentation during U-Net training for better generalization
- **Docker containerization** — Package the application for portable deployment
- **Batch inference** — Support batch processing of multiple MRI scans simultaneously
- **DICOM format support** — Accept standard medical imaging DICOM files directly
- **Model performance dashboard** — Integrate training history visualization and model comparison metrics

---

## 📄 License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Harsh Wadari

See the [LICENSE](LICENSE) file for full details.

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineStage(BaseModel):
    name: str
    status: str = "completed"
    description: str


class ClassificationModelResult(BaseModel):
    model_name: str
    prediction: str
    score: float = Field(..., description="Primary model score. Unit depends on score_type.")
    score_type: str = Field(..., description="How the score was produced by the trained model")
    score_unit: str = Field("", description="Display unit for score, for example '%'")
    probabilities: Dict[str, float] = Field(default_factory=dict)
    decision_scores: Optional[Dict[str, float]] = None


class PipelineImages(BaseModel):
    original: str = Field(..., description="Base64 data URL for original image")
    preprocessed: str = Field(..., description="Base64 data URL for preprocessed image")
    segmentation_mask: str = Field(..., description="Base64 data URL for U-Net predicted mask")
    extracted_roi: str = Field(..., description="Base64 data URL for extracted tumor ROI")
    roi_overlay: str = Field(..., description="Base64 data URL for ROI blend with yellow contour")


class PipelineMetadata(BaseModel):
    glcm_features_extracted: bool = True
    glcm_features_count: int = 96
    glcm_feature_names: List[str] = Field(default_factory=list)
    glcm_feature_values: List[float] = Field(default_factory=list)
    pca_applied: bool = True
    pca_components_count: int = 5
    pca_components: List[float] = Field(default_factory=list)
    segmentation_model: str = "U-Net"
    tumor_detected: bool = True
    tumor_pixel_count: int = 0
    tumor_area_percentage: float = 0.0
    execution_time_ms: float = 0.0


class PredictionResponse(BaseModel):
    status: str = "success"
    stages: List[PipelineStage] = Field(default_factory=list)
    images: PipelineImages
    svm: ClassificationModelResult
    knn: ClassificationModelResult
    metadata: PipelineMetadata


class SystemInfoResponse(BaseModel):
    status: str = "online"
    service: str = "Brain Tumor Analysis API"
    version: str = "1.0.0"
    target_classes: List[str]
    input_resolution: str = "256x256"
    models_loaded: bool
    classifier_notes: Dict[str, str] = Field(default_factory=dict)


class SampleImageInfo(BaseModel):
    class_name: str
    filename: str
    relative_path: str

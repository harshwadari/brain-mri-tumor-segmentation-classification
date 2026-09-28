"""Independent, inference-only DWT artifacts; no GLCM artifacts are reused."""
import joblib
import numpy as np
import pandas as pd

from BACKEND.config import DWT_FEATURES_DIR, DWT_SCALER_PATH, DWT_PCA_PATH, DWT_SVM_MODEL_PATH, DWT_KNN_MODEL_PATH
from BACKEND.features import create_dwt_feature_names


class DWTModelManager:
    def __init__(self):
        self.scaler = self.pca = self.svm = self.knn = None
        self.feature_names = create_dwt_feature_names()
        self.pca_feature_names = []

    def load_models(self):
        if self.svm is not None:
            return
        scaler = joblib.load(DWT_SCALER_PATH)
        pca = joblib.load(DWT_PCA_PATH)
        svm = joblib.load(DWT_SVM_MODEL_PATH)
        knn = joblib.load(DWT_KNN_MODEL_PATH)
        pca_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
        # The scaler was fitted on arrays: CSV headers establish its input order.
        for prefix, expected in (("dwt", self.feature_names), ("dwt_pca", pca_names)):
            for split in ("train", "test"):
                columns = pd.read_csv(DWT_FEATURES_DIR / f"{prefix}_{split}.csv", nrows=0).columns
                actual = [c for c in columns if c not in ("class", "filename")]
                if actual != expected:
                    raise ValueError(f"DWT {prefix} {split} feature columns/order do not match training.")
        if scaler.n_features_in_ != len(self.feature_names) or pca.n_features_in_ != len(self.feature_names):
            raise ValueError("DWT scaler/PCA input dimensions do not match the 20 extracted features.")
        if svm.n_features_in_ != len(pca_names) or list(svm.feature_names_in_) != pca_names:
            raise ValueError("DWT SVM columns do not match the saved DWT PCA output.")
        if knn.n_features_in_ != len(pca_names):
            raise ValueError("DWT KNN dimensions do not match the saved DWT PCA output.")
        self.scaler, self.pca, self.svm, self.knn = scaler, pca, svm, knn
        self.pca_feature_names = pca_names

    def classify_features(self, features):
        self.load_models()
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError("DWT requires 20 finite features in training order.")
        scaled = self.scaler.transform(values.reshape(1, -1))
        projected = self.pca.transform(scaled)
        if not np.isfinite(projected).all():
            raise ValueError("DWT PCA produced non-finite coordinates.")
        frame = pd.DataFrame(projected.astype(np.float32), columns=self.pca_feature_names)
        svm_result = self._classify(self.svm, frame, "DWT + PCA Support Vector Machine (SVM)", "SVM predict_proba probability")
        # Training_KNN_DWT uses float32 arrays, without DataFrame column names.
        knn_result = self._classify(self.knn, frame.to_numpy(),
                                    f"DWT + PCA K-Nearest Neighbors (KNN, k={self.knn.n_neighbors})",
                                    "KNN predict_proba posterior probability")
        return projected[0].tolist(), svm_result, knn_result

    @staticmethod
    def _classify(model, values, name, score_type):
        prediction = model.predict(values)[0]
        classes = list(model.classes_)
        probabilities = model.predict_proba(values)[0]
        return {
            "model_name": name,
            "prediction": str(prediction),
            "score": round(float(probabilities[classes.index(prediction)] * 100), 2),
            "score_type": score_type,
            "score_unit": "%",
            "probabilities": {str(c): round(float(p * 100), 2) for c, p in zip(classes, probabilities)},
            "decision_scores": None,
        }


dwt_model_manager = DWTModelManager()

"""Run with python -m unittest discover -s tests -v (uses local saved artifacts)."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from BACKEND.config import DATASET_DIR, PROJECT_ROOT
from BACKEND.dwt import DWTModelManager
from BACKEND.features import extract_dwt_features, extract_glcm_features
from BACKEND.main import app, model_manager, dwt_model_manager


class DWTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = DWTModelManager()
        cls.manager.load_models()

    def test_saved_train_and_test_rows_match_extraction_and_projection(self):
        for split in ("train", "test"):
            raw = pd.read_csv(DATASET_DIR / "Features_DWT" / f"dwt_{split}.csv")
            projected = pd.read_csv(DATASET_DIR / "Features_DWT" / f"dwt_pca_{split}.csv")
            for _, row in raw.groupby("class").head(2).iterrows():
                with self.subTest(split=split, filename=row["filename"]):
                    path = DATASET_DIR / "Tumor_ROI_Morphology" / split / row["class"] / row["filename"]
                    roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    self.assertIsNotNone(roi)
                    features = extract_dwt_features(roi)
                    np.testing.assert_allclose(features, row[self.manager.feature_names].to_numpy(float),
                                               rtol=2e-5, atol=1e-6)
                    values, result, knn_result = self.manager.classify_features(features)
                    saved = projected[(projected["filename"] == row["filename"]) &
                                      (projected["class"] == row["class"])]
                    self.assertEqual(len(saved), 1)
                    np.testing.assert_allclose(values, saved[self.manager.pca_feature_names].iloc[0],
                                               rtol=2e-5, atol=1e-5)
                    expected = self.manager.svm.predict(saved[self.manager.pca_feature_names].astype(np.float32))[0]
                    self.assertEqual(result["prediction"], expected)
                    expected_knn = self.manager.knn.predict(saved[self.manager.pca_feature_names].to_numpy(dtype=np.float32))[0]
                    self.assertEqual(knn_result["prediction"], expected_knn)
                    self.assertAlmostEqual(sum(knn_result["probabilities"].values()), 100, delta=0.02)
                    self.assertAlmostEqual(sum(result["probabilities"].values()), 100, delta=0.02)

    def test_empty_and_constant_rois(self):
        self.assertIsNone(extract_dwt_features(np.zeros((16, 16), np.uint8)))
        for shape in ((1, 1), (17, 19), (16, 16)):
            values = extract_dwt_features(np.full(shape, 120, np.uint8))
            self.assertEqual(values.shape, (20,))
            self.assertTrue(np.isfinite(values).all())

    def test_invalid_features_rejected(self):
        for features in (np.zeros(19), np.full(20, np.nan), np.full(20, np.inf)):
            with self.assertRaises(ValueError):
                self.manager.classify_features(features)

    def test_inconsistent_csv_order_rejected(self):
        with patch("BACKEND.dwt.pd.read_csv", return_value=pd.DataFrame(columns=["wrong"])):
            with self.assertRaisesRegex(ValueError, "columns/order"):
                DWTModelManager().load_models()


class APIAndFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_manager.load_all_models()
        cls.client = TestClient(app)
        cls.image = next((DATASET_DIR / "Classification" / "test" / "glioma").glob("*.jpg")).read_bytes()
        response = cls.client.post("/api/predict", files={"file": ("scan.jpg", cls.image, "image/jpeg")})
        if response.status_code != 200:
            raise AssertionError(response.text)
        cls.result = response.json()

    def test_real_upload_preserves_glcm_and_adds_dwt(self):
        result = self.result
        self.assertTrue({"svm", "knn", "images", "metadata", "stages", "status"} <= result.keys())
        self.assertEqual(result["metadata"]["dwt_status"], "success")
        self.assertEqual(len(result["metadata"]["dwt_feature_values"]), 20)
        self.assertEqual(len(result["metadata"]["dwt_pca_components"]), 15)
        self.assertEqual(len(result["images"]), 5)
        self.assertEqual(result["metadata"]["pca_components_count"], 50)
        self.assertEqual(len(result["metadata"]["glcm_feature_values"]), 96)
        _, expected_svm, expected_knn = dwt_model_manager.classify_features(result["metadata"]["dwt_feature_values"])
        self.assertEqual(result["dwt_svm"], expected_svm)
        self.assertEqual(result["dwt_knn"], expected_knn)
        ensemble = result["ensemble"]
        self.assertEqual(ensemble["status"], "success")
        for label in ("glioma", "meningioma", "pituitary"):
            expected = sum(result[key]["probabilities"][label] / sum(result[key]["probabilities"].values())
                           for key in ("svm", "knn", "dwt_svm", "dwt_knn")) * 25
            self.assertAlmostEqual(ensemble["probabilities"][label], expected)
        from FRONTEND.utils import base64_to_pil
        roi = np.asarray(base64_to_pil(result["images"]["extracted_roi"]).convert("L"))
        np.testing.assert_allclose(extract_dwt_features(roi), result["metadata"]["dwt_feature_values"])
        _, svm, knn = model_manager.classify_features(extract_glcm_features(roi))
        self.assertEqual(result["svm"]["prediction"], svm["prediction"])
        self.assertEqual(result["svm"]["score"], svm["score"])
        self.assertEqual(result["knn"]["prediction"], knn["prediction"])

    def test_dwt_failure_retains_existing_results(self):
        mask = np.ones((256, 256), np.uint8)
        with patch.object(model_manager, "segment_image", return_value=(mask, mask, mask, mask)), \
             patch.object(dwt_model_manager, "classify_features", side_effect=FileNotFoundError("missing DWT scaler")):
            response = self.client.post("/api/predict", files={"file": ("scan.jpg", self.image)})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNone(result["dwt_svm"])
        self.assertIsNone(result["dwt_knn"])
        self.assertEqual(result["ensemble"]["status"], "unavailable")
        self.assertEqual(result["metadata"]["dwt_status"], "error")
        self.assertIn("missing DWT scaler", result["metadata"]["dwt_error"])
        self.assertTrue(result["svm"]["prediction"])
        self.assertTrue(result["knn"]["prediction"])

    def test_invalid_upload_and_empty_roi(self):
        self.assertEqual(self.client.post("/api/predict", files={"file": ("bad.jpg", b"bad")}).status_code, 400)
        mask = np.zeros((256, 256), np.uint8)
        with patch.object(model_manager, "segment_image", return_value=(mask, mask, mask, mask)):
            response = self.client.post("/api/predict", files={"file": ("scan.jpg", self.image)})
        self.assertEqual(response.status_code, 422)

    def test_reference_tables_preserve_actual_columns_and_rows(self):
        for key, folder, prefix in (("glcm", "Features(1)", "glcm"),
                                     ("glcm_pca", "Features(1)", "pca"),
                                     ("dwt", "Features_DWT", "dwt"),
                                     ("dwt_pca", "Features_DWT", "dwt_pca")):
            for split in ("train", "test"):
                table = self.client.get(f"/api/feature-tables/{key}?split={split}&offset=1&limit=2").json()
                expected = pd.read_csv(DATASET_DIR / folder / f"{prefix}_{split}.csv")
                self.assertEqual(table["columns"], list(expected.columns))
                self.assertEqual(table["rows"], expected.iloc[1:3].to_dict("records"))
        self.assertEqual(self.client.get("/api/feature-tables/unknown").status_code, 404)
        self.assertEqual(self.client.get("/api/feature-tables/dwt?split=bad").status_code, 422)

    def test_frontend_existing_and_dwt_sections(self):
        with patch("FRONTEND.utils.check_backend_health", return_value=False):
            ui = AppTest.from_file(str(PROJECT_ROOT / "FRONTEND/app.py"))
            ui.session_state["analysis_result"] = self.result
            ui.run(timeout=30)
        self.assertEqual(len(ui.exception), 0)
        texts = " ".join(item.value for item in ui.markdown)
        for label in ("Classification Result", "DWT Classification Result", "Extracted GLCM Feature Table",
                      "Principal Component Analysis", "DWT Feature Table", "DWT + PCA Table",
                      "5-Stage Image Processing Pipeline"):
            self.assertIn(label, texts)
        self.assertEqual(len(ui.dataframe), 5)
        self.assertEqual([len(table.value) for table in ui.dataframe], [50, 96, 20, 15, 4])
        self.assertGreater(texts.index("Final Ensemble Prediction"), texts.index("DWT Classification Result"))
        self.assertIn("Final Ensemble Prediction", texts)
        self.assertIn("50 Components", texts)
        self.assertIn("96 Features", texts)
        self.assertIn("KNN, k=7", texts)
        self.assertTrue(ui.expander[0].proto.expanded)

    def test_frontend_missing_dwt_and_reference_loading(self):
        script = """
import streamlit as st
from FRONTEND.dwt import render_dwt_result, render_reference_tables
render_dwt_result(st.session_state['result'])
render_reference_tables()
"""
        ui = AppTest.from_string(script)
        ui.session_state["result"] = {"metadata": {"dwt_error": "Missing DWT scaler"}}
        ui.run()
        self.assertEqual(len(ui.exception), 0)
        self.assertIn("Missing DWT scaler", ui.warning[0].value)
        def table(key, split, offset):
            return self.client.get(f"/api/feature-tables/{key}?split={split}&offset={offset}").json()
        with patch("FRONTEND.dwt.get_feature_table", side_effect=table):
            ui.checkbox[0].check().run()
        self.assertEqual(len(ui.exception), 0)
        self.assertEqual(len(ui.dataframe), 4)
        with patch("FRONTEND.dwt.get_feature_table", side_effect=RuntimeError("offline")):
            ui.run()
        self.assertEqual(len(ui.exception), 0)
        self.assertEqual(len(ui.warning), 5)


if __name__ == "__main__":
    unittest.main()

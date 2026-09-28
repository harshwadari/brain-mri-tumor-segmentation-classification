import copy
import unittest
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from BACKEND.ensemble import combine_predictions, MODEL_KEYS


class EnsembleTests(unittest.TestCase):
    def models(self):
        return {key: SimpleNamespace(prediction="glioma", probabilities={
            "glioma": 60.0, "meningioma": 30.0, "pituitary": 10.0}) for key in MODEL_KEYS}

    def test_soft_vote_uses_probabilities_and_class_names(self):
        models = self.models()
        models["knn"].probabilities = {"pituitary": 10., "meningioma": 80., "glioma": 10.}
        models["dwt_knn"].probabilities = {"meningioma": 80., "glioma": 10., "pituitary": 10.}
        before = copy.deepcopy(models)
        result = combine_predictions(models)
        self.assertEqual(result.prediction, "meningioma")
        self.assertAlmostEqual(result.score, 55.)
        for label, expected in {"glioma": 35., "meningioma": 55., "pituitary": 10.}.items():
            self.assertAlmostEqual(result.probabilities[label], expected)
        self.assertAlmostEqual(sum(result.weights.values()), 1.)
        self.assertEqual(models, before)

    def test_rounding_and_ties(self):
        models = self.models()
        for model in models.values():
            model.probabilities = dict.fromkeys(("glioma", "meningioma", "pituitary"), 33.33)
        result = combine_predictions(models)
        self.assertAlmostEqual(sum(result.probabilities.values()), 100.)
        self.assertEqual(result.prediction, "glioma")

    def test_missing_or_invalid_distributions_do_not_make_partial_ensemble(self):
        for invalid in (None, {}, {"glioma": 100.},
                        {"glioma": float("nan"), "meningioma": 0., "pituitary": 0.},
                        {"glioma": -1., "meningioma": 51., "pituitary": 50.},
                        {"glioma": .6, "meningioma": .3, "pituitary": .1}):
            models = self.models()
            models["dwt_knn"] = None if invalid is None else SimpleNamespace(prediction="glioma", probabilities=invalid)
            result = combine_predictions(models)
            self.assertEqual(result.status, "unavailable")
            self.assertIsNone(result.prediction)
            self.assertIsNone(result.score)
            self.assertTrue(result.error)

    def test_frontend_success_old_response_and_missing_model(self):
        script = """
import streamlit as st
from FRONTEND.ensemble import render_ensemble_result
render_ensemble_result(st.session_state['result'])
"""
        ui = AppTest.from_string(script)
        ui.session_state["result"] = {"ensemble": combine_predictions(self.models()).model_dump()}
        ui.run()
        self.assertEqual(len(ui.exception), 0)
        self.assertEqual(len(ui.metric), 3)
        self.assertEqual(len(ui.dataframe[0].value), 4)
        self.assertIn("Weighted Soft Voting", ui.caption[0].value)
        ui.session_state["result"] = {}
        ui.run()
        self.assertEqual(len(ui.info), 1)
        self.assertEqual(len(ui.exception), 0)
        ui.session_state["result"] = {"ensemble": combine_predictions({}).model_dump()}
        ui.run()
        self.assertEqual(len(ui.warning), 1)
        self.assertEqual(len(ui.exception), 0)

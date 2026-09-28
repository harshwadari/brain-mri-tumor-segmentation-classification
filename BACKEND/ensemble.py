"""Combine existing percentage probabilities without changing model inference."""
import math

from BACKEND.config import CLASS_NAMES
from BACKEND.schemas import EnsembleResult


MODEL_KEYS = ("svm", "knn", "dwt_svm", "dwt_knn")
# Notebook CV scores mix macro-F1 and accuracy; the GLCM KNN notebook also
# saves a different artifact than the application loads. No comparable,
# artifact-matched validation weights exist for all four deployed classifiers.
WEIGHTS = dict.fromkeys(MODEL_KEYS, 0.25)
WEIGHT_SOURCE = "Equal weights: no comparable validation metrics for all four deployed models."


def combine_predictions(models):
    result = EnsembleResult(weights=WEIGHTS, weight_source=WEIGHT_SOURCE)
    vectors = {}
    for key in MODEL_KEYS:
        model = models.get(key)
        if model is None:
            result.error = "All four model results are required for the ensemble."
            return result
        probabilities = model.probabilities
        if set(probabilities) != set(CLASS_NAMES):
            result.error = f"{key}: class probabilities are missing or inconsistent."
            return result
        values = [probabilities[label] for label in CLASS_NAMES]
        if (not all(math.isfinite(v) and 0 <= v <= 100 for v in values)
                or not math.isclose(sum(values), 100.0, abs_tol=0.05)):
            result.error = f"{key}: invalid probability distribution."
            return result
        # Individual API results round to two decimals; normalize that rounding
        # drift, aligning by class name rather than dictionary/estimator order.
        vectors[key] = {label: probabilities[label] / sum(values) for label in CLASS_NAMES}
        result.model_predictions[key] = model.prediction
    combined = {
        label: 100.0 * sum(WEIGHTS[key] * vectors[key][label] for key in MODEL_KEYS)
        / sum(WEIGHTS.values()) for label in CLASS_NAMES
    }
    # Stable CLASS_NAMES order resolves an exact tie deterministically.
    result.prediction = max(combined, key=combined.get)
    result.score = combined[result.prediction]
    result.probabilities = combined
    result.status = "success"
    return result

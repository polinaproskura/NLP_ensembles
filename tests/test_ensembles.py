"""Tests for attention_ensembles.ensembles — pure numpy, no downloads."""

import numpy as np
import pytest

from attention_ensembles.ensembles import (
    weighted_prediction,
    equal_weight_prediction,
    logits_to_probs,
    ensemble_accuracy,
)


def _make_predictions(n_models=3, n_samples=10, n_classes=2, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_models, n_samples, n_classes))


class TestWeightedPrediction:
    def test_output_shape(self):
        preds = _make_predictions(n_models=3, n_samples=10, n_classes=2)
        result = weighted_prediction(preds, np.array([1.0, 1.0, 1.0]))
        assert result.shape == (10, 2)

    def test_equal_weights_is_average(self):
        preds = _make_predictions(n_models=3, n_samples=10)
        equal = weighted_prediction(preds, np.array([1.0, 1.0, 1.0]))
        mean = preds.mean(axis=0)
        np.testing.assert_allclose(equal, mean, atol=1e-10)

    def test_weights_are_normalised(self):
        preds = _make_predictions(n_models=2, n_samples=5)
        r1 = weighted_prediction(preds, np.array([1.0, 3.0]))
        r2 = weighted_prediction(preds, np.array([0.25, 0.75]))
        np.testing.assert_allclose(r1, r2, atol=1e-10)

    def test_single_model_weight_one(self):
        preds = _make_predictions(n_models=2, n_samples=5)
        result = weighted_prediction(preds, np.array([1.0, 0.0]))
        np.testing.assert_allclose(result, preds[0], atol=1e-10)

    def test_works_with_more_classes(self):
        preds = _make_predictions(n_models=2, n_samples=5, n_classes=4)
        result = weighted_prediction(preds, np.array([0.5, 0.5]))
        assert result.shape == (5, 4)


class TestEqualWeightPrediction:
    def test_is_mean(self):
        preds = _make_predictions(n_models=4, n_samples=8)
        result = equal_weight_prediction(preds)
        np.testing.assert_allclose(result, preds.mean(axis=0), atol=1e-10)

    def test_output_shape(self):
        preds = _make_predictions(n_models=5, n_samples=12, n_classes=3)
        assert equal_weight_prediction(preds).shape == (12, 3)


class TestLogitsToProbs:
    def test_output_range(self):
        logits = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        probs = logits_to_probs(logits)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_zero_gives_half(self):
        assert logits_to_probs(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_preserves_shape(self):
        logits = np.ones((3, 5, 2))
        assert logits_to_probs(logits).shape == (3, 5, 2)

    def test_large_positive_near_one(self):
        assert logits_to_probs(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-10)


class TestEnsembleAccuracy:
    def test_perfect_predictions(self):
        # Two models both perfectly predict: class 0 for first 5, class 1 for next 5
        preds = np.array([
            [[10, -10]] * 5 + [[-10, 10]] * 5,
            [[10, -10]] * 5 + [[-10, 10]] * 5,
        ], dtype=float)
        labels = np.array([0] * 5 + [1] * 5)
        assert ensemble_accuracy(preds, np.array([0.5, 0.5]), labels) == 1.0

    def test_returns_float_in_range(self):
        preds = _make_predictions(n_models=3, n_samples=20)
        labels = np.array([0] * 10 + [1] * 10)
        acc = ensemble_accuracy(preds, np.ones(3), labels)
        assert 0.0 <= acc <= 1.0

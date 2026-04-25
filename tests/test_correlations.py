"""Tests for attention_ensembles.correlations — pure numpy, no downloads."""

import numpy as np
import pytest

from attention_ensembles.correlations import (
    prediction_correlation,
    prediction_correlation_matrix,
    extract_submatrix,
    correlation_based_variance,
)


def _make_predictions(n_models=3, n_samples=20, n_classes=2, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_models, n_samples, n_classes))


class TestPredictionCorrelation:
    def test_self_correlation_positive(self):
        preds = _make_predictions(n_models=1, n_samples=50)[0]
        c = prediction_correlation(preds, preds)
        assert c > 0

    def test_returns_scalar(self):
        preds = _make_predictions(n_models=2, n_samples=20)
        c = prediction_correlation(preds[0], preds[1])
        assert isinstance(c, float)

    def test_symmetric(self):
        preds = _make_predictions(n_models=2, n_samples=20)
        c_ab = prediction_correlation(preds[0], preds[1])
        c_ba = prediction_correlation(preds[1], preds[0])
        assert c_ab == pytest.approx(c_ba)


class TestPredictionCorrelationMatrix:
    def test_shape(self):
        preds = _make_predictions(n_models=4, n_samples=20)
        matrix = prediction_correlation_matrix(preds)
        assert matrix.shape == (4, 4)

    def test_symmetric(self):
        preds = _make_predictions(n_models=3, n_samples=30)
        matrix = prediction_correlation_matrix(preds)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_diagonal_positive(self):
        preds = _make_predictions(n_models=3, n_samples=30)
        matrix = prediction_correlation_matrix(preds)
        assert np.all(np.diag(matrix) > 0)

    def test_single_model(self):
        preds = _make_predictions(n_models=1, n_samples=10)
        matrix = prediction_correlation_matrix(preds)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] > 0


class TestExtractSubmatrix:
    def test_correct_shape(self):
        full = np.arange(25).reshape(5, 5).astype(float)
        sub = extract_submatrix(full, [0, 2, 4])
        assert sub.shape == (3, 3)

    def test_correct_values(self):
        full = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)
        sub = extract_submatrix(full, [0, 2])
        expected = np.array([[1, 3], [7, 9]], dtype=float)
        np.testing.assert_array_equal(sub, expected)

    def test_single_index(self):
        full = np.eye(5)
        sub = extract_submatrix(full, [3])
        assert sub.shape == (1, 1)
        assert sub[0, 0] == 1.0

    def test_all_indices_returns_full(self):
        full = np.random.rand(4, 4)
        sub = extract_submatrix(full, [0, 1, 2, 3])
        np.testing.assert_array_equal(sub, full)


class TestCorrelationBasedVariance:
    def test_equal_weights_identity_matrix(self):
        # Identity correlation matrix + equal weights
        corr = np.eye(3)
        weights = np.array([1/3, 1/3, 1/3])
        var = correlation_based_variance(corr, weights)
        # w^T I w = sum(w_i^2) = 3 * (1/3)^2 = 1/3
        assert var == pytest.approx(1/3, abs=1e-10)

    def test_single_model(self):
        corr = np.array([[5.0]])
        weights = np.array([1.0])
        assert correlation_based_variance(corr, weights) == pytest.approx(5.0)

    def test_returns_scalar(self):
        corr = np.random.rand(4, 4)
        weights = np.ones(4) / 4
        assert isinstance(correlation_based_variance(corr, weights), float)

    def test_non_negative_for_psd_matrix(self):
        # Build a positive semi-definite matrix
        A = np.random.rand(3, 3)
        corr = A @ A.T
        weights = np.array([0.3, 0.5, 0.2])
        assert correlation_based_variance(corr, weights) >= 0

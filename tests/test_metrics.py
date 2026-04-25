"""Tests for attention_ensembles.metrics — pure numpy, no downloads."""

import numpy as np
import pytest

from attention_ensembles.metrics import (
    accuracy_rejection_curve,
    accuracy_rejection_auc,
    posterior_variance,
    posterior_expectation,
)


class TestAccuracyRejectionCurve:
    def test_starts_at_zero_rejection(self):
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = np.array([1, 1, 1, 0, 0])
        preds = np.array([1, 1, 1, 0, 0])
        rates, accs = accuracy_rejection_curve(probs, labels, preds, n_steps=5)
        assert rates[0] == 0.0

    def test_perfect_model_starts_at_one(self):
        probs = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 0, 1])
        preds = np.array([1, 0, 1])
        rates, accs = accuracy_rejection_curve(probs, labels, preds)
        assert accs[0] == 1.0

    def test_rejection_rate_increases(self):
        probs = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        preds = np.random.randint(0, 2, 100)
        rates, _ = accuracy_rejection_curve(probs, labels, preds, n_steps=10)
        assert rates == sorted(rates)

    def test_returns_two_equal_length_lists(self):
        probs = np.random.rand(50)
        labels = np.random.randint(0, 2, 50)
        preds = np.random.randint(0, 2, 50)
        rates, accs = accuracy_rejection_curve(probs, labels, preds, n_steps=10)
        assert len(rates) == len(accs)
        assert len(rates) > 1


class TestAccuracyRejectionAuc:
    def test_perfect_model_high_auc(self):
        # A perfect model with many samples should have AUC close to 1.0
        n = 200
        probs = np.linspace(0.99, 0.51, n)
        labels = np.array([1] * (n // 2) + [0] * (n // 2))
        preds = labels.copy()  # perfect predictions
        auc_val = accuracy_rejection_auc(probs, labels, preds, n_steps=100)
        assert auc_val > 0.95

    def test_returns_scalar(self):
        probs = np.random.rand(50)
        labels = np.random.randint(0, 2, 50)
        preds = np.random.randint(0, 2, 50)
        auc_val = accuracy_rejection_auc(probs, labels, preds)
        assert isinstance(auc_val, float)

    def test_bounded_zero_to_one(self):
        probs = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        preds = np.random.randint(0, 2, 100)
        auc_val = accuracy_rejection_auc(probs, labels, preds)
        assert 0.0 <= auc_val <= 1.0


class TestPosteriorVariance:
    def test_returns_non_negative(self):
        preds = np.random.rand(100, 2)
        labels = np.array([0] * 50 + [1] * 50)
        pv = posterior_variance(preds, labels, seed=42)
        assert pv >= 0.0

    def test_reproducible_with_seed(self):
        preds = np.random.rand(100, 2)
        labels = np.array([0] * 50 + [1] * 50)
        pv1 = posterior_variance(preds, labels, seed=42)
        pv2 = posterior_variance(preds, labels, seed=42)
        assert pv1 == pv2

    def test_returns_float(self):
        preds = np.random.rand(50, 2)
        labels = np.array([0] * 25 + [1] * 25)
        assert isinstance(posterior_variance(preds, labels, seed=0), float)


class TestPosteriorExpectation:
    def test_confident_model_low_pe(self):
        # Model always predicts class 0 with high probability
        preds = np.column_stack([np.ones(100) * 0.99, np.ones(100) * 0.01])
        pe = posterior_expectation(preds, seed=42)
        assert pe < 0.1

    def test_uncertain_model_high_pe(self):
        # Model always predicts ~0.5 for both classes
        preds = np.column_stack([np.ones(100) * 0.5, np.ones(100) * 0.5])
        pe = posterior_expectation(preds, seed=42)
        assert pe > 0.3

    def test_returns_float(self):
        preds = np.random.rand(50, 2)
        assert isinstance(posterior_expectation(preds, seed=0), float)

    def test_reproducible_with_seed(self):
        preds = np.random.rand(50, 2)
        pe1 = posterior_expectation(preds, seed=42)
        pe2 = posterior_expectation(preds, seed=42)
        assert pe1 == pe2

"""Tests for attention_ensembles.models — fast, no downloads required."""

import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from attention_ensembles.models import (
    build_weak_model,
    create_trainer,
    DEFAULT_TRAINING_KWARGS,
)


def _make_tiny_distilbert(n_layers: int = 6, num_labels: int = 2):
    """Build a tiny DistilBERT model for unit testing (no download)."""
    config = DistilBertConfig(
        vocab_size=100,
        dim=32,
        n_layers=n_layers,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=32,
        num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(config)


class TestBuildWeakModel:
    def test_truncates_to_one_layer(self):
        model = _make_tiny_distilbert(n_layers=6)
        weak = build_weak_model(model, n_layers=1)
        assert len(weak.distilbert.transformer.layer) == 1

    def test_truncates_to_three_layers(self):
        model = _make_tiny_distilbert(n_layers=6)
        weak = build_weak_model(model, n_layers=3)
        assert len(weak.distilbert.transformer.layer) == 3

    def test_original_model_unchanged(self):
        model = _make_tiny_distilbert(n_layers=6)
        build_weak_model(model, n_layers=1)
        assert len(model.distilbert.transformer.layer) == 6

    def test_returns_deep_copy(self):
        model = _make_tiny_distilbert(n_layers=6)
        weak = build_weak_model(model, n_layers=1)
        assert weak is not model
        assert weak.distilbert is not model.distilbert

    def test_keeps_classifier_head(self):
        model = _make_tiny_distilbert(n_layers=6)
        weak = build_weak_model(model, n_layers=1)
        assert hasattr(weak, "classifier")
        assert weak.classifier.out_features == 2

    def test_all_layers_is_identity(self):
        model = _make_tiny_distilbert(n_layers=4)
        weak = build_weak_model(model, n_layers=4)
        assert len(weak.distilbert.transformer.layer) == 4

    def test_zero_layers_raises(self):
        model = _make_tiny_distilbert(n_layers=6)
        with pytest.raises(ValueError, match="out of range"):
            build_weak_model(model, n_layers=0)

    def test_too_many_layers_raises(self):
        model = _make_tiny_distilbert(n_layers=4)
        with pytest.raises(ValueError, match="out of range"):
            build_weak_model(model, n_layers=5)

    def test_non_distilbert_raises(self):
        """A model without .distilbert should fail with a clear message."""
        from unittest.mock import MagicMock
        fake_model = MagicMock(spec=[])  # no .distilbert attribute
        with pytest.raises(ValueError, match="DistilBERT"):
            build_weak_model(fake_model, n_layers=1)

    def test_forward_pass_still_works(self):
        """Weak model should produce valid logits on dummy input."""
        import torch

        model = _make_tiny_distilbert(n_layers=6)
        weak = build_weak_model(model, n_layers=1)

        dummy_input = torch.randint(0, 100, (2, 8))  # batch=2, seq_len=8
        dummy_mask = torch.ones_like(dummy_input)
        output = weak(input_ids=dummy_input, attention_mask=dummy_mask)

        assert output.logits.shape == (2, 2)


class TestDefaultTrainingKwargs:
    def test_has_expected_keys(self):
        expected = {"learning_rate", "per_device_train_batch_size",
                    "per_device_eval_batch_size", "num_train_epochs", "weight_decay"}
        assert expected == set(DEFAULT_TRAINING_KWARGS.keys())

    def test_learning_rate_is_reasonable(self):
        assert 1e-6 < DEFAULT_TRAINING_KWARGS["learning_rate"] < 1e-3

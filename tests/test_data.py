"""Tests for attention_ensembles.data — fast tests use mocks, slow ones download."""

import pytest
from datasets import Dataset, DatasetDict

from attention_ensembles.data import split_test, DEFAULT_MODEL_NAME


def _make_mock_imdb(n_train: int = 20, n_test: int = 20) -> DatasetDict:
    """Build a tiny mock IMDB-shaped DatasetDict for testing splits."""
    train = Dataset.from_dict({
        "text": [f"train review {i}" for i in range(n_train)],
        "label": [i % 2 for i in range(n_train)],
    })
    test = Dataset.from_dict({
        "text": [f"test review {i}" for i in range(n_test)],
        "label": [i % 2 for i in range(n_test)],
    })
    return DatasetDict({"train": train, "test": test})


class TestSplitTest:
    def test_returns_three_datasets(self):
        mock = _make_mock_imdb(n_train=20, n_test=20)
        train, val, test = split_test(mock)
        assert len(train) == 20
        assert len(val) + len(test) == 20

    def test_default_split_is_even(self):
        mock = _make_mock_imdb(n_train=20, n_test=100)
        train, val, test = split_test(mock, test_size=0.5)
        assert len(val) == 50
        assert len(test) == 50

    def test_custom_split_ratio(self):
        mock = _make_mock_imdb(n_train=10, n_test=100)
        _, val, test = split_test(mock, test_size=0.8)
        assert len(val) == 20
        assert len(test) == 80

    def test_train_is_original(self):
        mock = _make_mock_imdb(n_train=30, n_test=10)
        train, _, _ = split_test(mock)
        assert list(train["text"]) == [f"train review {i}" for i in range(30)]

    def test_reproducible_with_same_seed(self):
        mock = _make_mock_imdb(n_train=10, n_test=50)
        _, val1, test1 = split_test(mock, seed=42)
        _, val2, test2 = split_test(mock, seed=42)
        assert list(val1["text"]) == list(val2["text"])
        assert list(test1["text"]) == list(test2["text"])

    def test_different_seed_gives_different_split(self):
        mock = _make_mock_imdb(n_train=10, n_test=50)
        _, val1, _ = split_test(mock, seed=42)
        _, val2, _ = split_test(mock, seed=99)
        assert list(val1["text"]) != list(val2["text"])


class TestDefaultModelName:
    def test_is_distilbert(self):
        assert "distilbert" in DEFAULT_MODEL_NAME


@pytest.mark.slow
class TestLoadImdb:
    """These tests download the real IMDB dataset + tokeniser (~85 MB).
    Run with: pytest -m slow
    """

    def test_returns_three_items(self):
        from attention_ensembles.data import load_imdb
        tokenized, tokenizer, collator = load_imdb()
        assert "train" in tokenized
        assert "test" in tokenized
        assert tokenizer is not None
        assert collator is not None

    def test_tokenized_has_input_ids(self):
        from attention_ensembles.data import load_imdb
        tokenized, _, _ = load_imdb()
        assert "input_ids" in tokenized["train"].column_names

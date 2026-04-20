"""
Data loading and preprocessing for IMDB sentiment classification.

Extracts the shared boilerplate that every notebook duplicates:
tokenisation, collator setup, and train/val/test splitting.
"""

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    set_seed,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def load_imdb(
    model_name: str = DEFAULT_MODEL_NAME,
    seed: int = 42,
) -> tuple[DatasetDict, PreTrainedTokenizerBase, DataCollatorWithPadding]:
    """Load and tokenise the IMDB dataset.

    Args:
        model_name: HuggingFace model identifier for the tokeniser.
        seed: Random seed set before loading (for reproducibility).

    Returns:
        (tokenized_imdb, tokenizer, data_collator)
    """
    set_seed(seed)
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_imdb, tokenizer, data_collator


def split_test(
    tokenized_imdb: DatasetDict,
    test_size: float = 0.5,
    seed: int = 42,
) -> tuple:
    """Split the IMDB test set into validation and test halves.

    The original IMDB dataset has train (25k) and test (25k) splits but no
    validation set. This function splits the test portion into val + test
    so that ensemble weights can be tuned on val without touching test.

    Args:
        tokenized_imdb: Output of :func:`load_imdb`.
        test_size: Fraction of the original test set reserved for final test.
        seed: Random seed for the split.

    Returns:
        (train, val, test) datasets.
    """
    split = tokenized_imdb["test"].train_test_split(test_size=test_size, seed=seed)
    train = tokenized_imdb["train"]
    val = split["train"]
    test = split["test"]
    return train, val, test

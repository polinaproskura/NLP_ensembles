"""
Model loading, weak-model construction, and trainer creation.

Replaces the duplicated ``load_trainer`` / ``load_trainer_weak`` functions
that appear in every notebook with a configurable, testable interface.
"""

import copy

from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


DEFAULT_TRAINING_KWARGS = dict(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)


def load_model(
    checkpoint_path: str,
    num_labels: int = 2,
) -> AutoModelForSequenceClassification:
    """Load a fine-tuned classification model from a local or HF Hub checkpoint.

    Args:
        checkpoint_path: Local directory or HuggingFace Hub model ID.
        num_labels: Number of output classes.

    Returns:
        The loaded model (on CPU by default).
    """
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=num_labels
    )


def build_weak_model(
    model: AutoModelForSequenceClassification,
    n_layers: int = 1,
) -> AutoModelForSequenceClassification:
    """Truncate a DistilBERT model to its first *n_layers* transformer layers.

    This creates a "weak" model that retains only the initial transformer
    blocks. The original model is not modified — a deep copy is returned.

    Args:
        model: A DistilBERT-based sequence classification model.
        n_layers: Number of transformer layers to keep (default 1).

    Returns:
        A deep copy of the model with only the first *n_layers* layers.

    Raises:
        ValueError: If the model has fewer layers than requested or does not
            have the expected DistilBERT structure.
    """
    if not hasattr(model, "distilbert"):
        raise ValueError(
            "build_weak_model expects a DistilBERT model "
            f"(got {type(model).__name__} with no .distilbert attribute)"
        )

    total_layers = len(model.distilbert.transformer.layer)
    if n_layers < 1 or n_layers > total_layers:
        raise ValueError(
            f"n_layers={n_layers} out of range [1, {total_layers}]"
        )

    weak = copy.deepcopy(model)
    weak.distilbert.transformer.layer = nn.ModuleList(
        list(weak.distilbert.transformer.layer)[:n_layers]
    )
    return weak


def create_trainer(
    model: AutoModelForSequenceClassification,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_collator: DataCollatorWithPadding,
    seed: int = 42,
    output_dir: str = "./results",
    **training_kwargs,
) -> Trainer:
    """Create a HuggingFace Trainer with sensible defaults.

    Any key from ``DEFAULT_TRAINING_KWARGS`` can be overridden via
    ``**training_kwargs``.

    Args:
        model: The model to train or evaluate.
        train_dataset: Tokenised training split.
        eval_dataset: Tokenised evaluation split.
        tokenizer: Tokeniser (needed by the Trainer for padding).
        data_collator: Data collator returned by :func:`~attention_ensembles.data.load_imdb`.
        seed: Random seed for training.
        output_dir: Directory for checkpoints and logs.
        **training_kwargs: Override any default training argument.

    Returns:
        A configured :class:`~transformers.Trainer`.
    """
    merged = {**DEFAULT_TRAINING_KWARGS, **training_kwargs}
    args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        **merged,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

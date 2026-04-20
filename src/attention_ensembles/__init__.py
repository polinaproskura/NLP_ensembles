"""
attention_ensembles
===================
Ensemble weight optimisation for fine-tuned BERT models using topological
features of attention layers.

Typical usage::

    from attention_ensembles.data import load_imdb, split_test
    from attention_ensembles.models import load_model, build_weak_model, create_trainer

    tokenized, tokenizer, collator = load_imdb()
    train, val, test = split_test(tokenized)

    model = load_model("checkpoints/bert_1")
    weak  = build_weak_model(model, n_layers=1)

    trainer = create_trainer(model, train, test, tokenizer, collator)
    predictions = trainer.predict(val)
"""

from attention_ensembles.data import load_imdb, split_test
from attention_ensembles.models import load_model, build_weak_model, create_trainer

__all__ = [
    "load_imdb",
    "split_test",
    "load_model",
    "build_weak_model",
    "create_trainer",
]

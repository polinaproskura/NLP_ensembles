"""
attention_ensembles
===================
Ensemble weight optimisation for fine-tuned BERT models using topological
features of attention layers.

Typical usage::

    from attention_ensembles.data import load_imdb, split_test
    from attention_ensembles.models import load_model, build_weak_model, create_trainer
    from attention_ensembles.ensembles import weighted_prediction, ensemble_accuracy
    from attention_ensembles.correlations import prediction_correlation_matrix, correlation_based_variance
    from attention_ensembles.metrics import accuracy_rejection_curve, accuracy_rejection_auc

    tokenized, tokenizer, collator = load_imdb()
    train, val, test = split_test(tokenized)

    model = load_model("checkpoints/bert_1")
    weak  = build_weak_model(model, n_layers=1)

    trainer = create_trainer(model, train, test, tokenizer, collator)
    predictions = trainer.predict(val)
"""

from attention_ensembles.data import load_imdb, split_test
from attention_ensembles.models import load_model, build_weak_model, create_trainer
from attention_ensembles.ensembles import (
    weighted_prediction,
    equal_weight_prediction,
    ensemble_accuracy,
    logits_to_probs,
)
from attention_ensembles.metrics import (
    accuracy_rejection_curve,
    accuracy_rejection_auc,
    posterior_variance,
    posterior_expectation,
)
from attention_ensembles.correlations import (
    prediction_correlation,
    prediction_correlation_matrix,
    correlation_based_variance,
    extract_submatrix,
)

__all__ = [
    # data
    "load_imdb",
    "split_test",
    # models
    "load_model",
    "build_weak_model",
    "create_trainer",
    # ensembles
    "weighted_prediction",
    "equal_weight_prediction",
    "ensemble_accuracy",
    "logits_to_probs",
    # metrics
    "accuracy_rejection_curve",
    "accuracy_rejection_auc",
    "posterior_variance",
    "posterior_expectation",
    # correlations
    "prediction_correlation",
    "prediction_correlation_matrix",
    "correlation_based_variance",
    "extract_submatrix",
]

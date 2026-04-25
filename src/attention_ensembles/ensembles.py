"""
Ensemble prediction methods.

Provides weighted and equal-weight combination of model predictions,
plus sigmoid conversion from logits to probabilities.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def weighted_prediction(
    predictions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Combine model predictions using a weighted average.

    Args:
        predictions: Array of shape ``(n_models, n_samples, n_classes)``
            containing raw logits or probabilities from each model.
        weights: Array of shape ``(n_models,)`` — will be normalised to
            sum to 1.

    Returns:
        Weighted ensemble predictions of shape ``(n_samples, n_classes)``.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n_classes = predictions.shape[2]
    columns = []
    for cls in range(n_classes):
        columns.append(np.dot(weights, predictions[:, :, cls]))
    return np.column_stack(columns)


def equal_weight_prediction(predictions: np.ndarray) -> np.ndarray:
    """Combine model predictions using equal weights (simple average).

    Args:
        predictions: Array of shape ``(n_models, n_samples, n_classes)``.

    Returns:
        Averaged ensemble predictions of shape ``(n_samples, n_classes)``.
    """
    return predictions.mean(axis=0)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to probabilities via sigmoid.

    Args:
        logits: Array of any shape containing raw model outputs.

    Returns:
        Probabilities in [0, 1] with the same shape as input.
    """
    return 1.0 / (1.0 + np.exp(-logits))


def ensemble_accuracy(
    predictions: np.ndarray,
    weights: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute accuracy of a weighted ensemble.

    Convenience function that combines predictions, takes argmax, and
    compares to true labels.

    Args:
        predictions: Shape ``(n_models, n_samples, n_classes)``.
        weights: Shape ``(n_models,)``.
        true_labels: Shape ``(n_samples,)``.

    Returns:
        Accuracy as a float in [0, 1].
    """
    combined = weighted_prediction(predictions, weights)
    predicted_labels = np.argmax(combined, axis=1)
    return accuracy_score(true_labels, predicted_labels)

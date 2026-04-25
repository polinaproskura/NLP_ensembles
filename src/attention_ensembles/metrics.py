"""
Evaluation metrics for ensemble quality.

Provides accuracy-rejection curves (uncertainty calibration),
posterior variance, and posterior expectation — the metrics
used throughout the notebooks to compare weighting strategies.
"""

import numpy as np
from sklearn.metrics import accuracy_score, auc


def accuracy_rejection_curve(
    probs: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_steps: int = 10_000,
) -> tuple[list[float], list[float]]:
    """Compute an accuracy-rejection curve.

    Sorts samples by descending confidence (``probs``), then iteratively
    removes the least-confident samples and recomputes accuracy.  A good
    model's accuracy rises as uncertain samples are rejected.

    Args:
        probs: Per-sample confidence scores (e.g. ``np.max(softmax, axis=1)``).
        true_labels: Ground-truth labels.
        predicted_labels: Model-predicted labels.
        n_steps: Number of rejection steps (higher = smoother curve).

    Returns:
        (rejection_rates, accuracies) — two lists of equal length.
    """
    N = len(probs)
    step = max(1, N // n_steps)
    idx = np.argsort(probs)[::-1]

    r_rate = [0.0]
    r_accuracy = [accuracy_score(true_labels, predicted_labels)]

    for i in range(step, N, step):
        idx = idx[: (N - i)]
        r_rate.append(i / N)
        r_accuracy.append(accuracy_score(true_labels[idx], predicted_labels[idx]))

    return r_rate, r_accuracy


def accuracy_rejection_auc(
    probs: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_steps: int = 10_000,
) -> float:
    """Area under the accuracy-rejection curve (single scalar summary).

    Higher is better — means the model's confidence is well-calibrated
    (rejecting uncertain samples improves accuracy more).
    """
    r_rate, r_accuracy = accuracy_rejection_curve(
        probs, true_labels, predicted_labels, n_steps
    )
    return auc(r_rate, r_accuracy)


def posterior_variance(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    n_classes: int = 2,
    n_samples: int = 10,
    seed: int | None = None,
) -> float:
    """Estimate posterior variance via Monte Carlo sampling.

    For each class, draws random subsets of predictions and computes how
    much the estimated class probability varies across draws.

    Args:
        predictions: Probability array of shape ``(n_samples, n_classes)``.
        true_labels: Ground-truth labels.
        n_classes: Number of output classes.
        n_samples: Number of Monte Carlo draws per class.
        seed: Optional random seed for reproducibility.

    Returns:
        Average posterior variance across classes.
    """
    rng = np.random.default_rng(seed)
    pv = 0.0
    for cls in range(n_classes):
        preds = predictions[true_labels == cls]
        ps = []
        for _ in range(n_samples):
            inds = rng.choice(len(preds), size=n_samples)
            picked = preds[inds, :]
            p = (picked[:, cls] > picked[:, (cls + 1) % n_classes]).sum()
            p /= float(n_samples)
            ps.append(p)
        p_av = (preds[:, cls] > preds[:, (cls + 1) % n_classes]).mean()
        pv += ((np.array(ps) - p_av) ** 2).sum()
    pv /= n_classes
    return float(pv)


def posterior_expectation(
    predictions: np.ndarray,
    n_classes: int = 2,
    n_draws: int = 10,
    draw_size: int = 100,
    seed: int | None = None,
) -> float:
    """Estimate posterior expectation (expected uncertainty).

    Draws random subsets of predictions and estimates how far the
    maximum class probability is from 1.0 on average.

    Args:
        predictions: Probability array of shape ``(n_samples, n_classes)``.
        n_classes: Number of output classes.
        n_draws: Number of Monte Carlo draws.
        draw_size: Number of samples per draw.
        seed: Optional random seed for reproducibility.

    Returns:
        Average posterior expectation (lower = more confident).
    """
    rng = np.random.default_rng(seed)
    pe = 0.0
    for _ in range(n_draws):
        class_means = []
        for cls in range(n_classes):
            inds = rng.choice(len(predictions), size=draw_size)
            picked = predictions[inds, :]
            class_means.append(picked[:, cls].sum() / draw_size)
        pe += 1 - max(class_means)
    pe /= float(n_draws)
    return float(pe)

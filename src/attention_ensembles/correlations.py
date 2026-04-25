"""
Model correlation and diversity measures.

Provides prediction-based correlation and correlation-based posterior
variance (the QP objective).  RTD (topological) correlation requires
the ``rtd`` package and is handled separately via :func:`rtd_correlation_matrix`.
"""

import numpy as np


def prediction_correlation(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> float:
    """Compute cross-correlation between two models' prediction vectors.

    Uses the first class logits (column 0) as the signal for correlation.

    Args:
        pred_a: Predictions from model A, shape ``(n_samples, n_classes)``.
        pred_b: Predictions from model B, shape ``(n_samples, n_classes)``.

    Returns:
        Scalar cross-correlation value (unnormalised).
    """
    return float(np.correlate(pred_a[:, 0], pred_b[:, 0])[0])


def prediction_correlation_matrix(
    predictions: np.ndarray,
) -> np.ndarray:
    """Build a full correlation matrix from an array of model predictions.

    Args:
        predictions: Shape ``(n_models, n_samples, n_classes)``.

    Returns:
        Symmetric matrix of shape ``(n_models, n_models)``.
    """
    n_models = predictions.shape[0]
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i, n_models):
            c = prediction_correlation(predictions[i], predictions[j])
            matrix[i, j] = c
            matrix[j, i] = c
    return matrix


def rtd_correlation_matrix(
    attention_weights: list[np.ndarray],
) -> np.ndarray:
    """Build a correlation matrix using Representation Topological Divergence.

    Requires the ``rtd`` package (install from
    https://github.com/IlyaTrofimov/RTD.git).

    Args:
        attention_weights: List of numpy arrays, one per model — typically
            extracted from the last transformer layer's attention output
            projection (``model.distilbert.transformer.layer[-1]
            .attention.out_lin.weight``).

    Returns:
        Symmetric dissimilarity matrix of shape ``(n_models, n_models)``,
        normalised to [0, 1] where 0 = identical representations.

    Raises:
        ImportError: If ``rtd`` is not installed.
    """
    try:
        import rtd as rtd_lib
    except ImportError:
        raise ImportError(
            "The 'rtd' package is required for topological correlation. "
            "Install with: pip install git+https://github.com/IlyaTrofimov/RTD.git"
        )

    n_models = len(attention_weights)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i, n_models):
            d = rtd_lib.rtd(attention_weights[i], attention_weights[j])
            matrix[i, j] = d
            matrix[j, i] = d

    # Normalise and invert: high RTD → low correlation
    if matrix.max() > 0:
        matrix = matrix / matrix.max()
    matrix = 1.0 - matrix
    return matrix


def extract_submatrix(
    full_matrix: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Extract a submatrix for a subset of models.

    Args:
        full_matrix: Shape ``(n_models, n_models)`` correlation matrix.
        indices: List of model indices to include.

    Returns:
        Submatrix of shape ``(len(indices), len(indices))``.
    """
    return full_matrix[np.ix_(indices, indices)]


def correlation_based_variance(
    corr_matrix: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the correlation-based posterior variance for given weights.

    This is the QP objective: ``w^T C w`` where C is the correlation matrix.
    Lower values indicate a more diverse (less correlated) ensemble.

    Args:
        corr_matrix: Shape ``(n_models, n_models)`` correlation matrix.
        weights: Shape ``(n_models,)`` ensemble weights.

    Returns:
        Scalar posterior variance.
    """
    w = np.asarray(weights).reshape(1, -1)
    return float((w @ corr_matrix @ w.T).item())

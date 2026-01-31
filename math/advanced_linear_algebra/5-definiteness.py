#!/usr/bin/env python3
"""Module to calculate the definiteness of a square matrix."""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a square NumPy matrix."""

    # Check type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check for empty or non-square matrix
    if (matrix.size == 0
            or len(matrix.shape) != 2
            or matrix.shape[0] != matrix.shape[1]):
        return None

    # Check for symmetry
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Determine definiteness
    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0) and not pos
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0) and not neg

    if pos:
        return "Positive definite"
    elif pos_semi:
        return "Positive semi-definite"
    elif neg:
        return "Negative definite"
    elif neg_semi:
        return "Negative semi-definite"
    elif np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"
    else:
        return None

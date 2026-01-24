#!/usr/bin/env python3
"""Module that performs matrix multiplication using numpy."""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiply two numpy arrays using matrix multiplication.

    Args:
        mat1 (numpy.ndarray): First 2D array.
        mat2 (numpy.ndarray): Second 2D array.

    Returns:
        numpy.ndarray: Product of mat1 and mat2.
        None: If matrices cannot be multiplied.
    """
    if mat1.shape[1] != mat2.shape[0]:
        return None
    return np.matmul(mat1, mat2)

#!/usr/bin/env python3
"""Module that performs element-wise arithmetic on numpy arrays."""


def np_elementwise(mat1, mat2):
    """
    Return element-wise sum, difference, product, and quotient of two arrays.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray or scalar): Second array.

    Returns:
        tuple: (sum, difference, product, quotient) arrays.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2

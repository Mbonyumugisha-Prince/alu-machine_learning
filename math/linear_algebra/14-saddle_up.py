#!/usr/bin/env python3
"""Module that performs matrix multiplication using numpy
without conditionals.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiply two numpy arrays using matrix multiplication.

    Returns None if shapes are incompatible.
    """
    try:
        return np.matmul(mat1, mat2)
    except ValueError:
        return None

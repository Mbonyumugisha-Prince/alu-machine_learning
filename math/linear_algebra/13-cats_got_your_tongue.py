#!/usr/bin/env python3
"""Module to concatenate two numpy arrays along a given axis."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along the specified axis.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray): Second array.
        axis (int, optional): Axis along which to concatenate. Defaults to 0.

    Returns:
        numpy.ndarray: New concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)

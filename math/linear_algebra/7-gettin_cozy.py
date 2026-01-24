#!/usr/bin/env python3
"""
This module contains a function to concatenate two 2D matrices along a given
axis.
"""

import copy


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along the specified axis.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.
        axis (int, optional): Axis along which to concatenate.
            0=row, 1=column. Defaults to 0.

    Returns:
        list of list: New concatenated matrix.
        None: If matrices cannot be concatenated along the given axis.
    """
    new_matrix = copy.deepcopy(mat1)

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_matrix.extend(copy.deepcopy(mat2))
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(new_matrix)):
            new_matrix[i].extend(mat2[i])
    else:
        return None

    return new_matrix

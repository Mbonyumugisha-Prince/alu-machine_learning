#!/usr/bin/env python3
"""
This module contains a function to concatenate two 2D matrices along a given
axis.
"""


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
    # Make a deep copy of mat1 manually
    new_matrix = [row[:] for row in mat1]

    if axis == 0:
        # Concatenate along rows: number of columns must match
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat2:
            new_matrix.append(row[:])
    elif axis == 1:
        # Concatenate along columns: number of rows must match
        if len(mat1) != len(mat2):
            return None
        for i in range(len(new_matrix)):
            new_matrix[i].extend(mat2[i])
    else:
        return None

    return new_matrix

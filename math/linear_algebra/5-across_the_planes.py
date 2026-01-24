#!/usr/bin/env python3
"""
This module contains a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.

    Returns:
        list of list: Element-wise sum of mat1 and mat2.
        None: If matrices are not the same shape.
    """
    if len(mat1) != len(mat2):
        return None
    new_matrix = []
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
        new_matrix.append([a + b for a, b in zip(row1, row2)])
    return new_matrix

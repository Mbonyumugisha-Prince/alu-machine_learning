#!/usr/bin/env python3
"""
This module contains a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of list): A 2D matrix.

    Returns:
        list of list: The transposed matrix.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = []
    for c in range(cols):
        transposed.append([matrix[r][c] for r in range(rows)])
    return transposed

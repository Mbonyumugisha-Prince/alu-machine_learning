#!/usr/bin/env python3
"""
This module contains a function to calculate the shape of a nested matrix.
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a nested matrix.

    Args:
        matrix (list): A list (possibly nested) representing a matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = [len(matrix)]
    if isinstance(matrix[0], list):
        shape.extend(matrix_shape(matrix[0]))
    return shape

#!/usr/bin/env python3
"""
This module contains a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.

    Returns:
        list of list: Result of matrix multiplication.
        None: If matrices cannot be multiplied.
    """
    # Number of columns in mat1 must equal number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Create result matrix
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat2)):
                s += mat1[i][k] * mat2[k][j]
            row_result.append(s)
        result.append(row_result)

    return result

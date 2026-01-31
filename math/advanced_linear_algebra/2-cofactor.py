#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""


def determinant(matrix):
    """Helper function to calculate determinant"""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub_matrix = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)

    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""

    # Type check
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    # Empty or not square
    if matrix == []:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 matrix
    if n == 1:
        return [[1]]

    # Compute cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [
                row[:j] + row[j + 1:]
                for idx, row in enumerate(matrix)
                if idx != i
            ]
            minor_det = determinant(sub_matrix)
            cofactor_value = ((-1) ** (i + j)) * minor_det
            row_cofactors.append(cofactor_value)
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix

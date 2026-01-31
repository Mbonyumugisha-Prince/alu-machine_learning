#!/usr/bin/env python3
"""Module that calculates the inverse of a square matrix."""


def inverse(matrix):
    """Calculates the inverse of a square matrix."""

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is non-empty and square
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Handle 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    # Helper function: calculate determinant
    def determinant(m):
        if len(m) == 1:
            return m[0][0]
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        det = 0
        for c in range(len(m)):
            sub = [[m[i][j] for j in range(len(m)) if j != c]
                   for i in range(1, len(m))]
            det += ((-1) ** c) * m[0][c] * determinant(sub)
        return det

    det = determinant(matrix)
    if det == 0:
        return None

    # Helper function: get cofactor matrix
    def cofactor(m):
        cof = []
        for r in range(len(m)):
            row = []
            for c in range(len(m)):
                sub = [[m[i][j] for j in range(len(m)) if j != c]
                       for i in range(len(m)) if i != r]
                row.append(((-1) ** (r + c)) * determinant(sub))
            cof.append(row)
        return cof

    # Transpose a matrix
    def transpose(m):
        return [[m[j][i] for j in range(len(m))]
                for i in range(len(m))]

    cof = cofactor(matrix)
    adj = transpose(cof)

    # Multiply adjugate by 1/det to get inverse
    inv = [[adj[i][j] / det for j in range(n)]
           for i in range(n)]
    return inv

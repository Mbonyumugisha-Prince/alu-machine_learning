#!/usr/bin/env python3
"""
This module contains a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Args:
        arr1 (list of int/float): First array.
        arr2 (list of int/float): Second array.

    Returns:
        list: Element-wise sum of arr1 and arr2.
        None: If arrays are not the same length.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]

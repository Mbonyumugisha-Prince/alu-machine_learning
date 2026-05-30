#!/usr/bin/env python3
"""Module for one-hot encoding."""

import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix."""
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes < 1 or classes <= np.max(Y):
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None

#!/usr/bin/env python3
"""Module for creating the training operation."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the training operation using gradient descent."""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op

#!/usr/bin/env python3
"""Module for creating TensorFlow placeholders."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Return two placeholders x and y for the neural network.

    Args:
        nx: number of feature columns in the input data
        classes: number of classes in the classifier

    Returns:
        x: placeholder for the input data
        y: placeholder for the one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y

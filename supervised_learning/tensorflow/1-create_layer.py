#!/usr/bin/env python3
"""Module for creating a neural network layer."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a layer with He et al. initialization."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.dense(
        prev, n, activation=activation,
        kernel_initializer=initializer, name='layer'
    )
    return layer

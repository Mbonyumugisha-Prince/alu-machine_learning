#!/usr/bin/env python3
"""Module for creating a neural network layer."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a dense layer with He et al. initialization.

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function the layer should use

    Returns:
        tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.dense(
        prev, n, activation=None,
        kernel_initializer=initializer, name='layer'
    )
    if activation is not None:
        act_name = activation.__name__.capitalize()
        layer = activation(layer, name='layer/' + act_name)
    return layer

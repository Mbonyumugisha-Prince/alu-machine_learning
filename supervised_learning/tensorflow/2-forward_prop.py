#!/usr/bin/env python3
"""Module for forward propagation graph."""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create the forward propagation graph for the neural network."""
    output = x
    for n, activation in zip(layer_sizes, activations):
        output = create_layer(output, n, activation)
    return output

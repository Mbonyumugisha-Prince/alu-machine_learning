#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in the encoder
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Reparameterization trick."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mean)[0], latent_dims)
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    dec_inputs = keras.Input(shape=(latent_dims,))
    x = dec_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    dec_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(dec_inputs, dec_outputs)

    z_out, z_mean_out, z_log_var_out = encoder(inputs)
    outputs = decoder(z_out)
    auto = keras.Model(inputs, outputs)

    reconstruction_loss = keras.backend.sum(
        keras.backend.binary_crossentropy(inputs, outputs), axis=-1
    )
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var_out
        - keras.backend.square(z_mean_out)
        - keras.backend.exp(z_log_var_out),
        axis=-1
    )
    auto.add_loss(keras.backend.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto

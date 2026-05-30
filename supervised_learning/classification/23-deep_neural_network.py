#!/usr/bin/env python3
"""Module defining a deep neural network."""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize DeepNeuralNetwork."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for idx in range(1, self.__L + 1):
            if not isinstance(layers[idx - 1], int) or layers[idx - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights['W' + str(idx)] = (
                np.random.randn(layers[idx - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.__weights['b' + str(idx)] = np.zeros((layers[idx - 1], 1))
            prev = layers[idx - 1]

    @property
    def L(self):
        """Get number of layers."""
        return self.__L

    @property
    def cache(self):
        """Get cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Get weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate forward propagation of the deep neural network."""
        self.__cache['A0'] = X
        for idx in range(1, self.__L + 1):
            W = self.__weights['W' + str(idx)]
            b = self.__weights['b' + str(idx)]
            A_prev = self.__cache['A' + str(idx - 1)]
            Z = np.dot(W, A_prev) + b
            self.__cache['A' + str(idx)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the cost using logistic regression."""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the deep neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent on the neural network."""
        m = Y.shape[1]
        L = self.__L
        dZ = cache['A' + str(L)] - Y
        for idx in range(L, 0, -1):
            A_prev = cache['A' + str(idx - 1)]
            W = self.__weights['W' + str(idx)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if idx > 1:
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights['W' + str(idx)] -= alpha * dW
            self.__weights['b' + str(idx)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        A, cache = self.forward_prop(X)
        if verbose:
            print("Cost after 0 iterations: {}".format(self.cost(Y, A)))
        if graph:
            costs.append(self.cost(Y, A))
            iters.append(0)

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                A, _ = self.forward_prop(X)
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    costs.append(c)
                    iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

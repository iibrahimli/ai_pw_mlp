# -*- coding: utf-8 -*-

import numpy as np
import functions as func


class network:
    """
    A simple multilayer perceptron implementation

    Attributes:
        n_layers (int): Number of layers

        n_neurons (list of int): Number of neurons in layers.
            len(n_neurons) = n_layers

        weights (np.ndarray): Weight matrices of the perceptron.
            Bias vectors are also included in the matrices.
            shape: (n_layers-1, n_input, n_output)

        interm (np.ndarray): Cache of intermediate activation
            values for backpropagation.
            shape: (n_layers, n_neurons[.])
        
        hidden_activation (mlp.activation): Activation function to use
            in hidden layers
            
        output_activation (mlp.activation): Activation function to use
            in output layer
        
        cost (mlp.cost): Cost function to optimize

        dtype: Data type to be used in the network

    """

    def __init__(self, layers: list, hidden_activation=func., dtype=np.float32):
        """
        Initialize a multilayer perceptron.

        Args:
            layers (list of int): List of number of neurons per layer
                ex: [4, 16, 1] - 4 input, 16 hidden, and 1 output
            
            hidden_activation (mlp.activation): Activation function to use
                in hidden layers
                default: mlp.tanh
            
            output_activation (mlp.activation): Activation function to use
                in output layer
                default: mlp.softmax
            
            cost (mlp.cost): Cost function to optimize
                default: mlp.categorical_crossentropy

            dtype: Data type to be used in the network
                default: np.float32

        """

        self.n_layers = len(layers)
        self.n_neurons = layers
        self.dtype = dtype

        # Xavier initialization: Gaussian with mean=0 and std=1/sqrt(n_input_neurons)
        self.weights = np.empty((n_layers, ), dtype=dtype)
        for i in range(n_layers):
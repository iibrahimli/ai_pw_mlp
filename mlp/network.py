# -*- coding: utf-8 -*-

import numpy as np
from .functions import *


class network:
    """
    A simple multilayer perceptron implementation

    Attributes:
        n_layers (int): Number of layers

        n_neurons (list of int): Number of neurons in layers.
            len(n_neurons) = n_layers

        weights (list of np.ndarray): Weight matrices of the perceptron.
            Bias vectors are also included in the matrices.
            shape: (n_input_neurons, n_output_neurons)

        interm (list of np.ndarray): Cache of intermediate activation
            values for backpropagation.
            shape: (n_neurons)
        
        hidden_activation (activation): Activation function to use
            in hidden layers
            
        output_activation (activation): Activation function to use
            in output layer
        
        cost (cost): Cost function to optimize

    """

    def __init__(self,
                 layers: list,
                 hidden_activation: activation = tanh,
                 output_activation: activation = softmax,
                 cost: cost = categorical_crossentropy):
        """
        Initialize a multilayer perceptron.

        Args:
            layers (list of int): List of number of neurons per layer
                ex: [4, 16, 1] - 4 input, 16 hidden, and 1 output
            
            hidden_activation (activation): Activation function to use
                in hidden layers
                default: tanh
            
            output_activation (activation): Activation function to use
                in output layer
                default: softmax
            
            cost (mlp.cost): Cost function to optimize
                default: categorical_crossentropy

        """

        self.n_layers = len(layers)
        self.n_neurons = layers
        self.dtype = dtype

        # Xavier initialization: Gaussian with mean=0 and std=1/sqrt(n_input_neurons)
        self.weights = []
        for i in range(n_layers-1):
            w_mat = np.random.normal(loc=0.0,
                                     scale=1.0 / np.sqrt(n_neurons[i]),
                                     size=(n_neurons[i], n_neurons[i+1]))
            self.weigths.append(w_mat)

            np.random.norm
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

        w (dict of np.ndarray): Weight matrices of the perceptron.
            shape: n_layers - 1, (n_input_neurons, n_output_neurons)
        
        b (dict of np.ndarray): Biases of the perceptron.
            shape: n_layers - 1, (n_neurons)

        z (dict of np.ndarray): Cache of intermediate z values
            for backpropagation. (input layer excluded)
            shape: n_layers - 1, (n_neurons)
        
        a (dict of np.ndarray): Cache of a values (layer activations)
            for backpropagation.
            shape: n_layers, (n_neurons)
        
        activations (list of activation): Activation function to use
            in layers (input layer excluded)
            len: n_layers - 1
        
        loss (cost): Loss function

    """

    def __init__(self, layers, activation_funcs, loss):
        """
        Initialize a multilayer perceptron.

        Args:
            layers (list of int): List of number of neurons per layer
                ex: [4, 16, 1] - 4 input, 16 hidden, and 1 output
            
            activation_funcs (list of activation): Activation functions of each layer
                Length must be 2 (hidden, output) or n_layers - 1
            
            loss (cost): Loss function to optimize
                default: categorical_crossentropy

        """

        self.n_layers = len(layers)
        self.n_neurons = layers

        # weights and biases
        self.w = {}
        self.b = {}
        for i in range(1, self.n_layers):
            # Xavier initialization: Gaussian with mean=0 and std=2/sqrt(n_input_neurons)
            self.w[i] = np.random.normal(loc=0.0,
                                         scale=1.0 / np.sqrt(self.n_neurons[i]),
                                         size=(self.n_neurons[i-1], self.n_neurons[i]))
            self.b[i] = np.zeros(shape=(self.n_neurons[i]))
        
        # intermediate values
        self.z = {}
        
        # activation values
        self.a = {}

        # functions used
        self.activations = {}
        if len(activation_funcs) == 2:
            for i in range(1, self.n_layers - 1):
                self.activations[i] = activation_funcs[0]
            self.activations[self.n_layers - 1] = activation_funcs[1]
        elif len(activation_funcs) == self.n_layers - 1:
            for i in range(1, self.n_layers):
                self.activations[i] = activation_funcs[i - 1]
        else:
            raise ValueError("Invalid number of activation functions")

        self.loss = loss


    def _forward(self, x):
        """
        Forward pass computes the z values and intermediate activation values
        a = f(z) = f(XW + b)

        Args:
            x (np.ndarray): Batch of samples. Each sample is a row vector, the shape of x
                is expected to be (batch_size, n_features). Batch size can be 1 in which
                case x will contain a single sample

        """

        self.a[0] = x
        for i in range(1, self.n_layers):
            self.z[i] = np.dot(self.a[i-1], self.w[i]) + self.b[i]
            self.a[i] = self.activations[i].forward(self.z[i])
        


    def predict(self, x):
        """
        Predict y value

        Args:
            x (np.ndarray): Input batch. Shape: (batch_size, n_neurons[0])

        Returns:
            y_pred (np.ndarray): Prediction for the given batch. Shape: (batch_size, n_neurons[-1])
        """

        self._forward(x)
        return self.a[self.n_layers - 1]


    def _update_params(self, index, dw, delta, lr):
        """
        Apply the gradient to parameters

        Arguments:
            index (int): Index of layer to update
            dw (np.ndarray): Gradient of cost function wrt parameters
            delta (np.ndarray): Backpropagating error
            lr (float): Learning rate

        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, axis=0)


    def _backward(self, y_true):
        """
        Backpropagation computes the gradient of cost function with respect to parameters using
        the cached values and updates the parameters

        """

        # determine partial derivative and delta for the output layer
        delta = self.loss.backward(y_true, self.a[self.n_layers - 1]) \
              * self.activations[self.n_layers - 1].backward()
        dw = np.dot(self.a[self.n_layers - 1].T, delta)


    def fit(self, x, y, lr, n_epochs, batch_size, val_ratio=None, shuffle=True, verbose=True):
        """
        Train the multilayer perceptron on the training set

        Args:
            x, y (np.ndarray): Training data

            lr (float): Learning rate

            n_epochs (int): Number of epochs

            batch_size (int): Batch size

            val_ratio (float): If provided, fraction of data is used for validation
                default: None

            shuffle (bool): Whether to shuffle the data
                default: True
            
            verbose (bool): Whether to output stuff during training
                default: True

        Returns:
            history (dict of list): value of loss over epochs.
                Keys: 'train_loss' [, 'val_loss']

        """

        if not x.shape[0] == y.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        history = []

        for e in range(n_epochs):
            # shuffle the data
            seed = np.arange(x.shape[0])
            if shuffle:
                np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = min((j + 1) * batch_size, x.shape[0])
                self._forward(x_[k:l])
                self._backward(self.z, self.a, y_[k:l])
            
            if verbose and e % 10 == 0:
                # TODO: if validation validate, print
                pass

        return history
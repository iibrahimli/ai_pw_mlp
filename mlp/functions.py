# -*- coding: utf-8 -*-

"""
Activation, loss and cost functions.

Todo:
    * tanh
    * sigmoid
    * relu
    * leaky_relu
    * softmax
    * categorical_crossentropy

"""

import numpy as np


class activation:
    """
    Abstract class of a differentiable function that
    contains a forward and a backward pass (for backprop)

    """

    def forward(self, z):
        """
        Calculates the output
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def backward(self, z):
        """
        Calculates the gradient wrt z
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward() not implemented")


class cost:
    """
    Abstract class of a differentiable cost that
    contains a forward and a backward pass (for backprop)

    """

    def forward(self, y_true, y_pred):
        """
        Calculates the cost (scalar)
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def backward(self, y_true, y_pred):
        """
        Calculates the gradient of cost wrt inputs
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward() not implemented")


# activation

class tanh(activation):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        return 1 - np.tanh(z)**2


class sigmoid(activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class softmax(activation):
    def forward(self, z):
        pass

    def backward(self, z):
        pass


# cost

class mean_squared_error(cost):
    def forward(self, y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_true, y_pred):
        return y_pred - y_true


class categorical_crossentropy(cost):
    def forward(self, y_true, y_pred):
        pass

    def backward(self, y_true, y_pred):
        pass


class binary_crossentropy(cost):
    def forward(self, y_true, y_pred):
        pass

    def backward(self, y_true, y_pred):
        pass
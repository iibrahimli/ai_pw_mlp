# -*- coding: utf-8 -*-

"""
Activation, loss and other utility functions.

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

    def forward(self, x):
        """
        Calculates the output
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")


    def backward(self, y, x_cache):
        """
        Calculates the gradient wrt inputs
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
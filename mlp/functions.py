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
    contains a forward and a backward pass

    """

    def forward(self, x):
        """
        Forward pass calculates the output
        """
        raise NotImplementedError(f"{cls}forward() not implemented")


    def backward(self, y):
        """
        Backward pass calculates the gradient wrt inputs
        """
        raise NotImplementedError(f"backward() not implemented")
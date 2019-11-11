# -*- coding: utf-8 -*-

"""
Activation, loss and cost functions.

"""

import numpy as np

_epsilon = 1e-10


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

    def backward(self, z, a):
        """
        Calculates the gradient wrt z, (a: cached activation value)
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

class relu(activation):
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z, a):
        dz = np.ones_like(z)
        dz[dz < 0] = 0
        return dz


class leaky_relu(activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        y1 = ((z > 0) * z)
        y2 = ((z <= 0) * z * alpha)
        return y1 + y2

    def backward(self, z, a):
        dz = np.ones_like(z)
        dz[dz < 0] = alpha
        return dz


class tanh(activation):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a):
        return 1 - a**2


class sigmoid(activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z, a):
        return a * (1 - a)


class softmax(activation):
    def forward(self, z):
        z = z - z.max(axis=1, keepdims=True)
        exps = np.exp(z)
        return exps / exps.sum(axis=1, keepdims=True)

    def backward(self, z, a):
        self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, np.newaxis])


# cost

class mean_squared_error(cost):
    def forward(self, y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_true, y_pred):
        return y_pred - y_true


class categorical_crossentropy(cost):
    def forward(self, y_true, y_pred):
        # efficient, but assumes y is one-hot
        return -np.log(y_pred[np.where(y_true)])

    def backward(self, y_true, y_pred):
        return -y_true / y_pred


class binary_crossentropy(cost):
    pass
    # def forward(self, y_true, y_pred):
    #     pass

    # def backward(self, y_true, y_pred):
        # pass
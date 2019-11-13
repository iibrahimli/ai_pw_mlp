"""
Function for data preprocessing
"""

import numpy as np
import pandas as pd


def feat_normalize(x):
    """
    Column-wise min-max normalize an array of features to range [0, 1]

    Args:
        x (np.ndarray): Feature array to normalize
    
    Returns:
        res (np.ndarray): result
        mins (np.ndarray): feature minimums
        maxs (np.ndarray): feature maximums
    """

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    res = (x - mins) / (maxs - mins)
    return (res, mins, maxs)


def feat_standardize(x):
    """
    Column-wise standardize an array of features to have
    mean=0 and std=1

    Args:
        x (np.ndarray): Feature array to standardize
    
    Returns:
        res (np.ndarray): result
        means (np.ndarray): feature means
        stds (np.ndarray): feature standard deviations
    """

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    res = (x - means) / stds
    return (res, means, stds)


def int_to_one_hot(x):
    """
    One-hot encode integer labels

    Args:
        x (np.ndarray): Array of integer labels
        Shape: (n_samples,)
    
    Returns:
        oh (np.ndarray): Array of one hot encoded labels
        Shape: (n_samples, n_classes)
    """

    oh = np.zeros((x.size, x.max()+1))
    oh[np.arange(x.size), x] = 1
    return oh


def shuffle(x, y):
    """
    Randomly shuffle data
    """

    seed = np.arange(x.shape[0])
    np.random.shuffle(seed)
    return x[seed], y[seed]


def split(x, y, ratio=0.7):
    """
    Split data into training and testing sets
    """

    idx = round(len(x) * ratio)
    x_train, y_train = x[:idx], y[:idx]
    x_test,  y_test  = x[idx:], y[idx:]
    return (x_train, y_train), (x_test, y_test)
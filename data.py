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

    pass


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

    pass
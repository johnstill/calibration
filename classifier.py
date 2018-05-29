#! /usr/bin/env python -B
"""
Produces classifications of MNIST so we have something to develop calibration
tools against.
"""

import numpy as np
from sklearn.datasets import fetch_mldata


def get_mnist():
    """Load, split, and shuffle the data.

    Returns:
        A 4-tuple of (X_train, X_test, y_train, y_test)
    """
    mnist = fetch_mldata('MNIST original', data_home='.')
    X, y = mnist['data'], mnist['target']
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]
    np.random.seed(42)
    indices = np.random.permutation(60_000)
    X_train = X_train[indices]
    y_train = y_train[indices]
    return X_train, X_test, y_train, y_test

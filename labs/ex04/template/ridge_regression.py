# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    w = np.linalg.inv(np.matmul(np.transpose(tx), tx)+ lambda_*np.identity(np.shape(tx)[1]))
    w = np.matmul(w,np.transpose(tx))
    w = np.matmul(w,y)
    return w
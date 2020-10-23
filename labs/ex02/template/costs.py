# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = (y-(tx*w[0] + w[1]))**2
    return np.mean(e)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    raise NotImplementedError
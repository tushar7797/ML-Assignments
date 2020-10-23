# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    #Calculate the loss. You can calculate the loss using mse or mae
    e = (y-tx.dot(w))**2
    #print(y.shape, tx.dot(w))
    return np.mean(e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    # ***************************************************
    e = (y-tx.dot(w))
    return np.transpose(tx).dot(e)/len(y)
    raise NotImplementedError

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    a = np.zeros((len(x),degree))
    for i in range(0,len(x)):
        for j in range(0,degree):
            a[i][j] = x[i]**j
            
    return a


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.inv(np.matmul(np.transpose(tx), tx))
    w = np.matmul(w,np.transpose(tx))
    w = np.matmul(w,y)
    return w



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


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = (y-tx.dot(w))
    return np.transpose(tx).dot(e)/len(y)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        for minibatch_y, minibatch_tx in batch_iter(y, tx, len(y)):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            grad = compute_gradient(minibatch_y, minibatch_tx,w) 
            w = w + gamma*compute_gradient(y,tx,w)
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    #raise NotImplementedError
    return losses, ws
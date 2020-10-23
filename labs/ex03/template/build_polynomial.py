# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    a = np.zeros((np.shape(x)[0],degree))
    for i in range(0,np.shape(x)[0]):
        for j in range(0,degree):
            a[i][j] = x[i]**j
            
    return a
    


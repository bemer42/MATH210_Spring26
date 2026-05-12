#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 01:45:49 2024

@author: brooksemerick
"""

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

# Objective: This script outputs various matrices built using specific built-in 
# Python/NumPy commands. These commands are used to build important matrices 
# quickly and efficiently. This file solves Problem 2 on Homework Assignment 2. 

# The 'ones' command creates a matrix of ones. 
# Multiply this matrix of ones by any constant to create a matrix with that 
# constant as every element.
A = 3 * np.ones((30, 20))

# The 'reshape' command assembles a vector into a matrix of specified 
# dimensions.
B = np.reshape(np.arange(1, 101), (10, 10)).T
    
# The 'diag' command creates a matrix with a vector on the diagonal.
C = np.diag(-2 * np.ones(16), 0) + np.diag(np.ones(15), 1) + \
    np.diag(np.ones(15), -1) + np.diag([1], 15) + np.diag([1], -15)

# Plot the sparsity of the matrix C: 
plt.figure(1)
plt.spy(C)

# The 'toeplitz' command creates a Toeplitz matrix.
D = LA.toeplitz(np.hstack([[-2, 1], np.zeros(13), [1]]))

# The 'triu' and 'tril' commands make the lower or upper triangular part of a 
# matrix zero.
E = np.triu(LA.toeplitz(np.arange(1, 9)))

# Plot the sparsity of the matrix E: 
plt.figure(2)
plt.spy(E)

# Toeplitz matrix again using a reciprocal vector.
F = LA.toeplitz(1.0 / np.arange(1, 9))
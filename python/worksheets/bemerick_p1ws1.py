#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:15:26 2026

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1.

# Discretize the domain with a vector, x:
N = int(1e3)
x = np.linspace(-1, 2, N)  

# Create a function
f = lambda x:  3 * x ** 4 + 8 * x ** 3 - 18 * x ** 2

# Create the derivative function
# fp = lambda x: 12 * x ** 3 + 24 * x ** 2 - 36 * x
h = 1e-5
fp = lambda x: (f(x-h)-f(x+h)) / 2 / h

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp, bracket=(-1/2, 1/2), method="brentq")
c_2 = optimize.root_scalar(fp, bracket=(1/2, 3/2), method="brentq")
c = np.array([c_1.root, c_2.root])

# Plot the function's graph
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('P1, Worksheet 1, Plot 1', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(f(x)), np.max(f(x))])
plt.legend()

# Save the file in high quality format: 
plt.savefig('P1, Worksheet 1, Plot 1.eps', format='eps')

# Show the plot:
plt.show()

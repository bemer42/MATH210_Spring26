#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:54:42 2026

@author: brooksemerick
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#discretize domain
N = 1000
x = np.linspace(-1, 6, N)  

#define the functions
def f(x):
    return x**3 - 7 * x**2 + 10 * x

#calculate min and max
MIN = optimize.minimize_scalar(f, bounds=(-1,6), method= 'bounded')
MAX = optimize.minimize_scalar(lambda x: -f(x), bounds=(6-.1,6), method='bounded')



# Plot the graph of the function
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
# plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('P1WS1 Polynomial with Critical Points', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(f(x))-1, np.max(f(x))+1])
plt.legend()
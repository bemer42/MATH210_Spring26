#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 07:11:45 2026

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 2 problem 1.

L_vec = np.array(1,100,100)
x_max = np.empty(len(L_vec))
A_max = np.empty(len(L_vec))
x_min = np.empty(len(L_vec))
A_min = np.empty(len(L_vec))

# Create the function in part 1a
def f(x,L):
    return x **2 / 16 + (L - x) ** 2 / 4 / np.pi
    
for i in range(1, len(L_vec)):
    
    # Discretize the domain with a vector, x
    # N = int(1e3)
    # x = np.linspace(0, L[i], N)  

    Min = optimize.minimize_scalar(lambda x: f(x, L_vec[i]), bounds=(0,L_vec[i]), method="bounded")
    A_min[i] = Min.fun
    x_min[i] = Min.x
    
    Max = optimize.minimize_scalar(lambda x: -f(x, L_vec[i]), bounds=(0,L_vec[i]), method="bounded")
    A_max[i] = Max.fun
    x_max[i] = Max.x
    




# Create a general numerical derivative using a centered difference
h = 1e-5
fp = lambda x: (f(x+h)-f(x-h)) / 2 / h

# Use built-in solver to calculate critical points
# c_1 = optimize.root_scalar(fp1_num, bracket=(-5-.1, -5+.1), method="brentq")
# c_2 = optimize.root_scalar(fp1_num, bracket=(1-.1, 1+.1), method="brentq")

# Compile both roots into an array
# c = np.array([c_1.root, c_2.root])

# Plot the graph of the function
plt.figure(1)
plt.plot(L_vec, x_min, 'k-', linewidth=3, label='y = f(x)')
# plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('P1WS1 Polynomial with Critical Points', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
# plt.xlim([np.min(x), np.max(x)])
# plt.ylim([np.min(f(x))-1, np.max(f(x))+1])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
# plt.savefig('p1ws2_1_plot.eps', format='eps')
# plt.savefig('p1ws2_1_plot.png', format='png')

"""
Created on Sun Feb  1 22:15:26 2026

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1 problem 1a.  This file uses 
# scipy optimize with root_scalar to find critical points.

# Discretize the domain with a vector, x
N = int(1e3)
x = np.linspace(-1, 2, N)  

# Create the function in part 1a
f = lambda x:  3 * x ** 4 + 8 * x ** 3 - 18 * x ** 2

# Create the derivative function directly
fp = lambda x: 12 * x ** 3 + 24 * x ** 2 - 36 * x

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp, bracket=(-1/2, 1/2), method="brentq")
c_2 = optimize.root_scalar(fp, bracket=(1/2, 3/2), method="brentq")

# Compile both roots into an array
c = np.array([c_1.root, c_2.root])

# Plot the graph of the function
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('P1WS1 Polynomial with Critical Points', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(f(x)), np.max(f(x))])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws1_1a_plot.eps', format='eps')
plt.savefig('p1ws1_1a_plot.png', format='png')


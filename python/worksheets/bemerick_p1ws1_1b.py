"""
Created on Thu Feb  5 22:23:09 2026

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1 problem 1b.

# Discretize the domain with a vector, x
N = int(1e3)
x = np.linspace(-1, 4, N)  

# Create the function in part 1a
f = lambda x: np.cbrt(x) * (x-2) ** 2

# Create a general numerical derivative using a centered difference
h = 1e-5
fp = lambda x: (f(x+h)-f(x-h)) / 2 / h

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp, bracket=(2/7-.1, 2/7+.1), method="brentq")
c_2 = optimize.root_scalar(fp, bracket=(2-.1, 2+.1), method="brentq")

# Compile both roots into an array
c = np.array([c_1.root, c_2.root])

for x_inf in x:
    dfdx = fp(x_inf)
    if np.isfinite(dfdx) and abs(dfdx) > 1e2:
        c = np.append(c,x_inf)

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
# plt.ylim([np.min(f(x)), np.max(f(x))])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws1_1b_plot.eps', format='eps')
plt.savefig('p1ws1_1b_plot.png', format='png')
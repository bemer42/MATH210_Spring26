import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Homework 1, Problem 2.


# PART a
# Discretize the domain with a vector, x
N = int(1e3)
x = np.linspace(-1, 6, N)  

# Define the function y = f(x)
def f(x):
    return x**3 - 7 * x**2 + 10 * x

# Define a function that outputs the tangent line at c 
def TL(c,x):
    m = 3*(c)**2 - 14*(c) + 10
    b = f(c) - m*c
    return m*x+b

# Plot the functions
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = x^3 - 7x^2 + 10x')
plt.plot(x, TL(1,x), 'r-.', linewidth=2, label='Tangent at x = 1')
plt.plot(x, TL(3,x), 'g-.', linewidth=2, label='Tangent at x = 3')
plt.plot(x, TL(5,x), 'b-.', linewidth=2, label='Tangent at x = 5')
plt.plot(1, f(1), 'ro', markersize=5)
plt.plot(3, f(3), 'go', markersize=5)
plt.plot(5, f(5), 'bo', markersize=5)
plt.title('Homework 1, Plot 2', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(f(x)), np.max(f(x))])

# Save the file in high quality format 
plt.savefig('p1hw1_plot2.eps', format='eps')
plt.savefig('p1hw1_plot2.png', format='png')

# Show the plot
plt.show()

# PART b
# Finding the relative minimum and maximum on this interval
f_min = optimize.minimize(f,4)
x_min = f_min.x

f_max = optimize.minimize(lambda x: -f(x),1)
x_max = f_max.x

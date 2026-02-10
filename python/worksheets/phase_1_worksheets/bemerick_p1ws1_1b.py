import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1 problem 1b.  This file 
# uses a numerical finite difference derivative function to locate
# critical points with optimize.root_scalar. 

# Discretize the domain with a vector, x
N = int(1e4)
x = np.linspace(-1,4,N)

# Create the function in part 1a
f = lambda x: np.cbrt(x) * (x-2) ** 2

# Create a general numerical derivative using a centered difference
h = 1e-5
fp = lambda x: (f(x+h)-f(x-h)) / 2 / h

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp, bracket=(2/7-.1, 2/7+.1), method="brentq")
c_2 = optimize.root_scalar(fp, bracket=(2-.1, 2+.1), method="brentq")

# Use optimize.minimize_scalar to locate the origin where the 
# slope is very large (infinite)
c_3 = optimize.minimize_scalar(lambda x: -fp(x),bracket=(-1e-3,1e-3))

# Compile both roots into an array
c = np.array([c_1.root, c_2.root, c_3.x])

# Plot the graph of the function
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('Algebraic Function with Critical Points', fontsize=20)
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
plt.savefig('p1ws1_1b_plot.eps', format='eps')
plt.savefig('p1ws1_1b_plot.png', format='png')
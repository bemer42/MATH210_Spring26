import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 2 problem 1. We use the 
# optimize.minimize_scalar function to determine global
# extrema on the closed interval.

# Discretize the domain with a vector, x 
N = int(1e5)
x = np.linspace(0, 50, N)  

# Create the function in problem 1
def f(x):
    return x ** 2 / 16 + (50 - x) ** 2 / 4 / np.pi
    
# Use built-in solver to calculate minimum and maximum on teh interval
f_min = optimize.minimize_scalar(lambda x: f(x), bounds = (0,50), method = 'bounded')
f_max = optimize.minimize_scalar(lambda x: -f(x), bounds = (0,50), method = 'bounded')

# Plot the graph of the function
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
plt.plot(f_max.x, -f_max.fun, 'ro', markersize = '7', label='global max')
plt.plot(f_min.x, f_min.fun, 'mo', markersize = '7', label='global min')
plt.title('Area Function with Max/Min', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x)-2, np.max(x)+2])
plt.ylim([np.min(f(x))-10, np.max(f(x))+10])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws2_1_plot.eps', format='eps')
plt.savefig('p1ws2_1_plot.png', format='png')

import numpy as np
import matplotlib.pyplot as plt

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Homework 1, Problem 1.

# Discretize the domain with a vector, x
N = int(1e4)
x = np.linspace(-5, 20, N)  

# Create another vector, y, that describes the function of interest
y = x / 30 - np.exp(-x / 6) * np.cos(x)
y_p = 1 / 30 + np.exp(-x / 6) * np.cos(x) / 6 + np.exp(-x/6) * np.sin(x)

# Plot the function
plt.plot(x, y, 'k-', linewidth=3, label='y')
plt.plot(x, y_p, 'r-', linewidth=3, label='dy/dx')
plt.title('Homework 1, Plot 1', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(np.concatenate((y, y_p))), np.max(np.concatenate((y, y_p)))])
plt.legend()

# Save the file in high quality format 
plt.savefig('p1hw1_plot1.eps', format='eps')
plt.savefig('p1hw1_plot1.png', format='png')

# Show the plot:
plt.show()
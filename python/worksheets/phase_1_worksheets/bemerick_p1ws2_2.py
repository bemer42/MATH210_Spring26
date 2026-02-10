import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script wil plot the length function versus the 
# angle theta in Phase 1 - Worksheet 2, Problem 2.   

# Discretize the domain with a vector, x
N     = int(1e3)
theta = np.linspace(1e-5, np.pi/2-1e-5, N)  

# Define the hallway widths as parameters
a = 5
b = 6

# Create the function in problem 2
def L(theta):
    return a / np.sin(theta) + b / np.cos(theta)

# Create the derivative function
def Lp(theta):
    return - a / np.sin(theta) / np.tan(theta) + b / np.cos(theta) * np.tan(theta)

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(Lp, bracket=(1e-5, np.pi/2-1e-5), method="brentq")

# Plot the graph of the function
plt.figure(1)
plt.plot(theta, L(theta), 'k-', linewidth=3, label='y = L(theta)')
plt.plot(c_1.root, L(c_1.root), 'ro', linewidth=3, label='critical point')
plt.title('Pivot Problem', fontsize=20)
plt.xlabel('theta', fontsize=15)
plt.ylabel('y=L(theta)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(theta), np.max(theta)])
plt.ylim([0, 100])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws2_2_plot.eps', format='eps')
plt.savefig('p1ws2_2_plot.png', format='png')

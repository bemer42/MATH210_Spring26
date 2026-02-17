import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs graphical relactionships
# between the variable of interest and the parameters for 
# Phase 1 - Worksheet 3 problem 1.

# Discretize the parameter space
N = int(1e3)
x = np.linspace(1e0,3e1,N)

# Define cost function
def C(x,cs,cb,V):
    return 4*cs*V/x + cb*x**2

# Define a set of parameters
cs = 4
cb = 2
V = 200

# Define the minimum value
x_min = np.cbrt(2 * cs * V / cb)
c_min = (np.cbrt(32)+np.cbrt(4))*np.cbrt(cs**2 * cb * V**2)

# Use Python to find minimum value and check
MIN = optimize.minimize(lambda x: C(x,cs,cb,V),x_min+20)

# Check our answer against python solver
error_x = float(np.abs(MIN.x-x_min))
error_fun = float(np.abs(MIN.fun-c_min))

# Plot the graph of the obj fun with minimum
fig1 = plt.figure(1)
plt.plot(x, C(x,cs,cb,V), 'k-', linewidth=3, label='cost')
plt.plot(x_min, c_min, 'ro', markersize=7, label='minimum cost')
plt.title('Cost (C) versus Base Length (x)', fontsize=20)
plt.xlabel('x (base length)', fontsize=15)
plt.ylabel('C (Cost)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x)-1, np.max(x)+1])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
fig1.savefig('p1ws3_3_plot.eps', format='eps')
fig1.savefig('p1ws3_3_plot.png', format='png')
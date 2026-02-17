import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs graphical relactionships
# between the variable of interest and the parameters for 
# Phase 1 - Worksheet 3 problem 1.

# Discretize the parameter space
N = int(1e3)
L_vec = np.linspace(0,100,N)

# Create empty arrays for storage:
x_max = np.empty(len(L_vec))
A_max = np.empty(len(L_vec))
x_min = np.empty(len(L_vec))
A_min = np.empty(len(L_vec))

# Create the function in problem 1
def f(x,L):
    return x **2 / 16 + (L - x) ** 2 / 4 / np.pi
    
# For loop to store the min and max values
for i in range(1, len(L_vec)):
    
    Min = optimize.minimize_scalar(lambda x: f(x, L_vec[i]), bounds=(0,L_vec[i]), method="bounded")
    A_min[i] = Min.fun
    x_min[i] = Min.x
    
    Max = optimize.minimize_scalar(lambda x: -f(x, L_vec[i]), bounds=(0,L_vec[i]), method="bounded")
    A_max[i] = -Max.fun
    x_max[i] = Max.x
    
# Plot the graph of the cut mark versus wire length
fig1 = plt.figure(1)
plt.plot(L_vec, x_min, 'k-', linewidth=3, label='cut mark for min')
plt.plot(L_vec, x_max, 'm-', linewidth=3, label='cut mark for max')
# plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('Cut Mark versus Length of Wire', fontsize=20)
plt.xlabel('L (length of wire)', fontsize=15)
plt.ylabel('x (cut mark)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(L_vec)-1, np.max(L_vec)+1])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
fig1.savefig('p1ws3_1a_plot.eps', format='eps')
fig1.savefig('p1ws3_1a_plot.png', format='png')

# Plot the graph of the area versus wire length
fig2 = plt.figure(2)
plt.plot(L_vec, A_min, 'k-', linewidth=3, label='min area')
plt.plot(L_vec, A_max, 'm-', linewidth=3, label='max area')
# plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('Optimized Area versus Length of Wire', fontsize=20)
plt.xlabel('L (length of wire)', fontsize=15)
plt.ylabel('Area', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(L_vec)-1, np.max(L_vec)+1])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
fig2.savefig('p1ws3_1b_plot.eps', format='eps')
fig2.savefig('p1ws3_1b_plot.png', format='png')

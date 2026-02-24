import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script solves Phase 1 -- Homework 2, problem 
# number 2. 

# Discretize the parameter space
N = int(1e3)
A = np.linspace(0,100,N)

# Create the function in problem 2
def V(n,A):
    return 4/(4-n)*np.sqrt(A/6)**3

# Create arrays for each plot
n = np.array((0, 1, 2))
c = ['k', 'r', 'b']
l = ['a. n = 0', 'b. n = 1', 'c. n = 2']

# Loop through each part and plot
for i in range(len(n)):
    plt.plot(A,V(n[i],A),c[i],linewidth = 4,label = l[i])
 
# Apply other features to plot   
plt.title('Volume versus Surface Area', fontsize=20)
plt.xlabel('A (surface area)', fontsize=15)
plt.ylabel('$V_{max}$ (volume)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()
plt.xlim([np.min(A), np.max(A)])

# Save the file in high quality format 
plt.savefig('p1hw2_plot1.eps', format='eps')
plt.savefig('p1hw2_plot1.png', format='png')

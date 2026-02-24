import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script solves Phase 1 -- Homework 2, problem 
# number 3. 

# Discretize the parameter space
N = int(1e3)
r = np.linspace(1e-1,10,N)

# Create the function in problem 1
def f(r,c,ct,cb,V):
    return (2*ct-cb/3)*np.pi*r**2 + 2*c*np.pi*r + 2*cb*V/r

# Provide a set of parameter values and solve for minimum
c  = 2.0
ct = 3.0
cb = 2.5
V  = 100.0

# Find the minimial r value given a set of parameter values
MIN = optimize.minimize_scalar(lambda r: f(r,c,ct,cb,V), bracket = (np.min(r), np.max(r)))

r_min = float(MIN.x)
f_min = float(MIN.fun)

# Plot the graph of the obj fun with minimum
fig1 = plt.figure(1)
plt.plot(r, f(r,c,ct,cb,V), 'k-', linewidth=3, label='cost')
plt.plot(r_min, f_min, 'ro', markersize=7, label='minimum cost')
plt.title('Cost (C) versus Radius (r)', fontsize=20)
plt.xlabel('r (radius)', fontsize=15)
plt.ylabel('C (cost)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(r)-1, np.max(r)+1])
plt.legend()





import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import curve_fit

# Objective: This script outputs the solution to Homework 1-2
# problem 3, which fits two different types of functions to 
# data pertaining to a different function. 

# Define the data: 
x = np.linspace(0, 2*np.pi, 200)
y = np.exp(np.sin(x-1))

# Define Vandermonde Matrix for 7th degree polynomial: 
# Define the Vandermonde matrix: 
A = np.vander(x, increasing=True)
A = A[:, :8]
    
# Define the Normal A and Normal y: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system for the coefficient vector c: 
c_poly = la.solve(A_norm, y_norm)

# Define the sinusoidal function to be fit:
def f_sin(x,c0,c1,c2,c3,c4):
    return c0 + c1*np.sin(x) + c2*np.cos(x) + c3*np.cos(2*x) + c4*np.sin(2*x)

# Perform curve fitting
c_sin = curve_fit(f_sin, x, y)
c_sin = c_sin[0]

# Plot the function and the data:
x_plot = np.linspace(x[0], x[-1], int(1e4))
p_plot = np.polyval(np.flip(c_poly), x_plot)

plt.figure(1)
plt.plot(x_plot, p_plot, 'r-', linewidth = 4, label='7th deg poly fit')
plt.plot(x_plot, f_sin(x_plot, c_sin[0],c_sin[1],c_sin[2],c_sin[3],c_sin[4]), \
         'b-', linewidth = 4, label='Sinusoidal fit')
plt.plot(x, y, 'ko', linewidth=6, label='data')
plt.show()

# Add titles and labels
plt.title('Polynomial and Sinusoidal Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format: 
plt.savefig('p2hw1-2_figure_2.eps', format='eps')

# Calculate the residual sum of squares (SS_res)
ss_res_poly = np.sum((y - np.polyval(np.flip(c_poly), x)) ** 2)
ss_res_sin  = np.sum((y - f_sin(x, c_sin[0],c_sin[1],c_sin[2],c_sin[3], \
                                c_sin[4])) ** 2)

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2_poly = 1 - (ss_res_poly / ss_tot)
R2_sin  = 1 - (ss_res_sin / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 for Polynomial Fit: {R2_poly}")
print(f"R^2 for Sinusoidal Fit: {R2_sin}")

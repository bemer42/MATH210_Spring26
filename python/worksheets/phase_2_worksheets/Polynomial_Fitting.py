
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

# Define the data points in vectors x and y: 
x = np.linspace(-4, 5, int(50))
y = (1/100*x**5 + 4*x**2 + 6*x+3) + 10*np.random.normal(0,1,len(x))

# Define degree of best-fit polynomial: 
n = 4; 

# Define the truncated Vandermonde matrix: 
A = np.vander(x, increasing=True)
A = A[:, :(n+1)]

# Define the Normal A and Normal y: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system for the coefficient vector c: 
c = LA.solve(A_norm, y_norm)
# c = A_norm.T@y_norm
# c = LA.lstsq(A,y)[0]

# Define plotting for the optimal polynomial:
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))
p_plot = np.polyval(np.flip(c), x_plot)

# Plot the data and the optimal polynomial: 
plt.figure(1)
plt.plot(x_plot, p_plot, 'r-', linewidth=4, label='Polynomial Fit')
plt.plot(x, y, 'bo', linewidth = 6, label = 'Data')

# Customize the plot:
plt.title('Polynomial Curve Fitting', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format: 
plt.savefig('Polynomial_Curve_Fitting.eps', format='eps')

# Show the plot:
plt.show()

# Output coefficients of polynomial fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res)
ss_res = np.sum((y - np.polyval(np.flip(c), x)) ** 2)

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")

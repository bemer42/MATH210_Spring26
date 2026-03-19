
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lapack

# Define the data points in vectors x and y
N = int(10)
x = np.linspace(-4, 5, N)
y = (1/100*x**5 + 4*x**2 + 6*x+3) + 10*np.random.normal(0,1,len(x))

# Define degree of best-fit polynomial: 
n = 3; 

# Define the truncated Vandermonde matrix for the best-fit portion: 
A = np.vander(x, increasing=True)
A = A[:, :(n+1)]

# Define the constrained matrix
# B = A[[0,N-1],:]
B = np.empty((3,n+1))
for i in range(n+1):
    B[0,i] = x[0]**i
    B[1,i] = x[-1]**i
    B[2,i] = 1

# Define the constrained right hand side
z = np.array([y[0],y[-1],1])

# Solve the linear system for the coefficient vector c
result = lapack.dgglse(A,B,y,z)
c = result[3]

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
plt.savefig('Constrained_Curve_Fitting.eps', format='eps')

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

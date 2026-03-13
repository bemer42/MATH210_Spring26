
import numpy as np
import matplotlib.pyplot as plt

# Define the data points in vectors x and y: 
x = np.array([-2, 5, 3, -10, 0, 6, 18])
y = np.random.normal(7,9, size=len(x))

# Define the Vandermonde matrix: 
A = np.vander(x, increasing=True)

# Solve the linear system for the coefficient vector c: 
c = np.linalg.solve(A, y)
# c = np.linalg.inv(A)@y
DET = np.linalg.det(A)

# Define plotting for the optimal polynomial:
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))
p_plot = np.polyval(np.flip(c), x_plot)

# Plot the data and the optimal polynomial: 
plt.figure(1)
plt.plot(x_plot, p_plot, 'r-', linewidth=4, label='Interpolant')
plt.plot(x, y, 'bo', linewidth = 6, label = 'Data')

# Customize the plot:
plt.title('Polynomial Interpolation', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format: 
plt.savefig('Polynomial_Interpolation.eps', format='eps')

# Show the plot:
plt.show()

# Output coefficients of polynomial interpolant: 
print(f"Coefficients = {c}")
print(f"Determinant = {DET}")


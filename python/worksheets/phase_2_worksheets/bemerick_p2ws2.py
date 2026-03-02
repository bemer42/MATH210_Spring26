import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# This program solves the systems presented in Phase 2, Worksheet 2:

###########
# Problem 1
###########

# Define the data points in vectors x and y 
x = np.array([-1, 0, 2, 3, 4])
y = np.array([0, 1, 0, 1, 2])

# Define the Vandermonde matrix 
A = np.vander(x, increasing=True)

# Solve the linear system for the coefficient vector c
c = la.solve(A, y)
# c = la.inv(A)@y
DET = la.det(A)

# Define plotting for the optimal polynomial
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))
p_plot = np.polyval(np.flip(c), x_plot)

# Plot the data and the optimal polynomial
fig1 = plt.figure(1)
plt.plot(x_plot, p_plot, 'b-', linewidth=4, label='Interpolant')
plt.plot(x, y, 'ko', linewidth = 9, label = 'Data')
plt.title('Polynomial Interpolation', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format
plt.savefig('p2ws2_figure_1.eps', format='eps')

# Show the plot
plt.show()

# Output coefficients of polynomial interpolant:
print(f"Coefficients = {c}")
print(f"Determinant = {DET}")



###########
# Problem 2
###########

# Define the matrix A and the right hand side b
x = np.array([0, 10, 20, 30])
y_1 = np.array([227.225, 249.623, 282.172, 308.282])
y_2 = np.array([984.736, 1148.364, 1263.638, 1330.141])
y_3 = np.array([78.298, 79.380, 82.184, 81.644])

A = np.vander(x, increasing=True)
DET = la.det(A)
print(f"Determinant = {DET}")

# Solve the linear system Ac = y for c 
c_1 = la.solve(A, y_1)
c_2 = la.solve(A, y_2)
c_3 = la.solve(A, y_3)

def f(x,y):
    c = la.solve(A,y)
    return np.polyval(np.flip(c), x)

# Create ararys of data and interpolant 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))
p1_plot = np.polyval(np.flip(c_1), x_plot)
p2_plot = np.polyval(np.flip(c_2), x_plot)
p3_plot = np.polyval(np.flip(c_3), x_plot)

# Plot the data and interpolant
fig2 = plt.figure(2)
plt.plot(x_plot+1980, f(x_plot, y_1), 'r-', linewidth=4, label = "USA")
plt.plot(x_plot+1980, p2_plot, 'b-', linewidth=4, label = "China")
plt.plot(x_plot+1980, p3_plot, 'g-', linewidth=4, label = "Germany")
plt.plot(x+1980, y_1, 'ko', linewidth=5)
plt.plot(x+1980, y_2, 'ko', linewidth=5)
plt.plot(x+1980, y_3, 'ko', linewidth=5)
plt.title('Population Interpolation', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Population (millions)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format
plt.savefig('p2ws2_figure_2.eps', format='eps')

# Show plot:
plt.show()

# Evaluate years: 
print(f"China Population in 1992 = {np.polyval(np.flip(c_2), 12)}")
print(f"USA Population in 1984 = {f(4,y_1)}")


###########
# Problem 3
###########

# Define the data points in the vector y 
y = np.array([0, 2, 1, -1], dtype = float)

# Define the matrix A
A = np.array([[1, 0, 0, 0],
              [1, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 2, 3]], dtype = float)

# Solve the linear system for the coefficient vector c
c = la.solve(A, y)

# Define plotting for the optimal polynomial
x_plot = np.linspace(0, 1, int(1e4))
p_plot = np.polyval(np.flip(c), x_plot)

# Plot the data and the optimal polynomial
fig3 = plt.figure(3)
plt.plot(x_plot, p_plot, 'b-', linewidth=4, label='Interpolant')
plt.plot((0,1), (0,2), 'ko', linewidth = 9, label = 'Data')
plt.title('Hermite Interpolation', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format
plt.savefig('p2ws2_figure_3.eps', format='eps')

# Show the plot
plt.show()


    
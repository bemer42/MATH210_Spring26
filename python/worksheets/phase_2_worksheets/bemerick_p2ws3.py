import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# This program solves the systems presented in Phase 2, Worksheet 3:

####################################
# Problem 1 -- exponential fitting
####################################
# Define the data vectors x and y:
x = np.linspace(1, 5, int(50))
y = (6*np.exp(2*x)) + 5*np.random.normal(0, 1, len(x))

# Build "Exponential Vandermonde":
A = np.vstack((np.ones(len(x)), x)).T

# Build Normal equation with the y data on log scale: 
A_norm = A.T@A
y_norm = A.T@np.log(y)

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = la.solve(A_norm, y_norm)
# c = la.inv(A_norm)@y_norm
# c = la.lstsq(A, y)[0]

# Define a and b: 
a = np.exp(c[0])
b = c[1]

# Create ararys of data for exponential curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and exponential curve fit:
plt.figure(1)
plt.plot(x_plot, a*np.exp(b*x_plot), 'r-', linewidth=4, label = "Exp Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Exponential Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('p2ws3_figure_1.eps', format='eps')

# Show plot:
plt.show()

# Output coefficients of exponential fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res) on log data:
ss_res = np.sum((np.log(y) - (c[0] + c[1]*x)) ** 2)

# Calculate the total sum of squares (SS_tot) on log data:
ss_tot = np.sum((np.log(y) - np.mean(np.log(y))) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")


####################################
# Problem 2 -- logistic fitting
####################################
# Define the data vectors x and y (y data must be in (0,1)):
x = np.linspace(2, 30, int(100))
y = (1/(1 + 860*np.exp(-.5*x))) + .05*np.random.normal(0, 1, len(x))

# Correct the data values that go less than zero or greater than 1:
y[y<0] = 0.01
y[y>1] = 0.99

# Build "Logistic Vandermonde":
A = np.vstack((np.ones(len(x)), -x)).T

# Build Normal equation with the y data on log scale: 
A_norm = A.T@A
y_norm = A.T@np.log(1/y-1)

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = la.solve(A_norm, y_norm)
# c = la.inv(A_norm)@y_norm
# c = la.lstsq(A, y)[0]

# Define a and b: 
a = np.exp(c[0])
b = c[1]

# Create ararys of data for exponential curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and exponential curve fit:
plt.figure(2)
plt.plot(x_plot, 1/(1+a*np.exp(-b*x_plot)), 'r-', linewidth=4, label = \
         "Logistic Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Logistic Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('p2ws3_figure_2.eps', format='eps')

# Show plot:
plt.show()

# Output coefficients of logistic fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res) on log data:
ss_res = np.sum((np.log(1/y-1) - (c[0] - c[1]*x)) ** 2)

# Calculate the total sum of squares (SS_tot) on log data:
ss_tot = np.sum((np.log(1/y-1) - np.mean(np.log(1/y-1))) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")


####################################
# Problem 3 -- agnesi fitting
####################################
# Define the data vectors x and y:
x = np.linspace(-5, 5, int(50))
y = (50/(.8*x**2 + 1)) + .3*np.random.normal(0, 1, len(x))

# Build "Agnesi Vandermonde":
A = np.vstack((np.ones(len(x)), -x**2 * y)).T

# Build Normal equation with the y data: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = la.solve(A_norm, y_norm)
# c = la.inv(A_norm)@y_norm
# c = la.lstsq(A, y)[0]

# Create ararys of data for Agnesi curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and Agnesi curve fit:
plt.figure(3)
plt.plot(x_plot, c[0]/(c[1]*x_plot**2 + 1), 'r-', linewidth=4, label = "Agnesi Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Agnesi Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('p2ws3_figure_3.eps', format='eps')

# Show plot:
plt.show()

# Output coefficients of Agnesi fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res):
ss_res = np.sum((y - (c[0]/(c[1]*x**2 + 1))) ** 2)

# Calculate the total sum of squares (SS_tot):
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")

####################################
# Problem 4 -- trignonmetric fitting
####################################
# Define the matrix A and the right hand side b:
x = np.linspace(1, 10, int(50))
y = (3 + np.sin(x) + 7*np.cos(x)) + 1*np.random.normal(0, 1, len(x))

# Build "Trig Vandermonde":
A = np.vstack((np.ones(len(x)), np.sin(x), np.cos(x))).T

# Build Normal equation: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system Ac = y for c: 
c = la.solve(A_norm, y_norm)
# c = la.inv(A_norm)@y_norm
# c = la.lstsq(A, y)[0]

# Create ararys of data and curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and trig curve fit:
plt.figure(4)
plt.plot(x_plot, c[0] + c[1]*np.sin(x_plot) + c[2]*np.cos(x_plot), 'r-', \
         linewidth=4, label = "Trig Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Trigonometric Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Show plot:
plt.show()

# Output coefficients of trigonometric curve fit: 
print(f"Coefficients = {c}")

# Save the file in high quality format: 
plt.savefig('p2ws3_figure_4.eps', format='eps')

# Calculate the residual sum of squares (SS_res)
ss_res = np.sum((y - (c[0] + c[1]*np.sin(x) + c[2]*np.cos(x))) ** 2)

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")

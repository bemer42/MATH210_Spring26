
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the data vectors x and y:
x = np.linspace(1, 19, int(150))
y = (10 + .2*x + np.sin(1*(x-3))) + .3*np.random.normal(0, 1, len(x))

# Define a custom nonlinear function with parameters:
def f(x, a, b, c, d, e):
    return a + b*x + c*np.sin(d*(x-e))
    
# Perform nonlinear curve fitting: 
c_fit = curve_fit(f, x, y)
c_fit = c_fit[0]

# Create ararys of data and optimized fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and optimized fit:
plt.figure(1)
plt.plot(x_plot, f(x_plot, c_fit[0], c_fit[1], c_fit[2], c_fit[3], c_fit[4]), \
         'r-', linewidth=4, label = "Nonlinear Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Nonlinear Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Show plot:
plt.show()

# Save the file in high quality format: 
plt.savefig('Nonlinear_Curve_Fitting.eps', format='eps')

# Show the plot:
plt.show()

# Output coefficients of polynomial fit: 
print(f"Coefficients = {c_fit}")

# Calculate the residual sum of squares (SS_res)
ss_res = np.sum((y - f(x, c_fit[0], c_fit[1], c_fit[2], c_fit[3], c_fit[4])) \
                 ** 2)

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")


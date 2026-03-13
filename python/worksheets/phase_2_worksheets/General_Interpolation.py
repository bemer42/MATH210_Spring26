
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define the data points in vectors x and y: 
x = np.arange(1, 31, 1)
y = np.random.normal(-2,54, size=len(x))

# Create interpolating function with scipy: 
f = interp1d(x, y, kind = 'cubic')

# Define plotting for the optimal polynomial:
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and the optimal polynomial: 
plt.figure(1)
plt.plot(x_plot, f(x_plot), 'r-', linewidth=4, label='Interpolant')
plt.plot(x, y, 'bo', linewidth = 6, label = 'Data')

# Customize the plot:
plt.title('General_Interpolation', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.legend()

# Save the file in high quality format: 
plt.savefig('General_Interpolation.eps', format='eps')

# Show the plot:
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Movie Loop Parameters
N_loop = 8
y0_0 = 1
y0_end = 30
vec = np.linspace(y0_0, y0_end, N_loop)
x0_vec, y0_vec = np.meshgrid(vec,vec)
x0_vec = x0_vec.flatten()
y0_vec = y0_vec.flatten()

# Define parameters
k1 = 1
m1 = 20
k2 = 1
m2 = 15
d1 = .06
d2 = .04
    
# Create mesh for the Vector Field: 
X, Y = np.meshgrid(np.linspace(0,30,31), np.linspace(0,30, 31))

# Define the Competing Species equations with x and y as input:
def VF(x, y):
    dxdt = k1 * x * (1-x/m1) - d1*x*y
    dydt = k2 * y * (1-y/m2) - d2*x*y
    return [dxdt, dydt]

# Calculate vector magnitudes for normalization:
L_norm = np.sqrt(VF(X, Y)[0]**2 + VF(X, Y)[1]**2)

# Plot the phase portrait using quiver:
plt.figure(1)
plt.quiver(X, Y, VF(X, Y)[0] / L_norm, VF(X, Y)[1] / L_norm, color=[.75, .75, .75], scale=30)

# Customizing the plot:
plt.title('Phase Portrait for Competing-Species', fontsize=28)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.grid(True, which='both')

# Loop through initial conditions
for i in range(N_loop**2):

    # Time Discretization
    t_0 = 0
    t_end = 500
    N_time = int(1e3)
    t_span = np.linspace(t_0, t_end, N_time)
      
    # Define initial conditions
    x_0 = x0_vec[i]
    y_0 = y0_vec[i]
    
    Y_0 = [x_0, y_0]
    
    def dYdt(t, Y):
        # Define variables
        x = Y[0]
        y = Y[1]
    
        # Define differential equations:
        dx_dt = k1 * x * (1-x/m1) - d1*x*y
        dy_dt = k2 * y * (1-y/m2) - d2*x*y
    
        # Assemble equations into a list (representing a column vector)
        return [dx_dt, dy_dt]
    
    # Implement ODE solver
    sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='BDF')
        
    # Extract Solutions
    x = sol.y[0, :]
    y = sol.y[1, :]
    t = sol.t
    
    # Plot trajectory  
    plt.plot(x,y,'k-',linewidth = 2)
    plt.plot(x[0],y[0],'go',markersize=5)
    if i == N_loop**2-1:
        plt.plot(x[-1], y[-1], 'ro',markersize =5)
        
plt.show()
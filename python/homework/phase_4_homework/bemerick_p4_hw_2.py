import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# This file solves Homework 4, Problem 2:

# Time Discretization
t_0 = 0
t_end = 6
N_time = int(1e4)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
R = 60

# Define initial conditions
x_0 = 10
y_0 = 20

Y_0 = [x_0, y_0]

def dYdt(t, Y):
    # Define variables
    x = Y[0]
    y = Y[1]

    # Define equations
    dx_dt = R*x - 4*x**2 - 3*x*y
    dy_dt = -2*y + x*y

    # Assemble equations into a list (representing a column vector)
    return [dx_dt, dy_dt]

# Implement ODE solver
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='BDF')

# Extract Solutions
x = sol.y[0, :]
y = sol.y[1, :]
t = sol.t

# Plot Solution
plt.figure(1,figsize=(10, 6))
plt.plot(t, x, 'k-', linewidth=5, label='Prey, x(t)')
plt.plot(t, y, 'b-', linewidth=5, label='Predator, y(t)')
plt.title('Time Series Plot', fontsize=28)
plt.xlabel('Time (t)', fontsize=26)
plt.ylabel('Populations (x(t) and y(t))', fontsize=26)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(visible = True, which = 'both')

plt.savefig('p4hw_figure_2.eps', format = 'eps', dpi = 300)

# Create mesh for the Vector Field: 
mx = np.floor(np.min(x))
Mx = np.ceil(np.max(x))
my = np.floor(np.min(y))
My = np.ceil(np.max(y))
X, Y = np.meshgrid(np.linspace(mx,Mx,30), np.linspace(my, My, 30))

# Define the Predator-Prey equations with x and y as input:
def dYdt(x, y):
    dxdt = R*x - 4*x**2 - 3*x*y
    dydt = -2*y + x*y
    return [dxdt, dydt]

# Calculate vector magnitudes for normalization:
L_norm = np.sqrt(dYdt(X, Y)[0]**2 + dYdt(X, Y)[1]**2)

# Plot the phase portrait using quiver:
plt.figure(2,figsize=(10, 6))
plt.quiver(X, Y, dYdt(X, Y)[0] / L_norm, dYdt(X, Y)[1] / L_norm, color=[.75, .75, .75], scale=25)
plt.plot(x,y,'k-',linewidth = 5)

# Customizing the plot:
plt.title('Phase Portrait and Vector Field', fontsize=28)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.grid(True, which='both')

plt.savefig('p4hw_figure_3.eps', format = 'eps', dpi = 300)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Time Discretization
t_0 = 0
t_end = 500
N_time = int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
a = .1
b = .02
c = .04
d = .02

# Define initial conditions
x_0 = 2
y_0 = 10

Y_0 = [x_0, y_0]

def dYdt(t, Y):
    # Define variables
    x = Y[0]
    y = Y[1]

    # Define equations
    dx_dt = a*x - b*x*y
    dy_dt = -c*y + d*x*y

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
plt.title('Predator-Prey Trajectory', fontsize=20)
plt.xlabel('Time (t)', fontsize=16)
plt.ylabel('Populations (x(t) and y(t))', fontsize=16)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(True)
plt.show()

# Create mesh for the Vector Field: 
mx = np.floor(np.min(x))
Mx = np.ceil(np.max(x))
my = np.floor(np.min(y))
My = np.ceil(np.max(y))
X, Y = np.meshgrid(np.linspace(mx,Mx,30), np.linspace(my, My, 30))

# Define the Predator-Prey equations with x and y as input:
def dYdt(x, y):
    dxdt = a*x - b*x*y
    dydt = -c*y + d*x*y
    return [dxdt, dydt]

# Calculate vector magnitudes for normalization:
L_norm = np.sqrt(dYdt(X, Y)[0]**2 + dYdt(X, Y)[1]**2)

# Plot the phase portrait using quiver:
plt.figure(2)
plt.quiver(X, Y, dYdt(X, Y)[0] / L_norm, dYdt(X, Y)[1] / L_norm, color=[.75, .75, .75], scale=25)
plt.plot(x,y,'k-',linewidth = 5)

# Customizing the plot:
plt.title('Phase Portrait for Predator-Prey', fontsize=28)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.grid(True, which='both')
plt.show()


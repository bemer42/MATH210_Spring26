import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time Discretization
t_0 = 0
t_end = 140
N_time = int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
k = 1
m = 3
b = 0.1

# Define initial conditions
y_0 = 0
v_0 = 9

Y_0 = [y_0, v_0]

# Forcing function: 
def f(t):
    return np.sin(np.sqrt(k/m)*t)

def dYdt(t, Y):
    # Define variables
    y = Y[0]
    v = Y[1]
        
    # Define equations
    dy_dt = v
    dv_dt = -k/m*y - b/m*v + k*f(t)
    
    # Assemble equations into a list (representing a column vector)
    return [dy_dt, dv_dt]

# Implement ODE solver
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='RK45')

# Extract Solutions
y = sol.y[0, :]
v = sol.y[1, :]
t = sol.t

# Plot Solution
plt.figure(1,figsize=(10, 6))
plt.plot(t, y, 'b-', linewidth=5, label='Displacement, y(t)')
plt.plot(t, v, 'r-', linewidth=5, label='Velocity, v(t)')
plt.title('Mass-Spring Trajectory', fontsize=20)
plt.xlabel('Time (t)', fontsize=16)
plt.ylabel('Displacement/Velocity (y(t) and v(t))', fontsize=16)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(True)
plt.show()

# Create mesh for the Vector Field: 
my = np.floor(np.min(y))
My = np.ceil(np.max(y))
mv = np.floor(np.min(v))
Mv = np.ceil(np.max(v))
Y, V = np.meshgrid(np.linspace(my,My,30), np.linspace(mv, Mv, 30))
      
# Define the Predator-Prey equations with x and y as input:
def dYdt(y, v):
    dydt = v
    dvdt = -k/m*y - b/m*v
    return [dydt, dvdt]

# Calculate vector magnitudes for normalization:
L_norm = np.sqrt(dYdt(Y, V)[0]**2 + dYdt(Y, V)[1]**2)

# Plot the phase portrait using quiver:
plt.figure(2)
plt.quiver(Y, V, dYdt(Y, V)[0] / L_norm, dYdt(Y, V)[1] / L_norm, color=[.75, .75, .75], scale=25)
plt.plot(y,v,'k-',linewidth = 5)

# Customizing the plot:
plt.title('Phase Portrait for Mass-Spring System', fontsize=28)
plt.xlabel('Displacement (y)', fontsize=26)
plt.ylabel('Velocity (v)', fontsize=26)
plt.grid(True, which='both')
plt.show()

    

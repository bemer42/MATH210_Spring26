import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
k = .2
m = 10

# Define right-hand side function (Logistic ODE)
def dydt(t, y):
    return k * y 

# Time Discretization
N = int(1e2)
t_0 = 0
t_end = 50
t_span = np.linspace(t_0, t_end, N)

# Initial condition
y_0 = 0.3

# Solve the differential equation using solve_ivp
sol = solve_ivp(dydt, [t_0, t_end], [y_0], t_eval=t_span)

# Extract time and solution
t = sol.t
y_solution = sol.y[0, :]

# Define analytical solution: 
def y(t): 
    return y_0*np.exp(k*t)

# Plot a trajectory
plt.figure(1)
plt.plot(t, y_solution, 'ko', linewidth=5, label='Numerical Solution')
plt.plot(t,y(t), 'r-',linewidth = 4, label='Exact Solution')
plt.plot(t, m * np.ones_like(t), 'k:', linewidth=2)

# Customize the plot
plt.title('Trajectory of Exponential Growth/Decay', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('y(t)', fontsize=16)
plt.xlim([t_0, t_end])
plt.ylim([0, 1.5*np.max(y_solution)])
plt.grid(True, which='both')
plt.legend()

plt.show()

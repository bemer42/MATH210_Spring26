import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Time Discretization
t_0 = 0
t_end = 24*10
N_time = t_end*int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
k1 = 0.1
k2 = 0.01

# Define initial conditions
Ag_0 = 0
Ab_0 = 0

A_0 = [Ag_0, Ab_0]

# Dose parameters
Dose = 10.0
Freq = 8.0
Dura = .5

def I(t):
    phase = t % Freq
    return (Dose / Dura) if phase < Dura else 0.0

def dAdt(t, A):
    # Define variables
    Ag = A[0]
    Ab = A[1]

    # Define equations
    dAg_dt =  I(t)   -k1*Ag
    dAb_dt =  k1*Ag - k2*Ab

    # Assemble equations into a list (representing a column vector)
    return [dAg_dt, dAb_dt]

# Implement ODE solver
sol = solve_ivp(dAdt, [t_0, t_end], A_0, t_eval=t_span, max_step = Dura, method='BDF')

# Extract Solutions
Ag = sol.y[0, :]
Ab = sol.y[1, :]
t = sol.t

# Plot Solution
plt.figure(1,figsize=(10, 6))
plt.plot(t/24, Ag, 'k-', linewidth=5, label='GI-Tract')
plt.plot(t/24, Ab, 'b-', linewidth=5, label='Bloodstream')
plt.title('Pharmacokinetics Trajectory', fontsize=20)
plt.xlabel('Time (t)', fontsize=16)
plt.ylabel('Concentrations A_g(t) and A_b(t)', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()



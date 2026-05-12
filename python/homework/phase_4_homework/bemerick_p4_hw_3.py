import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# This file solves Homework 4, Problem 3:

# Time Discretization
t_0 = 0
t_end = 8
N_time = int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
k1 = .09
k2 = .8
k3 = .07
k4 = .6
k12 = .05
k13 = .5
k14 = .06
k23 = .7
k24 = .08
k34 = .9

# Define initial conditions
P_0 = 1
P1_0 = 0
P2_0 = 0
P3_0 = 0
P4_0 = 0

Y_0 = [P_0, P1_0, P2_0, P3_0, P4_0]

def dYdt(t, Y):
    # Define variables
    P = Y[0]
    P1 = Y[1]
    P2 = Y[2]
    P3 = Y[3]
    P4 = Y[4]

    # Define equations
    dP_dt = -k1*P - k2*P - k3*P - k4*P
    dP1_dt = k1*P - k12*P1 - k13*P1 - k14*P1
    dP2_dt = k2*P + k12*P1 - k23*P2 - k24*P2
    dP3_dt = k3*P + k13*P1 + k23*P2 - k34*P3
    dP4_dt = k4*P + k14*P1 + k24*P2 + k34*P3

    # Assemble equations into a list (representing a column vector)
    return [dP_dt, dP1_dt, dP2_dt, dP3_dt, dP4_dt]

# Implement ODE solver
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='RK45')

# Extract Solutions
P = sol.y[0, :]
P1 = sol.y[1, :]
P2 = sol.y[2, :]
P3 = sol.y[3, :]
P4 = sol.y[4, :]
t = sol.t

# Plot Solution
plt.figure(1,figsize=(10, 6))
plt.plot(t, P, 'k-', linewidth=7, label='P(t)')
plt.plot(t, P1, 'k-', linewidth=6, label='P_1(t)')
plt.plot(t, P2, 'k-', linewidth=5, label='P_2(t)')
plt.plot(t, P3, 'k-', linewidth=4, label='P_3(t)')
plt.plot(t, P4, 'k-', linewidth=3, label='P_4(t)')
plt.title('Peptide Decomposition', fontsize=28)
plt.xlabel('Time (t)', fontsize=26)
plt.ylabel('Peptide Concentrations', fontsize=26)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(True, which = 'both')

plt.savefig('p4hw_figure_4.eps', format = 'eps', dpi = 300)
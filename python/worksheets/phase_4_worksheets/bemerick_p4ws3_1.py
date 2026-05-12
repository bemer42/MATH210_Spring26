import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time Discretization
t_0 = 0
t_end = 500
N_time = int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
N     = 303824640
alpha = 0.00872
beta  = 0.32851429
gamma = 0.11673
mu    = 0.00872

# Define initial conditions
S_0 = N-1
I_0 = 1
R_0 = 0

Y_0 = [S_0, I_0, R_0]

def dYdt(t, Y):
    # Define variables
    S = Y[0]
    I = Y[1]
    R = Y[2]
    
    # Define equations
    dS_dt = alpha*(S+I+R) - beta*S*I/N - mu*S
    dI_dt = beta*S*I/N - gamma*I - mu*I
    dR_dt = gamma*I - mu*R
    
    # Assemble equations into a list (representing a column vector)
    return [dS_dt, dI_dt, dR_dt]

# Implement ODE solver
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='BDF')

# Extract Solutions
S = sol.y[0, :]
I = sol.y[1, :]
R = sol.y[2, :]
t = sol.t

# Plot Solution
plt.figure(1,figsize=(10, 6))
plt.plot(t, S/N, 'b-', linewidth=5, label='Susceptibles, S(t)')
plt.plot(t, I/N, 'r-', linewidth=5, label='Infectious, I(t)')
plt.plot(t, R/N, 'g-', linewidth=5, label='Recoveries, R(t)')
plt.title('SIR Trajectory', fontsize=20)
plt.xlabel('Time (t)', fontsize=16)
plt.ylabel('Populations (S(t), I(t)), R(t))', fontsize=16)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(True)
plt.show()

# Create mesh for the Vector Field: 
mS = np.floor(np.min(S))
MS = np.ceil(np.max(S))
mI = np.floor(np.min(I))
MI = np.ceil(np.max(I))
SS, II = np.meshgrid(np.linspace(mS,MS,30), np.linspace(mI, MI, 30))

# Define the Predator-Prey equations with x and y as input:
def VF(S, I):
    dSdt = alpha*(S+I+(N-I-S)) - beta*S*I/N - mu*S
    dIdt = beta*S*I/N - gamma*I - mu*I
    return [dSdt, dIdt]

# Calculate vector magnitudes for normalization:
L_norm = np.sqrt(VF(SS, II)[0]**2 + VF(SS, II)[1]**2)

# Plot the phase portrait using quiver:
plt.figure(2)
plt.quiver(SS, II, VF(SS, II)[0] / L_norm, VF(SS, II)[1] / L_norm, color=[.75, .75, .75], scale=25)
plt.plot(S,I,'k-',linewidth = 5)

# Customizing the plot:
plt.title('Phase Portrait for SIR Model', fontsize=28)
plt.xlabel('Susceptbiles', fontsize=26)
plt.ylabel('Infectious', fontsize=26)
plt.grid(True, which='both')
plt.show()

# Print the Basic Reproduction Number: 
print(f"Basic Reproduction Number, R0 = {beta/gamma}")

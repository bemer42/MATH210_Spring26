#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 01:04:23 2024

@author: brooksemerick
"""

import numpy as np   
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system parameters
t_0 = 0    # Initial time
dt = 0.002  # Time step
t_end = 80  # End time
t_eval = np.arange(t_0, t_end, dt)

# Initial conditions for x, y, z and their velocities
x_0, vx_0 = 1, -1
y_0, vy_0 = 1, 1
z_0, vz_0 = 1, 1

# Combine initial conditions into a single vector
U_0 = [x_0, vx_0, y_0, vy_0, z_0, vz_0]

# Define system parameters
G = 5
M = 10
beta = 2

# Define right-hand side functions
def dUdt(t, U):
    x, vx, y, vy, z, vz = U
    r_squared = x**2 + y**2 + z**2
    r_pow_beta = r_squared ** ((1 + beta) / 2)
    
    fx = vx
    gx = -G * M * x / r_pow_beta
    
    fy = vy
    gy = -G * M * y / r_pow_beta
    
    fz = vz
    gz = -G * M * z / r_pow_beta
    
    return [fx, gx, fy, gy, fz, gz]

# Solve using Runge-Kutta Method (solve_ivp is similar to MATLAB's ode45)
sol = solve_ivp(dUdt, [t_0, t_end], U_0, t_eval=t_eval, method='RK45')

# Extract solutions for x, y, z and their velocities
x = sol.y[0]
vx = sol.y[1]
y = sol.y[2]
vy = sol.y[3]
z = sol.y[4]
vz = sol.y[5]
t = sol.t

# Plot results: 3D Phase Space Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z, 'k-', linewidth=4)
ax.scatter(0, 0, 0, color='r', s=100)  # Stationary body
ax.set_title('Trajectory in Phase Space', fontsize=20)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('z', fontsize=16)
ax.grid(True)
plt.show()

# Plot results: Time-Series Plot for x(t) and y(t)
plt.figure()
plt.plot(t, x, 'b', linewidth=4, label='x(t)')
plt.plot(t, y, 'k', linewidth=4, label='y(t)')
plt.title('Time-Series Plot', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('x(t) and y(t)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
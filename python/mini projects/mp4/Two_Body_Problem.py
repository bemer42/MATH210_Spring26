#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:54:02 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define time discretization
t_0 = 0
t_end = 40
N_time = int(1e5)
t_span = np.linspace(t_0, t_end, N_time)

# Define initial conditions
x1_0 = 1
y1_0 = 1
z1_0 = 1
vx1_0 = -1.5
vy1_0 = 1.5
vz1_0 = -1

x2_0 = -1
y2_0 = -1
z2_0 = -1
vx2_0 = 1.5
vy2_0 = -1.5
vz2_0 = 2

# Define system parameters
G = .1
m1 = 1000
m2 = 700
b = 2

# Define right-hand side functions
def dYdt(t, Y):
    
    # Define the input functions for body 1
    x1 = Y[0]
    vx1 = Y[1]
    y1 = Y[2]
    vy1 = Y[3]
    z1 = Y[4]
    vz1 = Y[5]

    # Define the input functions for body 2
    x2 = Y[6]
    vx2 = Y[7]
    y2 = Y[8]
    vy2 = Y[9]
    z2 = Y[10]
    vz2 = Y[11]
    
    # Define the derivatives for body 1: 
    dx1dt = vx1
    dy1dt = vy1
    dz1dt = vz1
    dvx1dt = -G*m2*(x1-x2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)
    dvy1dt = -G*m2*(y1-y2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)
    dvz1dt = -G*m2*(z1-z2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)

    # Define the derivatives for body 2:
    dx2dt = vx2
    dy2dt = vy2
    dz2dt = vz2
    dvx2dt = G*m1*(x1-x2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)
    dvy2dt = G*m1*(y1-y2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)
    dvz2dt = G*m1*(z1-z2)/((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**((1+b)/2)

    return [dx1dt, dvx1dt, dy1dt, dvy1dt, dz1dt, dvz1dt,\
            dx2dt, dvx2dt, dy2dt, dvy2dt, dz2dt, dvz2dt]    


# Initial conditions array
Y_0 = [x1_0, vx1_0, y1_0, vy1_0, z1_0, vz1_0,\
       x2_0, vx2_0, y2_0, vy2_0, z2_0, vz2_0]

# Solve using Runge-Kutta Method (solve_ivp equivalent to ode45)
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='RK45')

# Gather the solutions for x, y, vx, and vy
x1 = sol.y[0]
y1 = sol.y[2]
z1 = sol.y[4]
x2 = sol.y[6]
y2 = sol.y[8]
z2 = sol.y[10]
t = sol.t

# Define a vector that traces the center of mass: 
xcm = (m1*x1 + m2*x2)/(m1+m2)
ycm = (m1*y1 + m2*y2)/(m1+m2)
zcm = (m1*z1 + m2*z2)/(m1+m2)

# Create relative markersizes:    
mark1 = np.max([m1/np.max([m1, m2])*15, 1])
mark2 = np.max([m2/np.max([m1, m2])*15, 1])

# Create figure:
fig1 = plt.figure(1, figsize=(13,8))
ax1 = fig1.add_subplot(111, projection='3d')
plt.plot(x1,y1,z1,'b-',linewidth = 5)
plt.plot(x2,y2,z2,'r-',linewidth = 5)
plt.plot(xcm,ycm,zcm,'k-',linewidth = 2)
plt.plot(x1[0],y1[0],z1[0], 'go',markersize = mark1)
plt.plot(x2[0],y2[0],z2[0], 'go', markersize = mark2)
plt.plot(x1[-1],y1[-1],z1[-1], 'bo',markersize = 5)
plt.plot(x2[-1],y2[-1],z2[-1], 'ro', markersize = 5)

# Customizing the plot:
plt.title('Phase Portrait for Two Body Problem', fontsize=28)
ax1.set_xlim(np.min([x1, x2]), np.max([x1, x2]))
ax1.set_ylim(np.min([y1, y2]), np.max([y1, y2]))
ax1.set_zlim(np.min([z1, z2]), np.max([z1, z2]))
ax1.set_xlabel('x', fontsize=26)
ax1.set_ylabel('y', fontsize=26)
ax1.set_zlabel('z', fontsize=26)
plt.grid(True, which='both')

# Create a movie plot:
fig2 = plt.figure(2, figsize=(13,8))
ax2 = fig2.add_subplot(111, projection='3d')
plt.title('Movie Plot for Two Body Problem', fontsize=28)
ax2.set_xlim(np.min([x1, x2]), np.max([x1, x2]))
ax2.set_ylim(np.min([y1, y2]), np.max([y1, y2]))
ax2.set_zlim(np.min([z1, z2]), np.max([z1, z2]))
ax2.set_xlabel('x', fontsize=26)
ax2.set_ylabel('y', fontsize=26)
ax2.set_zlabel('z', fontsize=26)
plt.grid(True, which='both')

# Plot initial positions for both planets
body1, = ax2.plot([], [], [], 'bo',markersize=mark1)
body2, = ax2.plot([], [], [], 'ro',markersize=mark2)
cm, = ax2.plot([], [], [], 'ko',markersize=.5)
tail1, = ax2.plot([], [], [], 'b-', linewidth=1)  
tail2, = ax2.plot([], [], [], 'r-', linewidth=1)  
cmtail, = ax2.plot([], [], [], 'k-', linewidth=.5)  

# Initialize the animation function
def init():
    body1.set_data_3d([], [], [])
    body2.set_data_3d([], [], [])
    cm.set_data_3d([], [], [])
    tail1.set_data_3d([], [], [])
    tail2.set_data_3d([], [], [])
    cmtail.set_data_3d([], [], [])
    return body1, tail1, body2, tail2, cm, cmtail

# Update the animation for each frame
def update(frame):
    body1.set_data_3d([x1[frame]], [y1[frame]], [z1[frame]])
    body2.set_data_3d([x2[frame]], [y2[frame]], [z2[frame]])
    cm.set_data_3d([xcm[frame]], [ycm[frame]], [zcm[frame]])
    tail_len = 300
    ts = np.max([0,frame-tail_len])
    tail1.set_data_3d(x1[ts:frame], y1[ts:frame], z1[ts:frame])
    tail2.set_data_3d(x2[ts:frame], y2[ts:frame], z2[ts:frame])
    cmtail.set_data_3d(xcm[ts:frame], ycm[ts:frame], zcm[ts:frame])
    return body1, tail1, body2, tail2, cm, cmtail

# Create the animation
dframe = 200
ani = FuncAnimation(fig2, update, frames=range(0, len(t), dframe), init_func=init, interval=10, blit=False)

plt.show()

ani.save('Two_Body_Movie.mp4', writer='ffmpeg', fps=30)



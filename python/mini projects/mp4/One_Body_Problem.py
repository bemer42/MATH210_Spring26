import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


# Define time discretization
t_0 = 0
t_end = 10
N_time = int(1e4)
t_span = np.linspace(t_0, t_end, N_time)

# Define initial conditions
x_0 = 1
y_0 = 1
z_0 = 1
vx_0 = -1.5
vy_0 = 1.5
vz_0 = -1

# Define system parameters
G = .1
M = 1800

# Define right-hand side functions
def dYdt(t, Y):
    
    # Define the input functions
    x = Y[0]
    vx = Y[1]
    y = Y[2]
    vy = Y[3]
    z = Y[4]
    vz = Y[5]
    
    # Define the derivatives: 
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -G*M*x/(x**2+y**2+z**2)**(3/2)
    dvydt = -G*M*y/(x**2+y**2+z**2)**(3/2)
    dvzdt = -G*M*z/(x**2+y**2+z**2)**(3/2)

    return [dxdt, dvxdt, dydt, dvydt, dzdt, dvzdt]    


# Initial conditions array
Y_0 = [x_0, vx_0, y_0, vy_0, z_0, vz_0]

# Solve using Runge-Kutta Method (solve_ivp equivalent to ode45)
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='RK45')

# Gather the solutions for x, y, vx, and vy
x = sol.y[0]
y = sol.y[2]
z = sol.y[4]
t = sol.t

# Create figure and axis for animation
fig1 = plt.figure(1, figsize=(10,6))
ax1 = fig1.add_subplot(111, projection='3d')
plt.plot(x,y,z,'k-',linewidth = 1)
plt.plot(0,0,0, 'bo',markersize = 20)
plt.plot(x[0], y[0],z[0], 'go', markersize = 10)
plt.plot(x[-1], y[-1], z[-1], 'ro', markersize = 3)

# Customizing the plot:
plt.title('Phase Portrait for One Body Problem', fontsize=28)
ax1.set_xlim(np.min(x), np.max(x))
ax1.set_ylim(np.min(y), np.max(y))
ax1.set_zlim(np.min(z), np.max(z))
ax1.set_xlabel('x', fontsize=26)
ax1.set_ylabel('y', fontsize=26)
ax1.set_zlabel('z', fontsize=26)
plt.grid(True, which='both')

# Create a movie plot:
fig2 = plt.figure(2, figsize=(10,6))
ax2 = fig2.add_subplot(111, projection='3d')
plt.plot(0,0,0, 'bo',markersize = 20)
plt.title('Movie Plot for One Body Problem', fontsize=28)
ax2.set_xlim(np.min(x), np.max(x))
ax2.set_ylim(np.min(y), np.max(y))
ax2.set_zlim(np.min(z), np.max(z))
ax2.set_xlabel('x', fontsize=26)
ax2.set_ylabel('y', fontsize=26)
ax2.set_zlabel('z', fontsize=26)
plt.grid(True, which='both')

# Plot initial positions for both planets
body, = ax2.plot([], [], [], 'go',markersize=10)
tail, = ax2.plot([], [], [], 'k-', linewidth=1)  

# Initialize the animation function
def init():
    body.set_data_3d([], [], [])
    tail.set_data_3d([], [], [])
    return body, tail

# Update the animation for each frame
def update(frame):
    body.set_data_3d([x[frame]], [y[frame]], [z[frame]])
    tail_len = 100
    ts = np.max([0,frame-tail_len])
    tail.set_data_3d(x[ts:frame], y[ts:frame], z[ts:frame])
    return body, tail

# Create the animation
dframe = 100
ani = FuncAnimation(fig2, update, frames=range(0, len(t), dframe), init_func=init, interval=10, blit=True)

ani.save("One_Body_Movie.gif", writer=PillowWriter(fps=30))

plt.show()



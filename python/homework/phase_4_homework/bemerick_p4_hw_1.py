import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# This file solves Homework 4, Problem 1 and plots thirty solutions
# on the same plot to demonstrate how different initial conditions 
# have different looking trajectories. 

# Movie Loop Parameters
N_loop = 31
y0_0 = 1
y0_end = 30
y0_vec = np.linspace(y0_0, y0_end, N_loop)

# Loop through initial conditions
for i in range(N_loop):

    #Parameters
    k = 0.7
    m = 10
    M = 20
     
    #Define the right-handed side function
    def dydt(t, y):
        return k * y * (1-y/M)*(y/m - 1)
     
    #Time Discretization
    N = int(1e3)
    t_0 = 0
    t_end = 10
    t_span = np.linspace(t_0, t_end, N)
     
    #Initial condition
    y0 = y0_vec[i]
     
    #Solve the initial value problem using solve ivp
    sol1 = solve_ivp(dydt, [t_0, t_end], [y0], t_eval = t_span)
     
    #Extract time and solution
    t = sol1.t
    y = sol1.y[0, :]
    
    #plot the numerical solution
    plt.figure(1)
    plt.plot(t, y, 'b-', linewidth=2)
    if i == N_loop-1:
        plt.plot(t,m*np.ones_like(t),'k:', linewidth=3)
        plt.plot(t,M*np.ones_like(t),'k:', linewidth=3)
                           
    #Customize Plot:
    plt.title('Population Growth with Alee Effect', fontsize=20)
    plt.xlabel('Time, t', fontsize=15)
    plt.ylabel('Concentration, y(t)', fontsize=15)
    plt.gca().tick_params(labelsize=10)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.ylim(-1, y0_end+1)
    
    
plt.savefig('p4hw_figure_1.eps', format = 'eps', dpi = 300) 
plt.show()
 
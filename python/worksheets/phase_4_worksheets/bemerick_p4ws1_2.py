import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Movie Loop Parameters
N_loop = 20
y0_0 = 1
y0_end = 120
y0_vec = np.linspace(y0_0, y0_end, N_loop)

# Loop through initial conditions
for i in range(N_loop):
    
    # Parameters
    k = .2
    ys = 72
    
    # Define right-hand side function (Logistic Equation)
    def dydt(t, y):
        return -k * (y - ys)
    
    # Time Discretization
    N = int(1e3)
    t_0 = 0
    t_end = 30
    t_span = np.linspace(t_0, t_end, N)
    
    # Initial condition
    y_0 = y0_vec[i]
    
    # Solve the differential equation using solve_ivp
    sol = solve_ivp(dydt, [t_0, t_end], [y_0], t_eval=t_span, method = 'RK45')
    
    # Extract time and solution
    t = sol.t
    y_solution = sol.y[0, :]
        
    # Plot a trajectory
    plt.figure(1)
    plt.plot(t, y_solution, 'b-', linewidth=5, label='Temperature of Substance')
    plt.plot(t, ys * np.ones_like(t), 'k:', linewidth=2, label='Room Temperature')
    
    # Customize the plot
    plt.title('Newtons Law of Cooling', fontsize=20)
    plt.xlabel('t', fontsize=16)
    plt.ylabel('y(t)', fontsize=16)
    plt.xlim([t_0, t_end])
    plt.grid(True, which='both')
    if i == 0:
        plt.legend()
    
plt.show()
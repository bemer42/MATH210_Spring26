import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script will provide the minimization code and plotting
# code for phase 1, mini-project 1. 

##############################################################################
# Define and plot the cost function for various parameter values.
##############################################################################
# Discretize the variable space
N = int(1e3)
r = np.linspace(1e-1,4,N)

# Define all parameters
ml = 1.0        # mass of payload
pl = 1.0        # density of payload          
hl = 1.0        # height of payload cylinder
pf = 1.0        # density of fuel
hf = 1.0        # height of fuel cylinder
cw = 1.0        # cost/area of wings
cs = 1.0        # cost/area of siding
ct = 1.0        # cost/area of nose tip

# Define the cost function
def f(r,ml,pl,hl,pf,hf,cw,cs,ct):
    pi = np.pi
    p  = 1/(1/pl + 3/pf)
    a  = 2592*ct*pi**3*p**2/75/ml**2
    b  = ml**2/36/p**2/pi**2
    c  = 12*cs*ml/5/p
    d  = 23*cw*ml**2/600/p**2/pi**2
    return a*(np.sqrt(1+b/r**4)**3-1)*r**8 + c/r + d/r**2

# Find the minimial r value given a set of parameter values
MIN = optimize.minimize_scalar(lambda r: f(r,ml,pl,hl,pf,hf,cw,cs,ct), bracket = (np.min(r), np.max(r)))

r_min = float(MIN.x)
f_min = float(MIN.fun)
buff  = .5

# Plot the graph of the obj fun with minimum
fig1 = plt.figure(1)
plt.plot(r, f(r,ml,pl,hl,pf,hf,cw,cs,ct), 'k-', linewidth=3, label='cost')
plt.plot(r_min, f_min, 'ro', markersize=7, label='minimum cost')
plt.title('Cost (C) versus Radius (r)', fontsize=18)
plt.xlabel('r (radius)', fontsize=15)
plt.ylabel('C (cost)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([(1-buff)*r_min, (1+buff)*r_min])
plt.ylim([(1-buff)*f_min, (1+buff)*f_min])
plt.legend()

# Figure 2: Effect of varying density of load
pl_list = [1.0, 2.0, 10.0, 1000.0]   
colors = plt.cm.viridis(np.linspace(0, 1, len(pl_list)))

fig2 = plt.figure(2)
for pl_i, col in zip(pl_list, colors):
    MIN_i = optimize.minimize_scalar(
        lambda rr: f(rr, ml, pl_i, hl, pf, hf, cw, cs, ct),
        bracket=(np.min(r), np.max(r)))
    
    rmin_i = float(MIN_i.x)
    fmin_i = float(MIN_i.fun)
    
    plt.plot(r, f(r, ml, pl_i, hl, pf, hf, cw, cs, ct),
             color=col, linewidth=2, label=fr"$\rho_l$={pl_i:g}")
    plt.plot(rmin_i, fmin_i, 'o', color=col, markersize=5)
plt.title('Cost for Varying Load Density', fontsize=18)
plt.xlabel('r (radius)', fontsize=15)
plt.ylabel('C (cost)', fontsize=15)
plt.grid(True, which='both')
plt.xlim([(1-buff)*r_min, (1+buff)*r_min])
plt.ylim([(1-buff)*f_min, (1+buff)*f_min])
plt.legend()


# Figure 3: Effect of varying load mass
ml_list = [0.01, 0.1, 0.5, 1.0]   
colors = plt.cm.viridis(np.linspace(0, 1, len(ml_list)))

fig2 = plt.figure(3)
for ml_i, col in zip(ml_list, colors):
    MIN_i = optimize.minimize_scalar(
        lambda rr: f(rr, ml_i, pl, hl, pf, hf, cw, cs, ct),
        bracket=(np.min(r), np.max(r)))
    
    rmin_i = float(MIN_i.x)
    fmin_i = float(MIN_i.fun)
    
    plt.plot(r, f(r, ml_i, pl, hl, pf, hf, cw, cs, ct),
             color=col, linewidth=2, label=fr"$m_l$={ml_i:g}")
    plt.plot(rmin_i, fmin_i, 'o', color=col, markersize=5)
plt.title('Cost for Varying Load Mass', fontsize=18)
plt.xlabel('r (radius)', fontsize=15)
plt.ylabel('C (cost)', fontsize=15)
plt.grid(True, which='both')
plt.xlim([0, 2])
plt.ylim([-0, 20])
plt.legend()

##############################################################################
# Figure 4 (modified): C_min versus rho_L (pl) for multiple fuel densities rho_F (pf)
##############################################################################
fig4 = plt.figure(4)
rhoL_vec = np.logspace(0, 3, 100)
rhoF_list = np.logspace(-1,3,3)  
Cmin_mat = np.zeros((len(rhoF_list), len(rhoL_vec)))
rmin_mat = np.zeros_like(Cmin_mat)
r_lo, r_hi = float(np.min(r)), float(np.max(r))
colors = plt.cm.plasma(np.linspace(0, 1, len(rhoF_list)))
for j, (rhoF_j, col) in enumerate(zip(rhoF_list, colors)):
    for i, rhoL in enumerate(rhoL_vec):
        MIN_ij = optimize.minimize_scalar(
            lambda rr: f(rr, ml, rhoL, hl, rhoF_j, hf, cw, cs, ct),
            method="bounded",
            bounds=(r_lo, r_hi)
        )
        rmin_mat[j, i] = float(MIN_ij.x)
        Cmin_mat[j, i] = float(MIN_ij.fun)
    plt.plot(rhoL_vec, Cmin_mat[j, :], color=col, linewidth=2,
             label=fr"$\rho_F$={rhoF_j:g}")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Minimum Cost $C_{\min}$ versus Payload Density $\rho_L$ (for varying $\rho_F$)", fontsize=16)
plt.xlabel(r"$\rho_L$ (payload density, pl)", fontsize=14)
plt.ylabel(r"$C_{\min}$", fontsize=14)
plt.grid(True, which='both')
plt.legend()


##############################################################################
# Figure 5: C_min versus rho_L (pl) for multiple payload masses m_L (ml)
##############################################################################
fig5 = plt.figure(5)
# rho_L values (pl) from 1 to 1000 on a log scale
rhoL_vec = np.logspace(-1, 2, 100)
# choose 3 payload masses to compare
ml_list = np.logspace(-1,3,5)  # <-- edit as needed
Cmin_mat = np.zeros((len(ml_list), len(rhoL_vec)))
rmin_mat = np.zeros_like(Cmin_mat)
r_lo, r_hi = float(np.min(r)), float(np.max(r))
colors = plt.cm.plasma(np.linspace(0, 1, len(ml_list)))
for j, (ml_j, col) in enumerate(zip(ml_list, colors)):
    for i, rhoL in enumerate(rhoL_vec):
        MIN_ij = optimize.minimize_scalar(
            lambda rr: f(rr, ml_j, rhoL, hl, pf, hf, cw, cs, ct),
            method="bounded",
            bounds=(r_lo, r_hi)
        )
        rmin_mat[j, i] = float(MIN_ij.x)
        Cmin_mat[j, i] = float(MIN_ij.fun)
    plt.plot(rhoL_vec, Cmin_mat[j, :], color=col, linewidth=2,
             label=fr"$m_L$={ml_j:g}")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Minimum Cost $C_{\min}$ versus Payload Density $\rho_L$ (for varying $m_L$)", fontsize=16)
plt.xlabel(r"$\rho_L$ (payload density, pl)", fontsize=14)
plt.ylabel(r"$C_{\min}$", fontsize=14)
plt.grid(True, which='both')
plt.legend()



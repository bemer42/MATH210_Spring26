import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1 problem 2a.  We use the sympy
# package to define symbolic versions of the function and its 
# derivatives to classify the critical points. 

# Discretize the domain with a vector, x
N = int(1e3)
x_vec = np.linspace(-6, 2, N)  

# Create a symbolic function for f(x)
x = sp.Symbol('x')
f = 2/3 * x ** 3 + 4 * x ** 2 - 10 * x + 5/3

# Compute first derivative
fp1 = sp.diff(f, x)

# Compute second derivative
fp2 = sp.diff(f, x, 2)

# Convert sa callable function
f_num = sp.lambdify(x, f, "numpy")
fp1_num = sp.lambdify(x, fp1, "numpy")
fp2_num = sp.lambdify(x, fp2, "numpy")

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp1_num, bracket=(-5-.1, -5+.1), method="brentq")
c_2 = optimize.root_scalar(fp1_num, bracket=(1-.1, 1+.1), method="brentq")

# Compile both roots into an array
c = np.array([c_1.root, c_2.root])

# Second Derivative Test
def classify(c):
    val = float(fp2_num(c))
    
    if val > 0:
        msg = f"x = {c} is a relative minimum (f''(x) = {val})"
        kind = "relative min"
    elif val < 0:
        msg = f"x = {c} is a relative maximum (f''(x) = {val})"
        kind = "relative max"
    else: 
        msg = f"x = {c} is inconclusive (f''(x) = 0)"
        kind = "inconclusive"
    
    print(msg)
    return kind
    
# Print results from Second Derivative Test
c1_kind = classify(c_1.root)
c2_kind = classify(c_2.root)

# Plot the graph of the function
plt.figure(1)
plt.plot(x_vec, f_num(x_vec), 'k-', linewidth=3, label='y = f(x)')
plt.plot(c, f_num(c), 'ro', linewidth=3, label='critical points')
plt.title('Polynomial with Critical Points', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x_vec), np.max(x_vec)])
plt.ylim([np.min(f_num(x_vec))-3, np.max(f_num(x_vec))+3])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws1_2a_plot.eps', format='eps')
plt.savefig('p1ws1_2a_plot.png', format='png')

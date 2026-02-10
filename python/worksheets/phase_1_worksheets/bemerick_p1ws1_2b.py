import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Objective: This script outputs some basic plotting commands 
# associated to Phase 1 - Worksheet 1 problem 2b.  We use the "def"
# syntax to define functions. 

# Discretize the domain with a vector, x
N = int(1e5)
x = np.linspace(-1, 10, N)  

# Create the function in part 1a
def f(x):
    return x ** 2 * np.exp(-x/2)

# Create a general numerical derivative using a centered difference
def fp(x):
    h = 1e-5
    return (f(x+h)-f(x-h)) / 2 / h

# Use built-in solver to calculate critical points
c_1 = optimize.root_scalar(fp, bracket=(0-.1, 0+.1), method="brentq")
c_2 = optimize.root_scalar(fp, bracket=(4-.1, 4+.1), method="brentq")

# Compile both roots into an array
c = np.array([c_1.root, c_2.root])

# Second Derivative Test
def classify(c):
    h = 1e-1
    val_l = float(fp(c-h))
    val_r = float(fp(c+h))
    
    if val_l < 0 and val_r > 0:
        msg = f"x = {c} is a relative minimum (f'(x-h)<0 and f'(x+h)>0)"
        kind = "relative min"
    elif val_l >0 and val_r <0:
        msg = f"x = {c} is a relative maximum (f'(x-h)>0 and f'(x+h)<0)"
        kind = "relative max"
    else: 
        msg = f"x = {c} is a terrace point"
        kind = "saddle"
    
    print(msg)
    return kind

# Print results from First Derivative Test
c1_kind = classify(c_1.root)
c2_kind = classify(c_2.root)

# Plot the graph of the function
plt.figure(1)
plt.plot(x, f(x), 'k-', linewidth=3, label='y = f(x)')
plt.plot(c, f(c), 'ro', linewidth=3, label='critical points')
plt.title('Transcendental Function with Critical Points', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y=f(x)', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(f(x))-1, np.max(f(x))+1])
plt.legend()

# Show the plot:
plt.show()

# Save the file in high quality format as eps and png: 
plt.savefig('p1ws1_2b_plot.eps', format='eps')
plt.savefig('p1ws1_2b_plot.png', format='png')

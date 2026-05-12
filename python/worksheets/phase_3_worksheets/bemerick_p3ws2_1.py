import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from plot_weighted_graph import plot_weighted_graph

# This program solves Phase 3, Worksheet 2, Problem 1.

# Set up nodes for each graph and define Astrong and Aweak
nodes = ["A", "B", "C", "D", "E"]
u, v = nodes.index("A"), nodes.index("B")

Astrong = np.array([
    [0, 4, 2, 0, 1],
    [4, 0, 3, 2, 0],
    [2, 3, 0, 1, 2],
    [0, 2, 1, 0, 5],
    [1, 0, 2, 5, 0]
], dtype=float)

Aweak = np.array([
    [0, 5, 4, 4, 5],
    [5, 0, 1, 0, 1],
    [4, 1, 0, 1, 0],
    [4, 0, 1, 0, 1],
    [5, 1, 0, 1, 0]
], dtype=float)

# Row sums:
s_strong = Astrong.sum(axis=1)
s_weak = Aweak.sum(axis=1)

# Define row sum functions for any alpha and any node:
def s(alpha,ind):
    return s_strong[ind] + alpha * s_weak[ind]

# Determine critical alpha value
alpha_star = root_scalar(lambda alpha: s(alpha,u) - s(alpha,v),bracket = (-10,10))

# Plot strength functions
alpha = np.linspace(0, 1, 300)
plt.plot(alpha, s(alpha,u), label=r"$s(A)$", linewidth=2)
plt.plot(alpha, s(alpha,v), label=r"$s(B)$", linewidth=2)
plt.xlabel(r"$\alpha$")
plt.ylabel("strength")
plt.title("Strengths of nodes A and B vs $\\alpha$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Print critical value
print(alpha_star.root)

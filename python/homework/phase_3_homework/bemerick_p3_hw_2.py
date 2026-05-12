import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import root_scalar
from plot_weighted_graph_2 import plot_weighted_graph_2


# This program solves Phase 3 Homework, Problem 2

nodes = ["A", "B", "C", "D", "E"]
u, v = nodes.index("C"), nodes.index("E")

Astrong = np.array([
    [0, 4, 2, 0, 0],
    [4, 0, 1, 2, 0],
    [2, 1, 0, 3, 0],
    [0, 2, 3, 0, 2],
    [0, 0, 0, 2, 0]
], dtype=float)

Amedium = np.array([
    [0, 0, 0, 2, 0],
    [0, 0, 1, 1, 3],
    [0, 1, 0, 0, 1],
    [2, 1, 0, 0, 0],
    [0, 3, 1, 0, 0]
], dtype=float)

Aweak = np.array([
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 2],
    [0, 0, 0, 0, 1],
    [1, 0, 2, 1, 0]
], dtype=float)

# row sums:
s_strong = Astrong.sum(axis=1)
s_medium = Amedium.sum(axis=1)
s_weak   = Aweak.sum(axis=1)

def s(alpha,u):
    return s_strong[u] + alpha * s_medium[u] + alpha**2 * s_weak[u]

alpha_star = root_scalar(lambda alpha: s(alpha,u) - s(alpha,v),bracket = (0,4))
print(alpha_star.root)

# plot
alpha = np.linspace(0, 3, 300)
plt.plot(alpha, s(alpha,u), 'b', label=r"$s_C(\alpha)$", linewidth=2)
plt.plot(alpha, s(alpha,v), 'r', label=r"$s_E(\alpha)$", linewidth=2)
plt.plot(alpha_star.root, s(alpha_star.root,u), 'mo', label=r"$\alpha^*$", markersize=8)
plt.xlabel(r"$\alpha$")
plt.ylabel("strength")
plt.title("Strengths of nodes C and E vs $\\alpha$")
plt.grid(visible = True, which = 'both', alpha=0.8)
plt.legend()
plt.tight_layout()

plt.savefig('p3hw_figure_1.eps', format='eps', dpi = 300)

# plot of graph
alpha = .5
G, fig, ax, pos = plot_weighted_graph_2(nodes, Astrong, Amedium, Aweak, alpha)

plt.savefig('p3hw_figure_2.eps', format='eps', dpi = 300)

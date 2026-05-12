import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

# This program solves Phase 3, Worksheet 4, Problem 2


# Build graph
V = list(range(1,11))
E = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6), 
         (5, 7), (6, 7), (6, 8), (7, 8), (8, 9), (8, 10), (9, 10)]

G = nx.Graph()
G.add_edges_from(E)

# Plot graph
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

# Define m and calculate degrees
m = G.number_of_edges()
deg = dict(G.degree())

print(f"Number of edges: m = {m}")
print("Degrees:")
for i in V:
    print(f"  k_{i} = {deg[i]}")

# Compute modularity for each community
C1 = {1, 2, 3, 4}
C2 = {5, 6, 7, 8, 9, 10}
partition = [C1, C2]

# Compute modularity using greedy algorithm
C_greedy = list(greedy_modularity_communities(G))
Q_greedy = modularity(G, C_greedy)

print("Greedy modularity communities:")
for c in C_greedy:
    print(" ", c)
print(f"Modularity of greedy partition: Q = {Q_greedy:.6f}")


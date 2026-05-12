import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# This program solves Phase 3, Worksheet 3, Problem 2

# Create the graph
G = nx.Graph()
edges = [(1,2), (1,3), (1,4)]
G.add_edges_from(edges)

# Plot graph
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

# Compute centralities
cc = nx.closeness_centrality(G)
bc = nx.betweenness_centrality(G)
ec = nx.eigenvector_centrality(G)

print("Closeness Centrality:")
print(cc)
print("Betweenness Centrality:")
print(bc)
print("Eigenvector Centrality:")
print(ec)

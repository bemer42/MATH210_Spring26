import networkx as nx
import matplotlib.pyplot as plt

# This program solves Phase 3 Homework, Problem 3

# Create the graph
# Build graph
V = list("ABCDEF")
E = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), 
     ("D", "E"), ("D", "F"), ("E", "F")]

G = nx.Graph()
G.add_nodes_from(V)
G.add_edges_from(E)

# Plot graph
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()

plt.savefig('p3hw_figure_3.eps', format='eps', dpi = 300)


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


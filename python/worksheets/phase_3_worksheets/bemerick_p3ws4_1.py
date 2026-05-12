import matplotlib.pyplot as plt
import networkx as nx

# This program solves Phase 3, Worksheet 4, Problem 1

# Build graph
V = list("ABCDEFGH")
E = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), 
     ("D", "F"), ("E", "F"), ("F", "G"), ("G", "H"), ("F", "H")]

G = nx.Graph()
G.add_nodes_from(V)
G.add_edges_from(E)

# Plot graph
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

# Edge betweenness centrality
ebc = nx.edge_betweenness_centrality(G)
print(ebc)

# Remove bridge-like edge C-D and find connected components
G_removed = G.copy()
G_removed.remove_edge("C", "D")

# Plot removed graph
pos = nx.spring_layout(G_removed, seed=1)
nx.draw_networkx(G_removed, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()
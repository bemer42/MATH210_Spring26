import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# This program solves Phase 3, Worksheet 3, Problem 1

# Define edge set and create the graph two ways
E = [("A","B",3), ("B","C",2), ("A","D",4), ("B","E",5),
     ("C","F",1), ("D","E",2), ("E","F",3), ("B","D",2)]

G = nx.Graph()
G.add_weighted_edges_from(E, weight="weight")

# Plot with weights
pos = nx.spring_layout(G, seed=1)
nx.draw(G, pos, with_labels=True)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# Compute strengths from Adjacency matrix
A  = nx.to_numpy_array(G, nodelist=["A", "B", "C", "D", "E", "F"], weight = "weight")  
st = np.sum(A,axis=0)
print("Strengths")
print(st)

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


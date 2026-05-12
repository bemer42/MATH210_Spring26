import networkx as nx
import matplotlib.pyplot as plt

# This program solves the systems presented in Phase 3, Worksheet 1, 
# Problem 3.
    
#################################
# Graph from edge list with weights
#################################
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

#################################
# Weight of a given path
#################################
path = ["A", "D", "E", "F"]
w = nx.path_weight(G, path, weight="weight")

#################################
# Strength of one node
#################################
strength_B = G.degree("B", weight="weight")

#################################
# Strength of all nodes
#################################
strength = dict(G.degree(weight="weight"))

#################################
# Shortest path
#################################
spl = nx.shortest_path_length(G, "A", "F", weight="weight")
sp = nx.shortest_path(G, "A", "F", weight="weight")


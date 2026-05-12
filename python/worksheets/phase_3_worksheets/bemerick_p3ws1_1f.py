import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# This program solves Phase 3, Worksheet 1, Problem 1f.
    
#################################
# Graphs from edge list
#################################

E = [("A","B"), ("B","C"), ("A","D"), ("B","E"),
     ("C","F"), ("D","E"), ("E","F"), ("B","D")]

G1 = nx.Graph()
G1.add_edges_from(E)

pos = nx.spring_layout(G1, seed=1)
nx.draw_networkx(G1, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

#################################
# Graphs from adjacency list
#################################
adj = {
    "A": ["B", "D"],
    "B": ["A", "C", "D", "E"],
    "C": ["B", "F"],
    "D": ["A", "B", "E"],
    "E": ["B", "D", "F"],
    "F": ["C", "E"]
}

G2 = nx.Graph(adj)

pos = nx.spring_layout(G2, seed=1)
nx.draw_networkx(G2, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

#################################
# Graphs from adjacency matrix
#################################
A = np.array([[0,1,0,1,0,0],
              [1,0,1,1,1,0],
              [0,1,0,0,0,1],
              [1,1,0,0,1,0],
              [0,1,0,1,0,1],
              [0,0,1,0,1,0]], dtype=int)

G3 = nx.from_numpy_array(A)     
nodes = ["A","B","C","D","E","F"]
G3 = nx.relabel_nodes(G3, dict(enumerate(nodes)))

pos = nx.spring_layout(G3, seed=1)
nx.draw_networkx(G3, pos, with_labels=True, node_size=600,
                 width=1.0, edge_color="#444444")
plt.axis("off")
plt.tight_layout()
plt.show()

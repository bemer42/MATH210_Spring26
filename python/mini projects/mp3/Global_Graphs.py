import numpy as np
import pandas as pd
import networkx as nx
from plot_weighted_graph_2 import plot_weighted_graph_2
import matplotlib.pyplot as plt

# Load in Data
df = pd.read_excel("Weighted_Edges_2006_2024.xlsx",
                   usecols=["from","to","majmaj weight","majmin weight","minmin weight"],
                   sheet_name=0)

# Create numpy and python lists from data frame
u = df["from"].astype(str).to_numpy()
v = df["to"].astype(str).to_numpy()
w_mm = df["majmaj weight"].astype(float).to_numpy()
w_mn = df["majmin weight"].astype(float).to_numpy()
w_nn = df["minmin weight"].astype(float).to_numpy()


# Define edge lists
edges_mm = list(zip(u, v, w_mm))
edges_mn = list(zip(u, v, w_mn))
edges_nn = list(zip(u, v, w_nn))

# Build weighted graphs
G_mm = nx.Graph()
G_mn = nx.Graph()
G_nn = nx.Graph()
G_mm.add_weighted_edges_from(edges_mm, weight="weight")
G_mn.add_weighted_edges_from(edges_mn, weight="weight")
G_nn.add_weighted_edges_from(edges_nn, weight="weight")

# Build nodes and adjacency matrices
nodes = sorted(set(G_mm.nodes()) | set(G_mn.nodes()) | set(G_nn.nodes()))
A_mm = nx.to_numpy_array(G_mm, nodelist=nodes, weight="weight")
A_mn = nx.to_numpy_array(G_mn, nodelist=nodes, weight="weight")
A_nn = nx.to_numpy_array(G_nn, nodelist=nodes, weight="weight")

print("Nodes:", len(nodes))
print("Matrix shapes:", A_mm.shape, A_mn.shape, A_nn.shape)

# Define A(alpha)
alpha = 0.5
A_alpha = A_mm + alpha * A_mn + (alpha**2) * A_nn
A_alpha = A_alpha + 0.0*(A_alpha==0)
print("Built A(alpha) with alpha =", alpha)

G, fig, ax, pos = plot_weighted_graph_2(nodes, A_mm, A_mn, A_nn, alpha, 0)
plt.show()

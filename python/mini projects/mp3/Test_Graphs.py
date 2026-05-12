import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from plot_weighted_graph_2 import plot_weighted_graph_2 as pg
import pickle
import community as cl
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

# LOAD
with open('A_by_year.pkl', 'rb') as f:
    Amm_st, Amn_st, Ann_st = pickle.load(f)

nodes = Amm_st["nodes"]
node_set = set(nodes)

def A(alpha,t):
    A_alpha = Amm_st["A_by_year"][t] + alpha * Amn_st["A_by_year"][t] + alpha**2 * Ann_st["A_by_year"][t]
    G_alpha = nx.from_numpy_array(A_alpha)
    return A_alpha, G_alpha

alpha = 0.5

Aa, Ga = A(alpha,2020)

# for t in range(2006,2025):
#     G, fig, ax, pos = pg(nodes, A(alpha,t), alpha, t)
#     plt.show()


# comms = greedy_modularity_communities(nx.from_numpy_array(A(.5,2020)), weight="weight", ) 
# comms = list(comms)  
# comms_labels = [ {nodes[i] for i in c} for c in comms ]
# comms

ec = nx.betweenness_centrality_numpy(Ga)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_weighted_graph_2(nodes, Astrong, Amedium, Aweak, alpha, *, seed=1):
    # Build A(alpha)
    A = Astrong + alpha * Amedium + alpha**2 * Aweak

    # Build complete weighted graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(A[i, j])
            if not np.isfinite(w):
                w = 0.0
            G.add_edge(nodes[i], nodes[j], weight=w)

    # Prepare edge styling
    edgelist = list(G.edges())
    weights = np.array([G[u][v].get("weight", 0.0) for (u, v) in edgelist], dtype=float)

    w_min, w_max = float(weights.min()), float(weights.max())
    den = (w_max - w_min) if (w_max - w_min) != 0 else 1.0
    w_norm = (weights - w_min) / den

    min_width, max_width = 0.8, 6.0
    widths = min_width + w_norm * (max_width - min_width)

    min_alpha, max_alpha = 0.35, 1.0  # bumped up so edges are visible
    edge_alphas = min_alpha + w_norm * (max_alpha - min_alpha)
    edge_colors = [(0, 0, 0, float(a)) for a in edge_alphas]

    # Layout + draw
    # pos = nx.spring_layout(G, seed=seed)
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color="white", edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist, width=list(widths), edge_color=edge_colors)

    ax.set_title(rf"Graph for $A(\alpha)$ with $\alpha={alpha}$")
    ax.axis("off")
    fig.tight_layout()

    return G, fig, ax, pos
    





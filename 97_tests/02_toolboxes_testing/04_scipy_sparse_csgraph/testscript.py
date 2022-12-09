import sys, time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

NNODES = 20
SEED = 12345
NETTYPE = 'fullyconnected'
NETTYPE = 'adhoc'

def main():
    """Main function"""

    # Generate original graph
    if NETTYPE == 'fullyconnected':
        OGnx = nx.complete_graph(NNODES)
    elif NETTYPE == 'adhoc':
        OGnx = nx.random_geometric_graph(NNODES, radius=0.5, seed=SEED)
    else:
        raise ValueError(f'Network type "{NETTYPE}" not understood.')
    # Generate node positions
    pos = nx.spring_layout(OGnx, seed=SEED)
    pos = nx.kamada_kawai_layout(OGnx)
    # Add edge weights
    for e in OGnx.edges():
        # Compute weight based on distance
        weight = np.linalg.norm(pos[e[0]] - pos[e[1]])
        OGnx[e[0]][e[1]]['weight'] = weight

    # Compute MSTs (using the three available algorithms)
    algos = ['kruskal', 'prim', 'boruvka']
    MSTs, timings = get_msts(OGnx, algos)

    # Plot
    plot_msts(OGnx, MSTs, pos, timings, nodeSize=100)
    plt.show()

    stop = 1

def get_msts(G, algos):
    """Compute MSTs using the listed algorithms."""
    timings = dict([(algo, None) for algo in algos])
    msts = dict([(algo, None) for algo in algos])
    for algo in algos:
        t0 = time.time()
        msts[algo] = nx.minimum_spanning_tree(
            G,
            weight='weight',
            algorithm=algo
        )
        t1 = time.time()
        timings[algo] = t1 - t0
    return msts, timings


def plot_msts(oggraph, msts: dict, pos, timings, nodeSize=100):
    """Plot MST-formation algorithms' output."""
    algos = list(msts.keys())
    nAlgos = len(algos)

    # Generate figure
    fig, axes = plt.subplots(1, nAlgos + 1)
    fig.set_size_inches(9/4 * (nAlgos + 1), 3)
    nx.draw(oggraph, pos=pos, ax=axes[0], node_size=nodeSize)
    axes[0].set_title(f'Original graph (type: "{NETTYPE}")')
    for ii, algo in enumerate(algos):
        nx.draw(msts[algo], pos=pos, ax=axes[ii + 1], edge_color='r', width=2.0, node_size=nodeSize)
        axes[ii + 1].set_title(f"{algo}'s MST ({np.round(timings[algo], 4)} s)")


if __name__ == '__main__':
    sys.exit(main())
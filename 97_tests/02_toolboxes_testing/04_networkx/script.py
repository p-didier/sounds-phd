import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

NNODES = 10
CONNECTIONDIST = 2  # maximum distance for two nodes to be able to communicate
SQUAREROOMDIM = 5
SEED = 12335

np.random.seed(SEED)

def main():
    
    # Generate random 2D node layout
    posNodes = np.random.uniform(0, SQUAREROOMDIM, size=(NNODES, 2))
    # Compute distance matrix
    distNodes = distance_matrix(posNodes, posNodes)

    # Generate connections matrix
    connMat = np.zeros_like(distNodes)
    connMat[distNodes <= CONNECTIONDIST] = 1
    # Make symmetric (https://stackoverflow.com/a/28904854)
    # and fill diagonal with 1's
    connMat = np.maximum(connMat, connMat.T)
    np.fill_diagonal(connMat, 1)
    # Create adjacency dictionary
    adjdict = _get_adjacency_dict(connMat)
    # List leaf nodes
    leaves =  np.array([ii for ii in range(NNODES) if len(adjdict[ii]) == 1])

    # Generate directed graph
    myG = nx.DiGraph(adjdict) 
    # Get layout dictionary (for plotting correctly)
    layout = _get_layout_dict(posNodes)

    # Generate random source position
    sourcePos = np.random.uniform(0, SQUAREROOMDIM, size=(1, 2))
    SNRs = 1 / distance_matrix(posNodes, sourcePos)  # distance (SNR-like)

    # Derive edge colors and directions
    edges = np.array(myG.edges)
    edgecolors = ['k'] * len(edges)
    for k in range(NNODES):
        edgeidx = np.array([h for h in range(len(edges)) if k in edges[h]])
        # Order indices
        firstNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][0] == k]
        secondNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][1] == k]
        reorderingIdx = np.concatenate((firstNodeidx, secondNodeidx))
        edgeidx = edgeidx[reorderingIdx]

        # Non-leaf nodes
        if k not in leaves:
            # List neighbors
            stemNeighs = np.array(adjdict[k])
            for neighIdx, q in enumerate(stemNeighs):
                if q not in leaves:
                    if SNRs[k] <= SNRs[q]:
                        passiveEdgeIdx = edgeidx[neighIdx]
                        activeEdgeIdx = edgeidx[neighIdx + len(edgeidx) // 2]
                    else:
                        passiveEdgeIdx = edgeidx[neighIdx + len(edgeidx) // 2]
                        activeEdgeIdx = edgeidx[neighIdx]
                    edgecolors[passiveEdgeIdx] = (0,0,0,0)
                    edgecolors[activeEdgeIdx] = 'r'
        # Leaf nodes
        else:
            for jj in range(len(edgeidx) // 2):
                # vvv make outgoing arrow transparent
                edgecolors[edgeidx[len(edgeidx) - jj - 1]] = 'g'
                # vvv make incoming arrow green
                edgecolors[edgeidx[jj]] = (0,0,0,0)
            

    # # Loop through nodes attached to leaves
    # stemsReached = []
    # for ii in range(len(leaves)):
    #     stemNode = adjdict[leaves[ii]][0]
    #     neighbors = np.array(adjdict[stemNode])
    #     # vvv only consider non-leaf neighbors
    #     neighbors = neighbors[neighbors != leaves[ii]]
    #     if all(SNRs[stemNode] > SNRs[neighbors]):  # best node among neighbors
    #         idx = np.array([k for k in range(len(edges))
    #             if stemNode in edges[k]])
    #         edgecolors[idx] = 'r'
    #     stemsReached.append(stemNode)
    # edgecolors[leaves] = 'g'  # leaf nodes infer
    # # Loop through remaining nodes
    # stemNodes = [ii for ii in np.arange(NNODES)\
    #     if ii not in leaves and ii not in stemsReached]
    # for ii in range(len(stemNodes)):
    #     currStem = stemNodes[ii]
    #     # vvv only consider neighbors that have not been considered yet
    #     neighbors = np.array([_ for _ in adjdict[stemNode] if _ in stemNodes])

        


    # Plot
    axes = plot_network(myG, layout, edgecolors)
    axes.scatter(sourcePos[0, 0], sourcePos[0, 1], s=100, c='r')
    plt.tight_layout()	
    plt.show()

    stop = 1


def plot_network(G, layout, edge_color=None):
    """Plot network (inc. topology)."""
    if edge_color is None:
        edge_color = ['k'] * len(G.edges())  # by default, all edges are black

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(3.5, 3.5)
    nx.draw(
        G,
        layout,
        with_labels=True,
        ax=axes,
        edge_color=edge_color,
        width=2)
    axes.grid()
    axes.set_axisbelow(True)
    # vvv counteract innerworkings of `nx.draw()`
    axes.set_axis_on()
    axes.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )
    axes.set_xlabel('$x$ [m]')
    axes.set_ylabel('$y$ [m]')
    

    return axes


def _get_adjacency_dict(T):
    """Get adjacency dictionary (for use with `networkx.Graph()` class)
    from connection matrix `T`."""
    d = dict()
    for ii in range(T.shape[0]):
        d[ii] = tuple([jj for jj in range(len(T[ii, :])) if jj != ii and T[ii, jj] == 1])
    return d


def _get_layout_dict(pos):
    """Get layout dictionary (for use in `networkx.draw()` function)
    from node coordinates in `pos`."""
    return {ii : pos[ii, :] for ii in range(pos.shape[0])}

if __name__=='__main__':
    sys.exit(main())
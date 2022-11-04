import networkx as nx
import numpy as np
import sys, copy
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

NNODES = 10
CONNECTIONDIST = 2  # maximum distance for two nodes to be able to communicate
SQUAREROOMDIM = 5
SEED = 12335
COLORDEFAULTLINK = 'k'
COLORLEAFLINK = 'g'
COLORBESTSNRLINK = 'r'
COLORINFERENCELINK = 'b'

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

    # Generate directed graph
    myG = nx.DiGraph(adjdict) 
    # Get layout dictionary (for plotting correctly)
    layout = _get_layout_dict(posNodes)

    # Generate random source position
    sourcePos = np.random.uniform(0, SQUAREROOMDIM, size=(1, 2))
    SNRs = 1 / distance_matrix(posNodes, sourcePos)  # distance (SNR-like)

    # Derive edge colors and directions
    edgecolors = get_graph_colors(myG, adjdict, SNRs)

    # Create new graph based on remaining edges
    remainingEdges = np.array(myG.edges)[np.array(edgecolors) == COLORDEFAULTLINK]
    remainingEdges = [tuple(ii) for ii in remainingEdges]
    myGnew = nx.DiGraph(remainingEdges)
    adjdictnew = {ii : tuple(myGnew.adj[ii]) for ii in list(myGnew.nodes)}
    edgecolors2 = get_graph_colors(myGnew, adjdictnew, SNRs[list(myGnew.nodes)])

    # Plot
    axes = plot_network(myG, layout, edgecolors)
    axes.scatter(sourcePos[0, 0], sourcePos[0, 1], s=100, c='r')
    plt.tight_layout()	
    plt.show()

    stop = 1


def get_graph_colors(myG, adjdict, SNRs):
    """Compute the edge colors of a directed graph according to SRO
    propagation (and redundancy reduction) rules."""
    
    edges = np.array(myG.edges)
    edgecolors = [COLORDEFAULTLINK] * len(edges)

    nodes = list(myG.nodes)

    # List leaf nodes
    leaves =  np.array([ii for ii in nodes if len(adjdict[ii]) == 1])

    for k in nodes:
        edgeidx = np.array([h for h in range(len(edges)) if k in edges[h]])
        if len(edgeidx) == 0:
            continue
        # Order indices
        firstNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][0] == k]
        secondNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][1] == k]
        reorderingIdx = np.concatenate((firstNodeidx, secondNodeidx))
        edgeidx = edgeidx[reorderingIdx]

        # Non-leaf nodes with best SNRs
        bestSNRnodes = []
        if k not in leaves:
            # List neighbors
            stemNeighs = np.array(adjdict[k])
            if all(SNRs[k] >= SNRs[stemNeighs]):
                bestSNRnodes.append(k)
                passiveEdgeIdx = edgeidx[np.arange(len(stemNeighs)) + len(edgeidx) // 2]
                activeEdgeIdx = edgeidx[np.arange(len(stemNeighs))]
                for ii in range(len(passiveEdgeIdx)):
                    edgecolors[passiveEdgeIdx[ii]] = (0,0,0,0)
                for ii in range(len(activeEdgeIdx)):
                    edgecolors[activeEdgeIdx[ii]] = COLORBESTSNRLINK
        # Leaf nodes
        else:
            for jj in range(len(edgeidx) // 2):
                # vvv make outgoing arrow transparent
                edgecolors[edgeidx[len(edgeidx) - jj - 1]] = COLORLEAFLINK
                # vvv make incoming arrow green
                edgecolors[edgeidx[jj]] = (0,0,0,0)
        
    # Consider other non-best-SNR nodes
    for k in nodes:
        if k not in bestSNRnodes and k not in leaves:
            # List neighbors
            neighbors = np.array(adjdict[k])
            # For each neighbor, check adjacency
            for ii in range(len(neighbors)):
                if neighbors[ii] in bestSNRnodes or neighbors[ii] in leaves:
                    pass
                else:
                    # Check if `k` and `neighbors[ii]` have common neighbors.
                    commonNeighs = _common_neighbor(k, neighbors[ii], adjdict)
                    # If they do, check whether they know their respective 
                    # SROs with those common neighbors.
                    for jj in range(len(commonNeighs)):
                        idx1 = _eidx(k, commonNeighs[jj], edges)
                        idx2 = _eidx(neighbors[ii], commonNeighs[jj], edges)
                        if edgecolors[idx1] not in [COLORDEFAULTLINK, COLORINFERENCELINK]\
                            and edgecolors[idx2] not in [COLORDEFAULTLINK, COLORINFERENCELINK]:
                            # vvv if they do, make both arrows blue
                            idx11 = _eidx(k, neighbors[ii], edges)
                            idx12 = _eidx(neighbors[ii], k, edges)
                            edgecolors[idx11] = COLORINFERENCELINK
                            edgecolors[idx12] = COLORINFERENCELINK
                            break

    return edgecolors


def _eidx(k,q,edges):
    """Returns boolean array to select the index of (directed) edge
    from `k` to `q`."""
    allidx = np.arange(len(edges))
    myidx = allidx[np.logical_and(edges[:, 0] == k, edges[:, 1] == q)]
    if len(myidx) > 1:
        raise ValueError('Issue here.')
    elif len(myidx) == 0:
        return None
    return myidx[0]


def _common_neighbor(k, q, adj):
    """Checks whether node `k` and `q` have a common neighbor."""
    neighs_k, neighs_q = adj[k], adj[q]
    commonNeighbors = []
    for ii in range(len(neighs_k)):
        if neighs_k[ii] != q and neighs_k[ii] in neighs_q:
            commonNeighbors.append(neighs_k[ii])
    return commonNeighbors


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
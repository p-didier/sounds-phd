import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

NNODES = 10
CONNECTIONDIST = 2  # maximum distance for two nodes to be able to communicate
SQUAREROOMDIM = 5

def main():
    
    # Generate random 2D node layout
    pos = np.random.uniform(0, SQUAREROOMDIM, size=(NNODES, 2))
    # Compute distance matrix
    dists = distance_matrix(pos, pos)
    # Generate connections matrix
    Tmat = np.zeros_like(dists)
    Tmat[dists <= CONNECTIONDIST] = 1
    # Make symmetric (https://stackoverflow.com/a/28904854) and fill diagonal with 1's
    TmatSym = np.maximum(Tmat, Tmat.T)
    np.fill_diagonal(TmatSym, 1)
    # Create adjacency dictionary
    adjdict = _get_adjacency_dict(TmatSym)

    # Generate graph
    myCompleteGraph = nx.Graph(adjdict) 
    # Get layout dictionary (for plotting correctly)
    layout = _get_layout_dict(pos)

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(3.5, 3.5)
    nx.draw(myCompleteGraph, layout, with_labels=True, ax=axes)
    axes.grid()
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
    plt.tight_layout()	
    plt.show()


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
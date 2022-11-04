import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from pathlib import Path
from dataclasses import dataclass

NNODES = 20
MINNODEDIST = 0.75   # minimum distance between nodes
MINDISTTOWALLS = 0.25   # minimum node-wall distance
CONNECTIONDIST = 4  # maximum distance for two nodes to be able to communicate
SQUAREROOMDIM = 10
SEED = 12335
COLORDEFAULTLINK = 'k'
COLORBESTSNRLINK = 'r'
COLORINFERENCELINK = 'b'
OUTPUTFOLDER = f'{Path(__file__).parent}/out/{NNODES}nodes'

NWASNS = 50

np.random.seed(SEED)

@dataclass
class ASCWASN():
    posNodes: np.ndarray = np.array([])     # node positions
    sourcePos: np.ndarray = np.array([])    # source position
    dim: int = 2    # problem dimensionality (2 or 3)

    def __post_init__(self):
        if self.dim not in [2, 3]:
            raise ValueError(f'Incorrect problem dimensionality ({self.dim}-D).')
        # Distance (SNR-like)
        self.SNRs = 1 / distance_matrix(self.posNodes, self.sourcePos)
        # Layout
        self.layout = _get_layout_dict(self.posNodes)
        
    def init_wasn(self):
        # Inter-node distance matrix
        distNodes = distance_matrix(self.posNodes, self.posNodes)
        # Generate connections matrix
        connMat = np.zeros_like(distNodes)
        connMat[distNodes <= CONNECTIONDIST] = 1
        # Make symmetric (https://stackoverflow.com/a/28904854)
        # and fill diagonal with 1's
        connMat = np.maximum(connMat, connMat.T)
        np.fill_diagonal(connMat, 1)
        # Create adjacency dictionary
        self.adjdict = _get_adjacency_dict(connMat)
        # Generate directed graph
        self.wasn = nx.DiGraph(self.adjdict)

    def _get_graph_colors(self):
        """Derive edge colors and directions"""
        self.edgecolors = get_graph_colors(self.wasn, self.adjdict, self.SNRs)

    def plotwasn(self):
        fig, axes = plot_network(self.wasn, self.layout, self.edgecolors)
        axes.scatter(
            self.sourcePos[0, 0],
            self.sourcePos[0, 1],
            s=70,
            c='m',
            label='Source')
        axes.legend()
        plt.tight_layout()
        return fig


def main():
    
    reduction = []
    for idxmain in range(NWASNS):
        print(f"Computing network topology and SRO inference pattern {idxmain+1}/{NWASNS}...")

        # Generate WASN
        data = ASCWASN(
            posNodes=generate_points_with_min_distance(
                NNODES, SQUAREROOMDIM, MINNODEDIST
            ),
            sourcePos=np.random.uniform(0, SQUAREROOMDIM, size=(1, 2)),
            dim=2
        )
        # Initialise WASN
        data.init_wasn()
        # Derive edge colors and directions
        data._get_graph_colors()

        # Compute reduction in computations
        reduction.append([
            len(data.edgecolors),
            1 - sum(np.array(data.edgecolors, dtype=object)\
                == COLORBESTSNRLINK) / len(data.edgecolors)
        ])

        # Plot
        fig = data.plotwasn()
        if not Path(OUTPUTFOLDER).is_dir():
            Path(OUTPUTFOLDER).mkdir()
        fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}.png')
        fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}.pdf')
        plt.close(fig)
        
    # Sort and plot reduction
    reduction = np.sort(np.array(reduction, dtype=float), axis=0)
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(5.5, 3.5)
    axes.scatter(reduction[:, 0], reduction[:, 1] * 100)
    coef = np.polyfit(reduction[:, 0], reduction[:, 1] * 100, 1)
    poly1d_fn = np.poly1d(coef) 
    x = np.arange(np.amin(reduction[:, 0]), np.amax(reduction[:, 0]))
    axes.plot(x, poly1d_fn(x), 'k--')
    axes.text(
        x=np.amin(reduction[:, 0]) * 1.01, y=np.amax(reduction[:, 1] * 100) * 0.95,
        s=f'Average rate of improvement:\n{np.round(coef[0], 3)} fewer SRO estimations per new link in the WASN.'
    )
    axes.grid()
    axes.set_xlabel('# of inter-node links in the WASN')
    axes.set_ylabel('Reduction in # of SRO estimations [%]')
    axes.set_title(f'{NWASNS} randomly generated WASNs composed of {NNODES} nodes')
    plt.tight_layout()
    fig.savefig(f'{OUTPUTFOLDER}/global.png')
    fig.savefig(f'{OUTPUTFOLDER}/global.pdf')
    plt.show()

    stop = 1


def get_graph_colors(myG, adjdict, SNRs):
    """Compute the edge colors of a directed graph according to SRO
    propagation (and redundancy reduction) rules."""
    
    edges = np.array(myG.edges)
    nodes = list(myG.nodes)
    edgecolors = [COLORDEFAULTLINK] * len(edges)

    for k in nodes:
        edgeidx = np.array([h for h in range(len(edges)) if k in edges[h]])
        if len(edgeidx) == 0:
            continue
        # Order indices
        edgeidx = _order_edge_idx(edgeidx, edges, k)
        # Nodes with best SNRs
        bestSNRnodes = []
        # List neighbors
        stemNeighs = np.array(adjdict[k])
        if all(SNRs[k] >= SNRs[stemNeighs]):
            bestSNRnodes.append(k)
            passiveEdgeIdx = edgeidx[np.arange(len(stemNeighs)) + len(edgeidx) // 2]
            activeEdgeIdx = edgeidx[np.arange(len(stemNeighs))]
            for ii in range(len(passiveEdgeIdx)):
                if edgecolors[passiveEdgeIdx[ii]] == COLORDEFAULTLINK:
                    edgecolors[passiveEdgeIdx[ii]] = (0,0,0,0)
            for ii in range(len(activeEdgeIdx)):
                if edgecolors[activeEdgeIdx[ii]] == COLORDEFAULTLINK:
                    edgecolors[activeEdgeIdx[ii]] = COLORBESTSNRLINK
        
    # Consider other non-best-SNR nodes
    edgecolors = _non_best_snr_node(
        nodes, bestSNRnodes, adjdict, edges, edgecolors
    )

    # Repeat process until entire network is covered
    while COLORDEFAULTLINK in edgecolors:
        # Get remaining edges and nodes
        remainingEdges = edges[
            np.array(edgecolors, dtype=object) == COLORDEFAULTLINK
        ]
        remainingNodes = np.unique(remainingEdges)
        # Repeat the process - derive best SNR nodes
        bestSNRnodes = []
        for k in remainingNodes:
            edgeidx = np.array([h for h in range(len(edges)) if k in edges[h]])
            # Order indices
            edgeidx = _order_edge_idx(edgeidx, edges, k)

            # List neighbors
            stemNeighs = np.array([ii for ii in adjdict[k] if ii in remainingNodes])
            if all(SNRs[k] >= SNRs[stemNeighs]):
                bestSNRnodes.append(k)
                passiveEdgeIdx = edgeidx[np.arange(len(adjdict[k])) + len(edgeidx) // 2]
                activeEdgeIdx = edgeidx[np.arange(len(adjdict[k]))]
                for ii in range(len(passiveEdgeIdx)):
                    if edgecolors[passiveEdgeIdx[ii]] == COLORDEFAULTLINK:
                        edgecolors[passiveEdgeIdx[ii]] = (0,0,0,0)
                for ii in range(len(activeEdgeIdx)):
                    if edgecolors[activeEdgeIdx[ii]] == COLORDEFAULTLINK:
                        edgecolors[activeEdgeIdx[ii]] = COLORBESTSNRLINK

        # Consider other non-best-SNR nodes
        edgecolors = _non_best_snr_node(
            remainingNodes, bestSNRnodes, adjdict, edges, edgecolors
        )

    return edgecolors

def _non_best_snr_node(nodes, bestSNRnodes, adjdict, edges, edgecolors):
    """Update edge colors for non-best SNR nodes: two-nodes-to-one SRO
    inference."""
    for k in nodes:
        if k not in bestSNRnodes:
            # List neighbors
            neighbors = np.array(adjdict[k])
            # For each neighbor, check adjacency
            for ii in range(len(neighbors)):
                if neighbors[ii] in bestSNRnodes:
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


def _order_edge_idx(edgeidx, edges, k):
    """Order edgeidx variable so that all outwards going edges are listed
    first, then all inwards edges are listed."""
    firstNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][0] == k]
    secondNodeidx = [ii for ii in range(len(edgeidx)) if edges[edgeidx[ii]][1] == k]
    reorderingIdx = np.concatenate((firstNodeidx, secondNodeidx))
    return edgeidx[reorderingIdx]


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
    fig.set_size_inches(5.5, 5.5)
    nx.draw(
        G,
        layout,
        with_labels=True,
        ax=axes,
        edge_color=edge_color,
        width=1.5)
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
    axes.set_xlim([0, SQUAREROOMDIM])
    axes.set_ylim([0, SQUAREROOMDIM])
    ti = f'Red: computes SRO (see arrow); Blue: infers SRO\n{sum(np.array(edge_color, dtype=object) == COLORBESTSNRLINK)} SRO estimations (vs. {len(edge_color)} links) -- {int((1 - sum(np.array(edge_color, dtype=object) == COLORBESTSNRLINK) / len(edge_color)) * 100)}% reduction.)'
    axes.set_title(ti)
    
    return fig, axes


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


def generate_points_with_min_distance(n, shape, min_dist):
    """Generates a set of `n` points within a `shape x shape` room, 
    with a minimum distance `min_dist` between each other."""
    
    coords = []
    while len(coords) < n:
        # Generate new random coordinates
        randCoords = np.random.uniform(
            0, shape - 2 * MINDISTTOWALLS, size=(2,)
        ) + MINDISTTOWALLS
        # Check distances
        addit = True
        for ii in range(len(coords)):
            if len(coords) > 0:
                if _dist(coords[ii], randCoords) < min_dist:
                    addit = False
                    break
            else:
                break
        if addit:
            coords.append(randCoords)

    return np.array(coords)

def _dist(x, y):
    """Compute Euclidean distance."""
    return np.sqrt(sum((x - y) ** 2))


if __name__=='__main__':
    sys.exit(main())
import networkx as nx
import numpy as np
import sys, copy
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from pathlib import Path
from dataclasses import dataclass

# NNODES = [10, 15, 20]
NNODES = [10]
MINNODEDIST = 0.75   # minimum distance between nodes
MINDISTTOWALLS = 0.25   # minimum node-wall distance
SQUAREROOMDIM = 10
SEED = 12335
CONNECDIST = 4
COLORDEFAULTLINK = '#595959'
COLORBESTSNRLINK = 'r'
COLORINFERENCELINK = 'b'
OUTPUTFOLDER = f'{Path(__file__).parent}/out/{NNODES}nodes'

NWASNS = 10
SAVEINDIVWASNPLOTS = True

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
        
    def init_wasn(self, connecDist):
        # Inter-node distance matrix
        distNodes = distance_matrix(self.posNodes, self.posNodes)
        # Generate connections matrix
        connMat = np.zeros_like(distNodes)
        connMat[distNodes <= connecDist] = 1
        # Make symmetric (https://stackoverflow.com/a/28904854)
        # and fill diagonal with 1's
        connMat = np.maximum(connMat, connMat.T)
        np.fill_diagonal(connMat, 1)
        # Create adjacency dictionary
        self.adjdict = _get_adjacency_dict(connMat)
        # Generate directed graph
        self.wasn = nx.Graph(self.adjdict)

    def get_mst(self):
        """Compute minimum spanning tree (MST)."""
        # Get node positions 
        nodesPos = dict(
            [(k, self.posNodes[k, :]) for k in range(self.posNodes.shape[0])]
        )
        
        # Add edge weights based on inter-node distance # TODO: is that a correct approach? TODO:
        for e in self.wasn.edges():
            weight = np.linalg.norm(nodesPos[e[0]] - nodesPos[e[1]])
            self.wasn[e[0]][e[1]]['weight'] = weight
        
        # Convert to minimum spanning tree
        self.wasn = nx.minimum_spanning_tree(
            self.wasn,
            weight='weight'
        )


    def _get_graph_colors(self):
        """Derive edge colors and directions (step-by-step while forming
        the `edgecolors` object in `get_graph_colors()`."""
        self.edgecolors = get_graph_colors(self.wasn, self.adjdict, self.SNRs)

    def plotwasn(self):
        fig, _ = plot_network(
            self.wasn,
            self.layout
        )
        plt.tight_layout()
        return fig


def main():
    """MAIN"""

    for idxNodes in range(len(NNODES)):
        for idxmain in range(NWASNS):
            print(f"Computing network topology and SRO inference pattern {idxmain+1}/{NWASNS}...")

            # Generate WASN
            data = ASCWASN(
                posNodes=generate_points_with_min_distance(
                    NNODES[idxNodes], SQUAREROOMDIM, MINNODEDIST
                ),
                sourcePos=np.random.uniform(0, SQUAREROOMDIM, size=(1, 2)),
                dim=2
            )
            # Initialise WASN
            data.init_wasn(connecDist=CONNECDIST)
            # Plot
            fig = data.plotwasn()
            if not Path(OUTPUTFOLDER).is_dir():
                Path(OUTPUTFOLDER).mkdir()
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_adhoc.png')
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_adhoc.pdf')
            plt.close(fig)

            # Initialise fully connected WASN
            data.init_wasn(connecDist=np.Inf)
            # Plot
            fig = data.plotwasn()
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_fc.png')
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_fc.pdf')
            plt.close(fig)

            # Get minimum spanning tree
            data.init_wasn(connecDist=CONNECDIST)
            data.get_mst()
            # Plot
            fig = data.plotwasn()
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_mst.png')
            fig.savefig(f'{OUTPUTFOLDER}/WASN{idxmain + 1}_mst.pdf')
            plt.close(fig)


def get_graph_colors(myG, adjdict, SNRs):
    """Compute the edge colors of a directed graph according to SRO
    propagation (and redundancy reduction) rules."""
    
    edges = np.array(myG.edges)
    nodes = list(myG.nodes)
    edgecolors = [COLORDEFAULTLINK] * len(edges)
    edgecolors_stepbystep = [copy.copy(edgecolors)]

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
    edgecolors_stepbystep.append(copy.copy(edgecolors))

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
        edgecolors_stepbystep.append(copy.copy(edgecolors))

    return edgecolors_stepbystep

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
        width=1.5,
        node_color='#e2e2e2',
        edgecolors='k',
        node_size=400)
    # axes.grid()
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
    # axes.set_xticklabels(['' for _ in range(len(axes.get_xticklabels()))])
    # axes.set_yticklabels(['' for _ in range(len(axes.get_yticklabels()))])
    axes.set_xlim([0, SQUAREROOMDIM])
    axes.set_ylim([0, SQUAREROOMDIM])
    
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
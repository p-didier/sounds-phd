import sys, time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

# Comparison between NetworkX's Minimum Spanning Tree (MST) formation
# algorithms.

SEED = 12345  # random generators seed
NNODES = np.arange(5, stop=21)
NNODES = [20]
NMC_PERNNODES = 50  # number of Monte-Carlo runs per number of nodes
ALGOS = ['kruskal', 'prim', 'boruvka']  # MST formation algorithms to consider
# NETTYPE = 'fullyconnected'
NETTYPE = 'adhoc'


def main():
    """Main function."""

    # vvvv Value format: [mean, standard deviation] across MC runs
    allTimings = dict([
        (n, dict([
            (algo, [None, None]) for algo in ALGOS
        ])) for n in NNODES
    ])
    for idxNnodes in range(len(NNODES)):
        nNodes = NNODES[idxNnodes]
        timingsCurrNnodes = dict([(algo, []) for algo in ALGOS])
        for idxMC in range(NMC_PERNNODES):
            # Inform user
            print(f'{nNodes} nodes -- MC sim. {idxMC + 1}/{NMC_PERNNODES}...')
            # Generate original graph
            if NETTYPE == 'fullyconnected':
                OGnx = nx.complete_graph(nNodes)
            elif NETTYPE == 'adhoc':
                OGnx = nx.random_geometric_graph(nNodes, radius=0.5, seed=SEED)
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
            MSTs, timings = get_msts(OGnx, ALGOS)

            # Plot
            if 1:
                fig = plot_mst(OGnx, MSTs[ALGOS[0]], pos, nodeSize=100)
                if 0:
                    fig.savefig(f'{Path(__file__).parent}/figs/MSTexample.pdf')
                    fig.savefig(f'{Path(__file__).parent}/figs/MSTexample.png')
                plt.show()
            
            # Aggregate timing results for current number of nodes
            for algo in ALGOS:
                timingsCurrNnodes[algo].append(timings[algo])
        
        # Aggregate global timing information
        for algo in ALGOS:
            allTimings[nNodes][algo][0] = np.mean(timingsCurrNnodes[algo])
            allTimings[nNodes][algo][1] = np.std(timingsCurrNnodes[algo])

    # Show results
    fig = final_plot(allTimings)
    plt.show()
    if 0:
        fig.savefig(f'{Path(__file__).parent}/figs/timing_comp.png')
        fig.savefig(f'{Path(__file__).parent}/figs/timing_comp.pdf')

    stop = 1


def final_plot(timings: dict):
    """Timing-comparison performance plot."""

    miniDelta = 0.05
    legHandles = np.zeros(len(timings[list(timings.keys())[0]].keys()), dtype=object)

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    for ii, nNodesStr in enumerate(timings.keys()):    # for each number of nodes considered
        for jj, algo in enumerate(timings[nNodesStr].keys()):  # for each algorithm considered
            obj = axes.errorbar(
                x=ii + miniDelta * jj,
                y=1e3 * timings[nNodesStr][algo][0],
                yerr=1e3 * timings[nNodesStr][algo][1],
                marker='o',
                color=f'C{jj}',
                ecolor=f'C{jj}',
            )
            legHandles[jj] = obj.lines[0]
    axes.grid()
    axes.set_xticks(np.arange(len(timings.keys())))
    axes.set_xticklabels(list(timings.keys()))
    axes.set_xlabel('Number of nodes')
    axes.set_ylabel('[ms]')
    axes.set_title(f'MST algs. timings ({NMC_PERNNODES} MCs / # of nodes)')
    axes.legend(
        legHandles,
        [f'Algo: {algo}' for algo in timings[list(timings.keys())[0]].keys()]
    )

    return fig


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


def plot_mst(oggraph, mst: dict, pos, nodeSize=100):
    """Plot MST-formation output."""

    # Generate figure
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(9, 6)
    nx.draw(oggraph, pos=pos, ax=axes[0], node_size=nodeSize)
    axes[0].set_title(f'Original graph (type: "{NETTYPE}")')
    nx.draw(
        mst,
        pos=pos,
        ax=axes[1],
        edge_color='r',
        width=2.0,
        node_size=nodeSize
    )
    axes[1].set_title('MST')

    return fig


if __name__ == '__main__':
    sys.exit(main())
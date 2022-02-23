import matplotlib.pyplot as plt
import numpy as np

def buffer_sizes_evolution_plot(asc, bufferLengths, neighbourNodes, masterClock, N, bufferFlags):
    """Plots the evolution of the number of samples in each buffer in the network
    throughout the DANSE iterations. Buffer over- or under-flow are marked.
    
    Parameters
    ----------
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    bufferLengths : [numNodes x 1] list of [Nt x nNeighbours(k)] np.ndarrays of ints
        Node-specific buffer lengths evolutions for each of its neighbour.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists of ints
        Network indices of neighbours, per node.
    masterClock : [Nt x 1] np.ndarray of floats
        Master clock time instants.
    N : int
        Processing frame size.
    bufferFlags : [numNodes x 1] list of [Nt x nNeighbours(k)] np.ndarrays of ints (0, -1, 1, or 2)
    	Buffer over- or under-flow flags. 0: no flag; 2: "end of signal" flag.
    """

    fig, axes = plt.subplots(asc.numNodes,1)
    # Plot buffer sizes
    for k in range(len(bufferLengths)):
        axes[k].grid()
        for q in range(len(neighbourNodes[k])):
            axes[k].plot(masterClock, bufferLengths[k][:, q], label=f'Neighbor {q+1}')
        axes[k].set_title(f'Node {k+1} z-buffer size')
        axes[k].hlines(N, 0, masterClock[-1], colors='black', linestyles=':')
        # Plot buffer flags
        for q in range(len(neighbourNodes[k])):
            cond = bufferFlags[k][:, q] == 1
            if sum(cond) > 0:
                markerline, stemlines, baseline = axes[k].stem(masterClock[cond],
                    np.full(shape=(sum(cond),), fill_value=np.amax(bufferLengths[k])),
                    linefmt=f'C{q}--', label=f'Buffer overflow, neighbor {q+1}')
                markerline.set_markerfacecolor('none')
                markerline.set_marker((3, 0, q/len(neighbourNodes[k])*360))
                markerline.set_markeredgecolor(f'C{q}')
                baseline.set_linewidth(0)
            cond = bufferFlags[k][:, q] == -1
            if sum(cond) > 0:
                markerline, stemlines, baseline = axes[k].stem(masterClock[cond],
                    np.full(shape=(sum(cond),), fill_value=np.amax(bufferLengths[k])),
                    linefmt=f'C{q}--', label=f'Buffer underflow, neighbor {q+1}')
                markerline.set_markerfacecolor('none')
                markerline.set_marker((3, 2, q/len(neighbourNodes[k])*360))
                markerline.set_markeredgecolor(f'C{q}')
                baseline.set_linewidth(0)
            cond = bufferFlags[k][:, q] == 2
            if sum(cond) > 0:
                markerline, stemlines, baseline = axes[k].stem(masterClock[cond],
                    np.full(shape=(sum(cond),), fill_value=np.amax(bufferLengths[k])),
                    linefmt=f'C{q}--')
                markerline.set_markerfacecolor('none')
                markerline.set_marker((3, 1, q/len(neighbourNodes[k])*360))
                markerline.set_markeredgecolor(f'C{q}')
                baseline.set_linewidth(0)
        axes[k].legend(loc='upper right')
    axes[-1].set_xlabel(f'$t$ [s]')
    plt.tight_layout()
    plt.show()
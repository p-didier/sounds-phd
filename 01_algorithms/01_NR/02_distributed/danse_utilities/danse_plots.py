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

    # Create figure object
    fig, axes = plt.subplots(asc.numNodes * 2,1)

    # Define y-axes limits
    limHigh1 = np.amax(bufferLengths)
    limLow1 = 0
    ignoreLastFlag = False
    for k in range(len(bufferLengths)):
        for q in range(len(neighbourNodes[k])):
            if np.abs(bufferFlags[k][-1, q]) == np.amax(np.abs(bufferFlags[k][:, q])):
                ignoreLastFlag = True   # exclude last iteration (special case, only due to limitations of simulations with finite signal lengths)
                continue
    if ignoreLastFlag:
        limHigh2 = np.amax([b[:-1, :] for b in bufferFlags]) + 0.5
        limLow2 = np.amin([b[:-1, :] for b in bufferFlags]) - 0.5
    else:
        limHigh2 = np.amax(bufferFlags) + 0.5
        limLow2 = np.amin(bufferFlags) - 0.5

    # Plot buffer sizes
    for k in range(len(bufferLengths)):
        idxSubplot1 = 2 * k
        idxSubplot2 = 2 * k + 1
        axes[idxSubplot1].grid()
        for q in range(len(neighbourNodes[k])):
            axes[idxSubplot1].plot(masterClock, bufferLengths[k][:, q], label=f'Neighbor {q+1}')
        axes[idxSubplot1].set_title(f'Node {k+1} z-buffer size')
        axes[idxSubplot1].hlines(N, 0, masterClock[-1], colors='black', linestyles=':')
        axes[idxSubplot1].legend(loc='upper right')
        axes[idxSubplot1].set_ylim([limLow1, limHigh1])

        # Plot buffer flags
        axes[idxSubplot2].grid()
        axes[idxSubplot2].hlines(0, 0, masterClock[-1], colors='black', linestyles='-')
        for q in range(len(neighbourNodes[k])):
            cond = bufferFlags[k][:, q] != 0
            if cond[-1] and np.abs(bufferFlags[k][-1, q]) == np.amax(np.abs(bufferFlags[k][:, q])):
                cond[-1] = False    # exclude last iteration (special case, only due to limitations of simulations with finite signal lengths)
            if sum(cond) > 0:
                markerline, stemlines, baseline = axes[idxSubplot2].stem(masterClock[cond],
                    bufferFlags[k][cond, q],
                    linefmt=f'C{q}-', label=f'Neighbor {q+1}')
                markerline.set_markerfacecolor('none')
                markerline.set_marker((3, 0, q/len(neighbourNodes[k])*360))
                markerline.set_markeredgecolor(f'C{q}')
                baseline.set_linewidth(0)
        axes[idxSubplot2].set_title(f'Node {k+1} over- or under-flows')
        axes[idxSubplot2].set_ylabel(f'[BC len]')
        axes[idxSubplot2].legend(loc='upper right')
        axes[idxSubplot2].set_ylim([limLow2, limHigh2])
    axes[-1].set_xlabel(f'$t$ [s]')
    # plt.tight_layout()
    plt.show()
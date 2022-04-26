

from turtle import color
import numpy as np
import matplotlib.pyplot as plt


def get_events_matrix(timeInstants, N, Ns, L, nodeLinks, fs):
    """Returns the matrix the columns of which to loop over in SRO-affected simultaneous DANSE.
    For each event instant, the matrix contains the instant itself (in [s]),
    the node indices concerned by this instant, and the corresponding event
    flag: "0" for broadcast, "1" for update, "2" for end of signal. 
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray (floats)
        Time instants corresponding to the samples of each of the Nn nodes in the network.
    N : int
        Number of samples used for compression / for updating the DANSE filters.
    Ns : int
        Number of new samples per time frame (used in SRO-free sequential DANSE with frame overlap) (Ns < N).
    L : int
        Number of (compressed) signal samples to be broadcasted at a time to other nodes.
    nodeLinks : [Nn x Nn] np.ndarray (bools)
        Node links matrix. Element `(i,j)` is `True` if node `i` and `j` are connected, `False` otherwise.
    fs : list of floats
        Sampling frequency of each node.
    
    Returns
    -------
    outputEvents : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    fs : [Nn x 1] list of floats
        Sampling frequency of each node.
    --------------------- vvv UNUSED vvv ---------------------
    initialTimeBiases : [Nn x 1] np.ndarray (floats)
        [s] Initial time difference between first update at node `row index`
        and latest broadcast instant at node `column index` (diagonal elements are
        all set to zero: there is no bias w.r.t. locally recorded data).
    """

    # Make sure time stamps matrix is indeed a matrix, correctly oriented
    if timeInstants.ndim != 2:
        if timeInstants.ndim == 1:
            timeInstants = timeInstants[:, np.newaxis]
        else:
            raise ValueError('Unexpected number of dimensions for input `timeInstants`.')
    if timeInstants.shape[0] < timeInstants.shape[1]:
        timeInstants = timeInstants.T

    # Number of nodes
    nNodes = timeInstants.shape[1]

    # Check for clock jitter and save sampling frequencies
    fs = np.zeros(nNodes)
    for k in range(nNodes):
        deltas = np.diff(timeInstants[:, k])
        precision = int(np.ceil(np.abs(np.log10(np.mean(deltas) / 1000))))  # allowing computer precision errors down to 1e-3*mean delta.
        if len(np.unique(np.round(deltas, precision))) > 1:
            raise ValueError(f'[NOT IMPLEMENTED] Clock jitter detected: {len(np.unique(deltas))} different sample intervals detected for node {k+1}.')
        fs[k] = 1 / np.unique(np.round(deltas, precision))[0]

    # Total signal duration [s] per node (after truncation during signal generation)
    Ttot = timeInstants[-1, :]

    # Get expected DANSE update instants
    numUpdatesInTtot = np.floor(Ttot * fs / Ns)   # expected number of DANSE update per node over total signal length
    updateInstants = [np.arange(np.ceil(N / Ns), int(numUpdatesInTtot[k])) * Ns/fs[k] for k in range(nNodes)]  # expected DANSE update instants
    #                               ^ note that we only start updating when we have enough samples
    # Get expected broadcast instants
    numBroadcastsInTtot = np.floor(Ttot * fs / L)   # expected number of broadcasts per node over total signal length
    broadcastInstants = [np.arange(N/L, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]   # expected broadcast instants
    #                              ^ note that we only start broadcasting when we have enough samples to perform compression
    # Ensure that all nodes have broadcasted at least once before performing any update
    minWaitBeforeUpdate = np.amax([v[0] for v in broadcastInstants])
    for k in range(nNodes):
        updateInstants[k] = updateInstants[k][updateInstants[k] >= minWaitBeforeUpdate]
    
    # Build event matrix
    outputEvents = build_events_matrix(updateInstants, broadcastInstants, nNodes)

    return outputEvents, fs


def build_events_matrix(updateInstants, broadcastInstants, nNodes):
    """Sub-function of `get_events_matrix`, building the events matrix
    from the update and broadcast instants.
    
    Parameters
    ----------
    updateInstants : list of np.ndarrays (floats)
        Update instants per node [s].
    broadcastInstants : list of np.ndarrays (floats)
        Broadcast instants per node [s].
    nNodes : int
        Number of nodes in the network.

    Returns
    -------
    outputEvents : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    """

    numUniqueUpdateInstants = sum([len(np.unique(updateInstants[k])) for k in range(nNodes)])
    # Number of unique broadcast instants across the WASN
    numUniqueBroadcastInstants = sum([len(np.unique(broadcastInstants[k])) for k in range(nNodes)])
    # Number of unique update _or_ broadcast instants across the WASN
    numEventInstants = numUniqueBroadcastInstants + numUniqueUpdateInstants

    # Arrange into matrix
    flattenedUpdateInstants = np.zeros((numUniqueUpdateInstants, 3))
    flattenedBroadcastInstants = np.zeros((numUniqueBroadcastInstants, 3))
    for k in range(nNodes):
        idxStart_u = sum([len(updateInstants[q]) for q in range(k)])
        idxEnd_u = idxStart_u + len(updateInstants[k])
        flattenedUpdateInstants[idxStart_u:idxEnd_u, 0] = updateInstants[k]
        flattenedUpdateInstants[idxStart_u:idxEnd_u, 1] = k
        flattenedUpdateInstants[:, 2] = 1    # event reference "1" for updates

        idxStart_b = sum([len(broadcastInstants[q]) for q in range(k)])
        idxEnd_b = idxStart_b + len(broadcastInstants[k])
        flattenedBroadcastInstants[idxStart_b:idxEnd_b, 0] = broadcastInstants[k]
        flattenedBroadcastInstants[idxStart_b:idxEnd_b, 1] = k
        flattenedBroadcastInstants[:, 2] = 0    # event reference "0" for broadcasts
    # Combine
    eventInstants = np.concatenate((flattenedUpdateInstants, flattenedBroadcastInstants), axis=0)
    # Sort
    idxSort = np.argsort(eventInstants[:, 0], axis=0)
    eventInstants = eventInstants[idxSort, :]
    # Group
    outputEvents = []
    eventIdx = 0    # init while-loop
    nodesConcerned = []             # init
    eventTypesConcerned = []        # init
    while eventIdx < numEventInstants:

        currInstant = eventInstants[eventIdx, 0]
        nodesConcerned.append(int(eventInstants[eventIdx, 1]))
        eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))

        if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be grouped with the current instant
            nextInstant = eventInstants[eventIdx + 1, 0]
            while currInstant == nextInstant:
                eventIdx += 1
                currInstant = eventInstants[eventIdx, 0]
                nodesConcerned.append(int(eventInstants[eventIdx, 1]))
                eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))
                if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be grouped with the current instant
                    nextInstant = eventInstants[eventIdx + 1, 0]
                else:
                    eventIdx += 1
                    break
            else:
                eventIdx += 1
        else:
            eventIdx += 1

        # Sort events at current instant
        nodesConcerned = np.array(nodesConcerned, dtype=int)
        eventTypesConcerned = np.array(eventTypesConcerned, dtype=int)
        # 1) First broadcasts, then updates
        originalIndices = np.arange(len(nodesConcerned))
        idxUpdateEvent = originalIndices[eventTypesConcerned == 1]
        idxBroadcastEvent = originalIndices[eventTypesConcerned == 0]
        # 2) Order by node index
        if len(idxUpdateEvent) > 0:
            idxUpdateEvent = idxUpdateEvent[np.argsort(nodesConcerned[idxUpdateEvent])]
        if len(idxBroadcastEvent) > 0:
            idxBroadcastEvent = idxBroadcastEvent[np.argsort(nodesConcerned[idxBroadcastEvent])]
        # 3) Re-combine
        indices = np.concatenate((idxBroadcastEvent, idxUpdateEvent))
        # 4) Sort
        nodesConcerned = nodesConcerned[indices]
        eventTypesConcerned = eventTypesConcerned[indices]

        # Build events matrix
        outputEvents.append(np.array([currInstant, nodesConcerned, eventTypesConcerned], dtype=object))
        nodesConcerned = []         # reset
        eventTypesConcerned = []    # reset

    return outputEvents



def events_parser(events, startUpdates=None, printouts=False):
    """Printouts to inform user of DANSE events.
    
    Parameters
    ----------
    events : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
        Output of `get_events_matrix` function.
    startUpdates : list of bools
        Node-specific flags to indicate whether DANSE updates have started. 
    printouts : bool
        If True, inform user about current events after parsing.    

    Returns
    -------
    t : float
        Current time instant [s].
    eventTypes : list of str
        Events at current instant.
    nodesConcerned : list of ints
        Corresponding node indices.
    """

    # Parse events
    t = events[0]
    nodesConcerned = events[1]
    eventTypes = ['update'if val==1 else 'broadcast' for val in events[2]]

    if printouts:
        if startUpdates is None:
            startUpdates = [True for _ in range(len(eventTypes))]   
        if 'update' in eventTypes:
            txt = f't={np.round(t, 3)}s -- '
            updatesTxt = 'Updating nodes: '
            broadcastsTxt = 'Broadcasting nodes: '
            flagCommaUpdating = False    # little flag to add a comma (`,`) at the right spot
            for idxEvent in range(len(eventTypes)):
                k = int(nodesConcerned[idxEvent])   # node index
                if eventTypes[idxEvent] == 'broadcast':
                    if idxEvent > 0:
                        broadcastsTxt += ','
                    broadcastsTxt += f'{k + 1}'
                elif eventTypes[idxEvent] == 'update':
                    if startUpdates[k]:  # only print if the node actually has started updating (i.e. there has been sufficiently many autocorrelation matrices updates since the start of recording)
                        if not flagCommaUpdating:
                            flagCommaUpdating = True
                        else:
                            updatesTxt += ','
                        updatesTxt += f'{k + 1}'
            print(txt + broadcastsTxt + '; ' + updatesTxt)

    return t, eventTypes, nodesConcerned


def visualize_events(eventMatrix, tmax=1, suptitle=None):
    """Visually displays events.
    
    Parameters
    ----------
    eventMatrix : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    tmax : float
        [s] Maximum time instant to plot.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    nNodes = 0
    for events in eventMatrix:
        t, nodesConcerned, eventTypes = events[0], events[1], events[2]

        if t > tmax:
            break
        else:
            for idxEvent, eventType in enumerate(eventTypes):
                # Count number of nodes
                if nNodes < nodesConcerned[idxEvent] + 1:
                    nNodes = nodesConcerned[idxEvent] + 1

                if eventType == 0:
                    markerstyle = 'x'
                elif eventType == 1:
                    markerstyle = '+'

                ax.stem(t, eventType + 1,
                        linefmt=f'C{nodesConcerned[idxEvent]}-',
                        markerfmt=f'C{nodesConcerned[idxEvent]}{markerstyle}')
        stop = 1

    ax.grid()
    fig.set_figwidth(8)  
    fig.set_figheight(0.5*nNodes)

    # Manual legend
    for ii in range(nNodes):
        plt.text(t * 0.9, 1.75 - 0.2 * ii, f'Node {ii+1}', color=f'C{ii}', fontweight='bold', fontsize='large')
        
    plt.yticks([1, 2], ['Broadcast', 'Update'])
    plt.xlabel('$t$ [s]')
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()	
    plt.show()

    return None
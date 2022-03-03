import numpy as np
from numba import njit
import scipy, time


def danse_init(yin, settings, asc):
    """DANSE algorithms initialization function.
    
    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray (real)
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.

    Returns
    ----------
    rng : np.random generator
        Random generator to be used in DANSE algorithm for matrices initialization.
    win : N x 1 np.ndarray
        Window
    frameSize : int
        Number of samples used within one DANSE iteration.
    nNewSamplesPerFrame : int
        Number of new samples per time frame (used in SRO-free sequential DANSE with frame overlap).
    numIterations : int
        Expected number of DANSE iterations, given the length of the input signals `yin`.
    numBroadcasts : int
        Expected number of node-to-node broadcasts, given the length of the input signals `yin`.
    neighbourNodes : list of np.ndarray's
        Lists of neighbour nodes indices, per node.
    """

    # Define random generator
    rng = np.random.default_rng(settings.randSeed)

    # Adapt window array format
    win = settings.danseWindow[:, np.newaxis]

    # Define useful quantities
    frameSize = settings.chunkSize
    nNewSamplesPerFrame = int(settings.chunkSize * (1 - settings.chunkOverlap))
    numIterations = int((yin.shape[0] - frameSize) / nNewSamplesPerFrame) + 1
    numBroadcasts = int(np.ceil(yin.shape[0] / settings.broadcastLength))

    # Identify neighbours of each node
    neighbourNodes = []
    allNodeIdx = np.arange(asc.numNodes)
    for k in range(asc.numNodes):
        neighbourNodes.append(np.delete(allNodeIdx, k))   # Option 1) - FULLY-CONNECTED WASN

    return rng, win, frameSize, nNewSamplesPerFrame, numIterations, numBroadcasts, neighbourNodes


def check_autocorr_est(Ryy, Rnn, nUpRyy=0, nUpRnn=0, min_nUp=0):
    """Performs checks on autocorrelation matrices to ensure their
    usability within a MWF/DANSE algorithm

    Parameters
    ----------
    Ryy : [N x N] np.ndarray
        Autocorrelation matrix 1.
    Rnn : [N x N] np.ndarray
        Autocorrelation matrix 2.
    nUpRyy : int
        Number of updates already performed on Ryy.
    nUpRnn : int
        Number of updates already performed on Rnn.
    min_nUp : int
        Minimum number of updates to be performed on an autocorrelation matrix before usage.

    Returns
    -------
    flag : bool
        If true, autocorrelation matrices can be used. Else, not.  
    """
    flag = False
    # Full rank matrices
    if np.linalg.matrix_rank(Ryy) == Ryy.shape[0] and np.linalg.matrix_rank(Rnn) == Rnn.shape[0]:
        # Sufficient number of updates to get reasonable result
        if nUpRyy >= min_nUp and nUpRnn >= min_nUp:
            flag = True
    return flag


# -------- JIT-ed subfunctions for perform_gevd() --------
@njit
def invert_jitted(A):
    B = np.linalg.inv(A)
    return B

@njit
def sortevls_jitted(Qmat, sigma, rank):
    idx = np.flip(np.argsort(sigma))
    GEVLs_yy = np.flip(np.sort(sigma))
    Sigma_yy = np.diag(GEVLs_yy)
    Qmat = Qmat[:, idx]
    diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
    diagveig = np.append(diagveig, np.zeros(Sigma_yy.shape[0] - rank))
    return diagveig, Qmat

@njit
def getw_jitted(Qmat, diagveig, Evect):
    diagveig = diagveig.astype(np.complex128)
    Evect = Evect.astype(np.complex128)
    return np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect

@njit
def getw_jitted_tensor(QmH, Dmat, Qhermitian, Evect):
    Dmat = Dmat.astype(np.complex128)
    Evect = Evect.astype(np.complex128)
    w = np.matmul(np.matmul(np.matmul(QmH, Dmat), Qhermitian), Evect)
    return w
# --------------------------------------------------------


def perform_gevd(Ryy, Rnn, rank=1, refSensorIdx=0, jitted=False):
    """GEVD computations for DANSE.
    
    Parameters
    ----------
    Ryy : [N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    rank : int
        GEVD rank approximation.
    refSensorIdx : int
        Index of the reference sensor (>=0).
    jitted : bool
        If True, just a Just-In-Time (JIT) implementation via numba.

    Returns
    -------
    w : [N x 1] np.ndarray (complex)
        GEVD-DANSE filter coefficients.
    Qmat : [N x N] np.ndarray (complex)
        Hermitian conjugate inverse of the generalized eigenvectors matrix of the pencil {Ryy, Rnn}.
    """
    # Reference sensor selection vector 
    Evect = np.zeros((Ryy.shape[0],))
    Evect[refSensorIdx] = 1
    # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted
    sigma, Xmat = scipy.linalg.eigh(Ryy, Rnn)
    if jitted:
        Qmat = invert_jitted(Xmat.conj().T)
        diagveig, Qmat = sortevls_jitted(Qmat, sigma, rank)
        w = getw_jitted(Qmat, diagveig, Evect)
    else:
        Qmat = np.linalg.inv(Xmat.conj().T)
        # Sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigma))
        GEVLs_yy = np.flip(np.sort(sigma))
        Qmat = Qmat[:, idx]
        diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
        diagveig = np.append(diagveig, np.zeros(Ryy.shape[0] - rank))
        # LMMSE weights
        w = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect
    return w, Qmat


def perform_gevd_noforloop(Ryy, Rnn, rank=1, refSensorIdx=0):
    """GEVD computations for DANSE, `for`-loop free.
    
    Parameters
    ----------
    Ryy : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    rank : int
        GEVD rank approximation.
    refSensorIdx : int
        Index of the reference sensor (>=0).

    Returns
    -------
    w : [M x N] np.ndarray (complex)
        GEVD-DANSE filter coefficients.
    Qmat : [M x N x N] np.ndarray (complex)
        Hermitian conjugate inverse of the generalized eigenvectors matrix of the pencil {Ryy, Rnn}.
    """
    # ------------ for-loop-free estimate ------------
    n = Ryy.shape[-1]
    nKappas = Ryy.shape[0]
    # Reference sensor selection vector 
    Evect = np.zeros((n,))
    Evect[refSensorIdx] = 1

    sigma = np.zeros((nKappas, n))
    Xmat = np.zeros((nKappas, n, n), dtype=complex)
    for kappa in range(nKappas):
        # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted / vectorized
        sigmacurr, Xmatcurr = scipy.linalg.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :])
        # Flip Xmat to sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigmacurr))
        sigma[kappa, :] = sigmacurr[idx]
        Xmat[kappa, :, :] = Xmatcurr[:, idx]

    Qmat = np.linalg.inv(np.transpose(Xmat.conj(), axes=[0,2,1]))
    # GEVLs tensor
    Dmat = np.zeros((nKappas, n, n))
    Dmat[:, 0, 0] = np.squeeze(1 - 1/sigma[:, :rank])
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=[0,2,1])
    QmH = np.linalg.inv(Qhermitian)
    w = np.matmul(np.matmul(np.matmul(QmH, Dmat), Qhermitian), Evect)

    return w, Qmat


def back_to_time_domain(x, n):
    """Performs an IFFT after pre-processing of a frequency-domain
    signal chunk.
    
    Parameters
    ----------
    x : np.ndarray of complex
        Frequency-domain signal to be transferred back to time domain.

    Returns
    -------
    xout : np.ndarray of floats
        Time-domain version of signal.
    """

    x[0] = x[0].real      # Set DC to real value
    x[-1] = x[-1].real    # Set Nyquist to real value
    x = np.concatenate((x, np.flip(x[:-1].conj())[:-1]))
    # Back to time-domain
    xout = np.fft.ifft(x, n)

    return xout


def danse_compression(yq, w, win):
    """Performs local signals compression according to DANSE theory [1].
    
    Parameters
    ----------
    yq : [N x Ns] np.ndarray (real)
        Local sensor signals.
    w : [Ns x 1] np.ndarray (complex)
        Local filter estimated (from previous DANSE iteration).
    win : [N x 1] np.ndarray (real)
        Time window.
        
    Returns
    -------
    zq : [N x 1] np.ndarray (real)
        Compress local sensor signals (1-D).
    """

    # (I)FFT scalings
    fftscale = 1.0 / win.sum()
    ifftscale = win.sum()

    # Go to frequency domain
    yq_hat = fftscale * np.fft.fft(yq * win, len(win), axis=0)  # np.fft.fft: frequencies ordered from DC to Nyquist, then -Nyquist to -DC (https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)
    # Keep only positive frequencies
    yq_hat = yq_hat[:int(len(win) / 2 + 1)]
    # Compress using filter coefficients
    zq_hat = np.einsum('ij,ij->i', w.conj(), yq_hat)  # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
    # Format for IFFT 
    zq = ifftscale * back_to_time_domain(zq_hat, len(win))
    zq = np.real_if_close(zq)

    return zq


def process_incoming_signals_buffers(zBufferk, zPreviousk, neighs, ik, frameSize, N, L, lastExpectedIter):
    """When called, processes the incoming data from other nodes, as stored in local node's buffers.
    Called whenever a DANSE update can be performed (`N` new local samples were captured).
    
    Parameters
    ----------
    zBufferk : [Nneighbors x 1] list of np.ndarrays (float)
        Buffers of node `k` at current iteration.
    zPreviousk : [Nneighbors x 1] list of np.ndarrays (float)
        Compressed signals used by node `k` at previous iteration (within $\\tilde{\mathbf{y}}_k$ in [1]'s notation)
    neighs : list of integers
        List of indices corresponding to node `k`'s neighbours in the WASN.
    ik : int
        DANSE iteration index at node `k`.
    frameSize : int
        Size of FFT / DANSE update frame.
    N : int
        Expected number of new (never seen before) samples at every DANSE update.
    L: int
        Number of samples expected to be transmitted per broadcast.
    lastExpectedIter : int
        Expected index of the last DANSE iteration. 
        
    Returns
    -------
    zk : [Nneighbors x frameSize] np.ndarray of floats
        Compressed signal frames to be used at iteration `ik` by node `k`.
    bufferFlags : [Nneighbors x 1] np.ndarray of ints
        Flags for buffer over- (positive value) or under-flows (negative value).
        Value 0: no over- or under-flow.
        Value NaN: last iteration, lack of samples due to cumulated SRO.
    """

    zk = np.empty((frameSize, 0), dtype=float)       # initialize compressed signal matrix ($\mathbf{z}_{-k}$ in [1]'s notation)
    bufferFlags = np.zeros(len(neighs))    # flags (positive: +1; negative: -1; or none: 0) -- indicate buffer over- or under-flow

    for idxq in range(len(neighs)):
        Bq = len(zBufferk[idxq])  # buffer size for neighbour node `q`
        if ik == 0: # first DANSE iteration case -- we are expecting an abnormally full buffer, with an entire DANSE chunk size inside of it
            if Bq == frameSize: 
                # There is no significant SRO between node `k` and `q`.
                # Response: node `k` uses all samples in the `q`^th buffer.
                zCurrBuffer = zBufferk[idxq]
            elif (frameSize - Bq) % L == 0 and Bq < frameSize:
                # Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.
                # Interpretation: Node `q` samples slower than node `k`. 
                # Response: ...
                raise ValueError('NOT IMPLEMENTED YET: Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.')
            elif (frameSize - Bq) % L == 0 and Bq > frameSize:
                # Node `q` has already transmitted too much data to node `k`.
                # Interpretation: Node `q` samples faster than node `k`.
                # Response: node `k` raises a positive flag and uses the last `frameSize` samples from the `q`^th buffer.
                bufferFlags[idxq] = +1 * int((frameSize - Bq) / L)      # raise negative flag
                zCurrBuffer = zBufferk[idxq][-frameSize:]
        else:   # not the first DANSE iteration -- we are expecting a normally full buffer, with a DANSE chunk size considering overlap
            if Bq == N:             # case 1: no broadcast frame mismatch between node `k` and node `q`
                pass
            elif (N - Bq) % L == 0 and Bq < N:       # case 2: negative broadcast frame mismatch between node `k` and node `q`
                print(f'[-] Buffer underflow at current node`s B_{neighs[idxq]+1} buffer | -{int(np.abs((N - Bq) / L))} broadcast(s)')
                bufferFlags[idxq] = -1 * int(np.abs((N - Bq) / L))      # raise negative flag
            elif (N - Bq) % L == 0 and Bq > N:       # case 3: positive broadcast frame mismatch between node `k` and node `q`
                print(f'[+] Buffer overflow at current node`s B_{neighs[idxq]+1} buffer | +{int(np.abs((N - Bq) / L))} broadcasts(s)')
                bufferFlags[idxq] = +1 * int(np.abs((N - Bq) / L))       # raise positive flag
            else:
                if (N - Bq) % L != 0 and np.abs(ik - lastExpectedIter) < 10:
                    print('[!] This is the last iteration -- not enough samples anymore due to cumulated SROs effect, skip update.')
                    bufferFlags[idxq] = np.NaN   # raise "end of signal" flag
                else:
                    if (N - Bq) % L == 0:
                        raise ValueError(f'NOT IMPLEMENTED YET: too large SRO, >1 broadcast length over- or under-flow.')
                    else:
                        raise ValueError(f'ERROR: Unexpected buffer size for neighbor node q={neighs[idxq]+1}.')
            # Build current buffer
            zCurrBuffer = np.concatenate((zPreviousk[-(frameSize - Bq):, idxq], zBufferk[idxq]), axis=0)
        # Stack compressed signals
        zk = np.concatenate((zk, zCurrBuffer[:, np.newaxis]), axis=1)

    return zk, bufferFlags


def count_samples(tVect, instant, lastSampIdx, nSampBC, nSampLocal):
    """Counts the number of past samples at master clock instant `instant`
    and updates the number of samples lined up for broadcast, as well as 
    the number of new samples captured since the last DANSE iteration.
    
    Parameters
    ----------
    tVect : [Nt x 1] np.ndarray
        Time stamps vectors for current node.
    instant : float
        Current master clock time instant [s].
    lastSampIdx : int
        Index of the last samples captured by node `k`.
    nSampBC : int
        Number of samples lined up for broadcast.
    nSampLocal : int
        Number of new local samples captured since the last DANSE update.

    Returns
    -------
    lastSampIdx : int
        Updated index of the last samples captured by node `k`.
    nSampBC : int
        Updated number of samples lined up for broadcast.
    nSampLocal : int
        Updated number of new local samples captured since the last DANSE update.
    """

    passedInstants = tVect[tVect <= instant]     # list of passed time stamps at node `k`
    idxCurrSample = len(passedInstants) - 1                 # number of samples accumulated at master-time `t` at node `k`
    nSampBC += idxCurrSample - lastSampIdx            # can be more than 1 when SROs are present...
    nSampLocal += idxCurrSample - lastSampIdx         # can be more than 1 when SROs are present...
    
    lastSampIdx = idxCurrSample    # update current sample index at node `k`
    
    return lastSampIdx, nSampBC, nSampLocal


def broadcast_flag_raising(compressStarted, nSampBC, N, L):
    """Returns a flag to let node know whether it should broadcast
    its local compressed observations to its neighbours.
    
    Parameters
    ----------
    compressStarted : bool
        If True, indicates that compression has already started.
    nSampBC : int
        Number of samples lined up for broadcast.
    N : int
        Processing frame size.
    L : int
        Broadcast chunk size.

    Returns
    -------
    broadcastFlag : bool
        If True, the node should broadcast now. Otherwise, not.
    """

    broadcastFlag = False
    if not compressStarted:   # first broadcast -- need `settings.stftWinLength` sample to perform compression in freq.-domain
        if nSampBC >= N:
            broadcastFlag = True
            compressStarted = True
    elif nSampBC >= L:     # not the first broadcast -- need `settings.broadcastLength` new samples to perform compression in freq.-domain
        broadcastFlag = True

    return broadcastFlag, compressStarted


def fill_buffers(k, neighbourNodes, lk, zBuffer, zLocal, L):
    """Fill in buffers -- simulating broadcast of compressed signals
    from one node (`k`) to its neighbours.
    
    Parameters
    ----------
    k : int
        Current node index.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists of ints
        Network indices of neighbours, per node.
    lk : [numNodes x 1] list of ints
        Broadcast index per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays of floats
        Compressed signals buffers for each node and its neighbours.
    zLocal : [n x 1] np.ndarray of floats
        Latest compressed local signals to be broadcasted from node `k`.
    L : int
        Broadcast chunk length.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays of floats
        Updated compressed signals buffers for each node and its neighbours.
    """

    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]             # network index of node `q`
        idxKforNeighbor = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighbor = idxKforNeighbor[0]    # node `k` index from node `q`'s perspective
        # Fill in neighbors' buffers with the L = `settings.broadcastLength` last samples of local compressed signals
        if lk[k] == 0:
            # Broadcast the all compressed signal (1st broadcast period, no "old" samples)
            zBufferCurr = np.concatenate((zBuffer[q][idxKforNeighbor], zLocal), axis=0)
        else:
            # Only broadcast the `L` last samples of local compressed signals
            zBufferCurr = np.concatenate((zBuffer[q][idxKforNeighbor], zLocal[-L:]), axis=0)
        zBuffer[q][idxKforNeighbor] = zBufferCurr   # necessary for `np.concatenate()` afterwards

    return zBuffer


def get_events_matrix(timeInstants, N, Ns, L):
    """Returns the matrix the columns of which to loop over in SRO-affected simultaneous DANSE.
    For each event instant, the matrix contains the instant itself (in [s]),
    the node indices concerned by this instant, and the corresponding event
    flag: "0" for broadcast, "1" for update.
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray of floats
        Time instants corresponding to the samples of each of the Nn nodes in the network.
    N : int
        Number of samples used for compression / for updating the DANSE filters.
    Ns : int
        Number of new samples per time frame (used in SRO-free sequential DANSE with frame overlap) (Ns < N).
    L : int
        Number of (compressed) signal samples to be broadcasted at a time to other nodes.
    
    Returns
    -------
    eventInstants : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    fs : [Nn x 1] list of floats
        Sampling frequency of each node.
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

    Ttot = timeInstants[-1, 0]      # total duration [s]
    
    # Get expected DANSE update instants
    numUpdatesInTtot = np.floor(Ttot * fs / Ns)   # expected number of DANSE update per node over total signal length
    updateInstants = [np.arange(np.ceil(N / Ns), int(numUpdatesInTtot[k])) * Ns/fs[k] for k in range(nNodes)]  # expected DANSE update instants
    #                            ^ note that we only start updating when we have enough samples
    # Get expected broadcast instants
    numBroadcastsInTtot = np.floor(Ttot * fs / L)   # expected number of broadcasts per node over total signal length
    broadcastInstants = [np.arange(N / L, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]   # expected broadcast instants
    #                               ^ note that we only start broadcasting when we have enough samples to perform compression

    # Number of unique update instants across the WASN
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
    eventInstantsFormatted = []
    eventIdx = 0    # init while-loop
    nodesConcerned = []             # init
    eventTypesConcerned = []        # init
    while eventIdx < numEventInstants:

        currInstant = eventInstants[eventIdx, 0]
        nodesConcerned.append(eventInstants[eventIdx, 1])
        eventTypesConcerned.append(eventInstants[eventIdx, 2])

        if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be groued with the current instant
            nextInstant = eventInstants[eventIdx + 1, 0]
            while currInstant == nextInstant:
                eventIdx += 1
                currInstant = eventInstants[eventIdx, 0]
                nodesConcerned.append(eventInstants[eventIdx, 1])
                eventTypesConcerned.append(eventInstants[eventIdx, 2])
                if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be groued with the current instant
                    nextInstant = eventInstants[eventIdx + 1, 0]
                else:
                    eventIdx += 1
                    break
            else:
                eventIdx += 1

        # Sort events at current instant
        nodesConcerned = np.array(nodesConcerned)
        eventTypesConcerned = np.array(eventTypesConcerned)
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
        eventInstantsFormatted.append(np.array([currInstant, nodesConcerned, eventTypesConcerned], dtype=object))
        nodesConcerned = []         # reset
        eventTypesConcerned = []    # reset

    return eventInstantsFormatted, fs


def broadcast(t, k, fs, L, yk, w, win, neighbourNodes, lk, zBuffer):
    """Performs the broadcast of data from node `k` to its neighbours.
    
    Parameters
    ----------
    t : float
        Time instant [s].
    k : int
        Node index in WASN.
    fs : float
        Node `k`'s sampling frequency [samples/s].
    L : int
        Number of samples to broadcast.
    yk : np.ndarray of floats
        Local sensor data from node `k`.
    w : np.ndarray of complex
        Filter coefficients used for compression.
    win : np.ndarray of floats
        Window used for passage from time- to freq.-domain and vice-versa.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists of ints
        Network indices of neighbours, per node.
    lk : [numNodes x 1] list of ints
        Broadcast index per node.
    zBuffer : [nNodes x 1] list of [Nneighbors x 1] lists of np.ndarrays (float)
        For each node, buffers at previous iteration.
    
    Returns
    -------
    zBuffer : [nNodes x 1] list of [Nneighbors x 1] lists of np.ndarrays (float)
        For each node, buffers at current iteration.
    """
    # Check inputs
    if np.round(t * fs) != np.round(t * fs, 10):
        raise ValueError('Unexpected time instant: does not correspond to a specific sample.')

    # Interpret variables
    Nchunk = len(win)     # number of samples per compression chunk

    # Compress current data chunk in the frequency domain
    zLocal = danse_compression(yk, w[:, :yk.shape[-1]], win)        # local compressed signals

    # Loop over node `k`'s neighbours and fill their buffers
    zBuffer = fill_buffers(k, neighbourNodes, lk, zBuffer, zLocal, L)
    
    lk[k] += 1  # increment local broadcast index

    return zBuffer

import numpy as np
from numba import njit
import scipy


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
        # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted
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
    bufferFlags : [Nneighbors x 1] np.ndarray of ints (+1, -1, 0, or 2)
        Flags for buffer over- (+1) or under-flows (-1).
        Value 0: no over- or under-flow.
        Value 2: last iteration, lack of samples due to cumulated SRO.
    """

    zk = np.empty((frameSize, 0), dtype=float)       # initialize compressed signal matrix ($\mathbf{z}_{-k}$ in [1]'s notation)
    bufferFlags = np.zeros(len(neighs))    # flags (positive: +1; negative: -1; or none: 0) -- indicate buffer over- or under-flow

    for idxq in range(len(neighs)):
        Bq = len(zBufferk[idxq])  # buffer size for neighbour node `q`
        if ik == 0: # first DANSE iteration case
            if Bq == frameSize: 
                # There is no significant SRO between node `k` and `q`.
                # Response: node `k` uses all samples in the `q`^th buffer.
                zCurrBuffer = zBufferk[idxq]
            elif Bq < frameSize:
                # Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.
                # Interpretation: Node `q` samples slower than node `k`. 
                # Response: ...
                raise ValueError('NOT YET IMPLEMENTED CASE: Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.')
            elif Bq > frameSize:
                # Node `q` has already transmitted too much data to node `k`.
                # Interpretation: Node `q` samples faster than node `k`.
                # Response: node `k` raises a positive flag and uses the last `frameSize` samples from the `q`^th buffer.
                bufferFlags[idxq] = 1      # raise negative flag
                zCurrBuffer = zBufferk[idxq][-frameSize:]
        else:
            if Bq == N:
                # Case 1: no broadcast frame mismatch between node `k` and node `q`
                # Build element of `z` with all the current buffer and a part of the previous `z` values
                pass
            elif Bq == N - L:     # case 2: negative broadcast frame mismatch between node `k` and node `q`
                print(f'>>> Buffer underflow (current node`s B_{neighs[idxq]+1} buffer).')
                bufferFlags[idxq] = -1      # raise negative flag
            elif Bq == N + L:     # case 3: positive broadcast frame mismatch between node `k` and node `q`
                print(f'>>> Buffer overflow (current node`s B_{neighs[idxq]+1} buffer).')
                bufferFlags[idxq] = 1       # raise positive flag
            else:
                if Bq < N and np.abs(ik - lastExpectedIter) < 10:
                    print('>>> This is the last iteration -- not enough data due to cumulated SROs effect, skip update.')
                    bufferFlags[idxq] = 2       # raise ultimate flag
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


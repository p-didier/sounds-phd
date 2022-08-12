
import numpy as np
from numba import njit
import scipy.linalg as sla
import scipy.signal as sig
from . import classes
from danse_utilities.events_manager import *
from .sro_subfcns import *
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import sys
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
import dsp.linalg

"""
References:
[1] Szurley, Joseph, Alexander Bertrand, and Marc Moonen.
"Topology-independent distributed adaptive node-specific signal
estimation in wireless sensor networks." IEEE Transactions on
Signal and Information Processing over Networks 3.1 (2016): 130-144.
"""


def danse_init(yin, settings: classes.ProgramSettings, asc, showWasnConnections=False):
    """DANSE algorithms initialization function.
    
    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray (real)
        The microphone signals in the time domain.
    settings : ProgramSettings object
        The settings for the current run.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).

    Returns
    ----------
    rng : np.random generator
        Random generator to be used in DANSE algorithm for matrices initialization.
    winWOLAanalysis : N x 1 np.ndarray
        WOLA analysis window (time domain to freq. domain).
    winWOLAsynthesis : N x 1 np.ndarray
        WOLA synthesis window (freq. domain to time domain).
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
    win = settings.danseWindow
    winWOLAanalysis = np.sqrt(win)      # WOLA analysis window
    winWOLAsynthesis = np.sqrt(win)     # WOLA synthesis window

    # Define useful quantities
    frameSize = settings.chunkSize
    nNewSamplesPerFrame = int(settings.chunkSize * (1 - settings.chunkOverlap))
    numIterations = int((yin.shape[0] - frameSize) / nNewSamplesPerFrame) + 1
    numBroadcasts = int(np.ceil(yin.shape[0] / settings.broadcastLength))

    # Identify neighbours of each node
    allNodeIdx = np.arange(asc.numNodes)

    # Option 1) - FULLY-CONNECTED WASN
    if settings.wasnTopology == 'fully_connected':
        neighbourNodes = []  
        for k in range(asc.numNodes):
            neighbourNodes.append(np.delete(allNodeIdx, k))

    # Option 2) - AD-HOC TOPOLOGY (via expanding communication radius)
    elif settings.wasnTopology == 'adhoc':
        comRad = 0  # initiate communication radius to 0 m
        allReachable = False
        nodeDistances = np.zeros((asc.numNodes, asc.numNodes))
        for ii in range(asc.numNodes):
            for jj in range(asc.numNodes):
                nodeDistances[ii, jj] = np.linalg.norm(asc.sensorCoords[ii,:] - asc.sensorCoords[jj,:])

        while not allReachable:
            # Connection matrix
            connectionMatrix = nodeDistances <= comRad
            # Check if "every node is reachable by some set of links to every other node in the WSN" [1]
            reachable = np.full((asc.numNodes, asc.numNodes), fill_value=False)
            for ii in range(asc.numNodes):
                for jj in range(asc.numNodes):
                    # Base case
                    if (ii == jj):
                        reachable[ii, jj] = True
                        continue

                    # Mark all the vertices as not visited
                    visited = [False for _ in range(asc.numNodes)]

                    # Create a queue for BFS
                    queue = deque()

                    # Mark the current node as visited and enqueue it
                    visited[ii] = True
                    queue.append(ii)

                    while (len(queue) > 0):
                    
                        # Dequeue a vertex from queue and print
                        s = queue.popleft()

                        # Get all adjacent vertices of the dequeued vertex s
                        # If a adjacent has not been visited, then mark it
                        # visited and enqueue it
                        for i in allNodeIdx[connectionMatrix[s,:]]:

                            # If this adjacent node is the destination node,
                            # then return true
                            if (i == jj):
                                reachable[ii, jj] = True

                            # Else, continue to do BFS
                            if (not visited[i]):
                                visited[i] = True
                                queue.append(i)

            if reachable.all():
                allReachable = True
            else:
                comRad += .1    # increase communication radius by a bit
                print(f'Increasing communication radius to {comRad} m...')

        # Build neighbourNodes object
        neighbourNodes = []
        for k in range(asc.numNodes):
            neighbours = allNodeIdx[connectionMatrix[k, :]]
            neighbourNodes.append(neighbours[neighbours != k])

        # Show WASN connections
        if showWasnConnections:
            plot_wasn_connections(asc.sensorCoords, connectionMatrix)

    return rng, winWOLAanalysis, winWOLAsynthesis, frameSize, nNewSamplesPerFrame, numIterations, numBroadcasts, neighbourNodes


def plot_wasn_connections(coords: np.ndarray, connectionMatrix: np.ndarray):
    """Quick visualization of the WASN connections (for an ad-hoc topology)."""

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111, projection='3d')
    # Plot nodes
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=50, c='r', marker='o')
    # Plot connections
    for ii in range(connectionMatrix.shape[0]):
        for jj in range(connectionMatrix.shape[1]):
            if ii > jj:     # only considered one half of the connection matrix (undirected graph)
                if connectionMatrix[ii, jj]:
                    ax.plot([coords[ii,0], coords[jj,0]],
                            [coords[ii,1], coords[jj,1]],
                            [coords[ii,2], coords[jj,2]], 'k--')
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.tight_layout()	
    plt.show()
    

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
    sigma, Xmat = sla.eigh(Ryy, Rnn, check_finite=False, driver='gvd')
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
    Xmat : [M x N x N] np.ndarray (complex)
        Qeneralized eigenvectors matrix of the pencil {Ryy, Rnn}.
    """
    # ------------ for-loop-free estimate ------------
    n = Ryy.shape[-1]
    nFreqs = Ryy.shape[0]
    # Reference sensor selection vector 
    Evect = np.zeros((n,))
    Evect[refSensorIdx] = 1

    sigma = np.zeros((nFreqs, n))
    Xmat = np.zeros((nFreqs, n, n), dtype=complex)

    # t0 = time.perf_counter()
    for kappa in range(nFreqs):
        # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted nor vectorized
        sigmacurr, Xmatcurr = sla.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :], check_finite=False, driver='gvd')
        # Flip Xmat to sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigmacurr))
        sigma[kappa, :] = sigmacurr[idx]
        Xmat[kappa, :, :] = Xmatcurr[:, idx]

    Qmat = np.linalg.inv(np.transpose(Xmat.conj(), axes=[0,2,1]))
    # GEVLs tensor
    Dmat = np.zeros((nFreqs, n, n))
    for ii in range(rank):
        Dmat[:, ii, ii] = np.squeeze(1 - 1/sigma[:, ii])
    # Dmat = np.zeros((nFreqs, n, n))
    # Dmat[:, 0, 0] = np.squeeze(1 - 1/sigma[:, :rank])
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=[0,2,1])
    w = np.matmul(np.matmul(np.matmul(Xmat, Dmat), Qhermitian), Evect)


    # Dmat2 = np.zeros((nFreqs, n, n))
    # for ii in range(rank):
    #     Dmat2[:, ii, ii] = np.squeeze(sigma[:, ii:(ii+1)] - 1)
    # w = np.zeros((nFreqs, n, n), dtype=complex)
    # for kappa in range(nFreqs):
    #     w[kappa, :] = np.linalg.pinv(Ryy[kappa, :, :]) @ Qmat[kappa, :, :] @ Dmat2[kappa, :, :] @ Qmat[kappa, :, :].conj().T @ Evect

    return w, Qmat


def perform_update_noforloop(Ryy, Rnn, refSensorIdx=0):
    """Regular DANSE update computations, `for`-loop free.
    No GEVD involved here.
    
    Parameters
    ----------
    Ryy : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    refSensorIdx : int
        Index of the reference sensor (>=0).

    Returns
    -------
    w : [M x N] np.ndarray (complex)
        Regular DANSE filter coefficients.
    """
    # Reference sensor selection vector
    Evect = np.zeros((Ryy.shape[-1],))
    Evect[refSensorIdx] = 1

    # Cross-correlation matrix update 
    ryd = np.matmul(Ryy - Rnn, Evect)
    # Update node-specific parameters of node k
    Ryyinv = np.linalg.inv(Ryy)
    w = np.matmul(Ryyinv, ryd[:,:,np.newaxis])
    w = w[:, :, 0]  # get rid of singleton dimension
    return w


def back_to_time_domain(x, n, axis=0):
    """Performs an IFFT after pre-processing of a frequency-domain
    signal chunk.
    
    Parameters
    ----------
    x : np.ndarray of complex
        Frequency-domain signal to be transferred back to time domain.
    n : int
        IFFT order.
    axis : int (0 or 1)
        Array axis where to perform IFFT -- not implemented for more than 2-D arrays.

    Returns
    -------
    xout : np.ndarray of floats
        Time-domain version of signal.
    """

    # Interpret `axis` parameter
    flagSingleton = False
    if x.ndim == 1:
        x = x[:, np.newaxis]
        flagSingleton = True
    elif x.ndim > 2:
        raise np.AxisError(f'{x.ndim}-D arrays not permitted.')
    if axis not in [0,1]:
        raise np.AxisError(f'`axis={axis}` is not permitted.')

    if axis == 1:
        x = x.T

    # Check dimension
    if x.shape[0] != n/2 + 1:
        raise ValueError('`x` should be (n/2 + 1)-dimensioned along the IFFT axis.')

    x[0, :] = x[0, :].real      # Set DC to real value
    x[-1, :] = x[-1, :].real    # Set Nyquist to real value
    x = np.concatenate((x, np.flip(x[:-1, :].conj(), axis=0)[:-1, :]), axis=0)
    
    if flagSingleton: # important to go back to original input dimensionality before FFT (bias of np.fft.fft with (n,1)-dimensioned input)
        x = np.squeeze(x)

    # Back to time-domain
    xout = np.fft.ifft(x, n, axis=0)

    if axis == 1:
        xout = xout.T

    return xout


def danse_compression_whole_chunk(yq, wHat, h, f, zqPrevious=None):
    """Performs local signals compression in the frequency domain.

    Parameters
    ----------
    yq : [N x nSensors] np.ndarray (float)
        Local sensor signals.
    wHat : [N/2 x nSensors] np.ndarray (float or complex)
        Frequency-domain local filter estimate (from latest DANSE iteration).
    h : [`n` x 1] np.ndarray (float)
        WOLA analysis window (time-domain to WOLA-domain).
    f : [`n` x 1] np.ndarray (float)
        WOLA synthesis window (WOLA-domain to time-domain).
    zqPrevious : [N x 1] np.ndarray (float)
        Previous time-domain chunk of compressed signal.

    Returns
    -------
    zqHat : [N/2 x 1] np.ndarray (complex)
        Frequency-domain compressed signal for current frame.
    zq : [N x 1] np.ndarray (float)
        Time-domain latest WOLA chunk of compressed signal (after OLA).
    """

    # Check for single-sensor case
    flagSingleSensor = False
    if wHat.shape[-1] == 1:
        wHat = np.squeeze(wHat)
        yq = np.squeeze(yq)
        flagSingleSensor = True
    
    # Transfer local observations to frequency domain
    n = len(yq)     # DFT order

    # WOLA analysis stage
    if flagSingleSensor:
        yqHat = np.fft.fft(np.squeeze(yq) * h, n, axis=0)
        # Keep only positive frequencies
        yqHat = yqHat[:int(n/2 + 1)]
        # Apply linear combination to form compressed signal
        zqHat = wHat.conj() * yqHat     # single sensor = simple element-wise multiplication
    else:
        yqHat = np.fft.fft(np.squeeze(yq) * h[:, np.newaxis], n, axis=0)
        # Keep only positive frequencies
        yqHat = yqHat[:int(n/2 + 1), :]
        # Apply linear combination to form compressed signal
        zqHat = np.einsum('ij,ij->i', wHat.conj(), yqHat)  # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850

    # WOLA synthesis stage
    if zqPrevious is not None:
        # IDFT
        zqCurr = back_to_time_domain(zqHat, n, axis=0)
        zqCurr = np.real_if_close(zqCurr)
        zqCurr *= f    # multiply by synthesis window

        if not np.any(zqPrevious):
            # No previous frame, keep current frame
            zq = zqCurr
        else:
            # Overlap-add
            zq = np.zeros(n)
            zq[:(n // 2)] = zqPrevious[-(n // 2):]   # TODO: consider case with multiple neighbor nodes (`len(zqPrevious) > 1`)
            zq += zqCurr
    else:
        zq = None
    
    return zqHat, zq


def danse_compression_few_samples(yq, wqqHat, n, L, wIRprevious,
                            winWOLAanalysis, winWOLAsynthesis, R, 
                            updateBroadcastFilter=False):
    """Performs local signals compression according to DANSE theory [1],
    in the time-domain (to be able to broadcast the compressed signal sample
    per sample between nodes).
    Approximate WOLA filtering process via linear convolution.
    
    Parameters
    ----------
    yq : [Ntotal x nSensors] np.ndarray (real)
        Local sensor signals.
    wqqHat : [N/2 x nSensors] np.ndarray (real or complex)
        Frequency-domain local filter estimate (from latest DANSE iteration).
    n : int
        FFT order.
    L : int
        Broadcast length [samples].
    winWOLAanalysis : [`n` x 1] np.ndarray (float)
        WOLA analysis window (time-domain to WOLA-domain).
    winWOLAsynthesis : [`n` x 1] np.ndarray (float)
        WOLA synthesis window (WOLA-domain to time-domain).
    R : int
        Sample shift between adjacent windows.
    updateBroadcastFilter : bool
        If True, update TD filter for broadcast.

    Returns
    -------
    zq : [N x 1] np.ndarray (real)
        Compress local sensor signals (1-D).
    """

    # Profiling
    if updateBroadcastFilter:
        # Distortion function approximation of the WOLA filterbank [see Word journal week 32 and previous, 2022]
        wqq = back_to_time_domain(wqqHat, n, axis=0)
        wqq = np.real_if_close(wqq)         
        wIR = np.zeros((2 * n - 1, wqq.shape[1]))
        for m in range(wqq.shape[1]):
            Hmat = sla.circulant(np.flip(wqq[:, m]))
            Amat = np.diag(winWOLAsynthesis) @ Hmat @ np.diag(winWOLAanalysis)
            for ii in np.arange(start=-n+1, stop=n):
                wIR[ii + n - 1, m] = np.sum(np.diagonal(Amat, ii))
        wIR /= R
    else:
        wIR = wIRprevious
    
    # Perform convolution
    yfiltLastSamples = np.zeros((L, yq.shape[-1]))
    for idxSensor in range(yq.shape[-1]):
        # yfiltLastSamples[:, idxSensor] = sum(yq[:, idxSensor] * np.flip(wIR[:, idxSensor]))   # manual convolution to only get the last sample
        # tmp = sig.convolve(yq[:, idxSensor], wIR[:, idxSensor], mode='full')
        # yfiltLastSamples[:, idxSensor] = tmp[-(n + L):-n]     # extract the `L` sample preceding the middle of the convolution output

        idDesired = np.arange(start=len(wIR) - L, stop=len(wIR))   # indices required from convolution output
        tmp = dsp.linalg.extract_few_samples_from_convolution(idDesired, wIR, yq[:, idxSensor])
        yfiltLastSamples[:, idxSensor] = tmp

    zq = np.sum(yfiltLastSamples, axis=1)

    return zq, wIR



def process_incoming_signals_buffers(zBufferk, zPreviousk, neighs, ik, N, Ns, L, lastExpectedIter, broadcastDomain):
    """When called, processes the incoming data from other nodes, as stored in local node's buffers.
    Called whenever a DANSE update can be performed (`N` new local samples were captured since last update).
    
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
    N : int
        Size of FFT / DANSE update frame.
    Ns : int
        Expected number of new (never seen before) samples at every DANSE update.
    L: int
        Number of samples expected to be transmitted per broadcast.
    lastExpectedIter : int
        Expected index of the last DANSE iteration. 
    broadcastDomain : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    
    Returns
    -------
    zk : [Nneighbors x frameSize] np.ndarray of floats
        Compressed signal frames to be used at iteration `ik` by node `k`.
    bufferFlags : [Nneighbors x 1] np.ndarray of ints
        Flags for buffer over- (positive value) or under-flows (negative value).
        Value 0: no over- or under-flow.
        Value NaN: last iteration, lack of samples due to cumulated SRO.
    """

    # Number of frequency lines expected (for freq.-domain processing)
    nExpectedFreqLines = int(N / 2 + 1)

    if broadcastDomain == 'wholeChunk_fd':
        zk = np.empty((nExpectedFreqLines, 0), dtype=complex)
    elif broadcastDomain == 'wholeChunk_td':
        zk = np.empty((N, 0), dtype=float)       # initialize compressed signal matrix ($\mathbf{z}_{-k}$ in [1]'s notation)
    elif broadcastDomain == 'fewSamples_td':
        zk = np.empty((N, 0), dtype=float)       # initialize compressed signal matrix ($\mathbf{z}_{-k}$ in [1]'s notation)
        # raise ValueError('[NOT YET IMPLEMENTED]')     # TODO: do that
    bufferFlags = np.zeros(len(neighs))    # flags (positive: +1; negative: -1; or none: 0) -- indicate buffer over- or under-flow

    for idxq in range(len(neighs)):
        
        Bq = len(zBufferk[idxq])  # buffer size for neighbour node `q`

        # Frequency-domain chunks broadcasting (naïve DANSE, simplest to visualize/implement, but inefficient)
        if broadcastDomain == 'wholeChunk_fd':
            if Bq == 0:     # under-flow with chunk-wise FD broadcasting
                if ik == 0:
                    raise ValueError(f'Unexpected buffer under-flow at first iteration in FD-broadcasting.')
                else:
                    print('FD BROADCASTING: BUFFER UNDERFLOW')
                    zCurrBuffer = zPreviousk[:, idxq]  # re-use previous z...
            elif Bq == 2 * nExpectedFreqLines:  # over-flow with chunk-wise FD broadcasting
                print('FD BROADCASTING: BUFFER OVERFLOW')
                zCurrBuffer = zBufferk[idxq][:Ns]
            elif Bq == nExpectedFreqLines:     # correctly filled-in buffer, no under-/over-flow
                zCurrBuffer = zBufferk[idxq]
            else:
                raise ValueError(f'Unexpected buffer over-/under-flow in FD-broadcasting: {Bq} samples instead of {nExpectedFreqLines} expected.')
        
        # Time-domain chunks broadcasting (naïve DANSE, simplest to visualize/implement, but inefficient)
        elif broadcastDomain == 'wholeChunk_td':
            if Bq == Ns:
                # All good, no under-/over-flows
                if not np.any(zPreviousk):
                    # Not yet any previous buffer -- need to appstart zeros
                    zCurrBuffer = np.concatenate((np.zeros(N - Bq), zBufferk[idxq]))
                else:
                    # Concatenate last `Ns` samples of previous buffer with current buffer
                    zCurrBuffer = np.concatenate((zPreviousk[-Ns:, idxq], zBufferk[idxq]))
            else:
                # Under- or over-flow...
                raise ValueError('[NOT YET IMPLEMENTED]')
                
        elif broadcastDomain == 'fewSamples_td':

            if ik == 0: # first DANSE iteration case -- we are expecting an abnormally full buffer, with an entire DANSE chunk size inside of it
                if Bq == N: 
                    # There is no significant SRO between node `k` and `q`.
                    # Response: node `k` uses all samples in the `q`^th buffer.
                    zCurrBuffer = zBufferk[idxq]
                elif (N - Bq) % L == 0 and Bq < N:
                    # Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.
                    # Interpretation: Node `q` samples slower than node `k`. 
                    # Response: ...
                    raise ValueError('Unexpected edge case: Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.')
                elif (N - Bq) % L == 0 and Bq > N:
                    # Node `q` has already transmitted too much data to node `k`.
                    # Interpretation: Node `q` samples faster than node `k`.
                    # Response: node `k` raises a positive flag and uses the last `frameSize` samples from the `q`^th buffer.
                    bufferFlags[idxq] = +1 * int((N - Bq) / L)      # raise negative flag
                    zCurrBuffer = zBufferk[idxq][-N:]

            else:   # not the first DANSE iteration -- we are expecting a normally full buffer, with a DANSE chunk size considering overlap
                if Bq == Ns:             # case 1: no broadcast frame mismatch between node `k` and node `q`
                    pass
                elif (Ns - Bq) % L == 0 and Bq < Ns:       # case 2: negative broadcast frame mismatch between node `k` and node `q`
                    nMissingBroadcasts = int(np.abs((Ns - Bq) / L))
                    print(f'[b-] Buffer underflow at current node`s B_{neighs[idxq]+1} buffer | -{nMissingBroadcasts} broadcast(s)')
                    bufferFlags[idxq] = -1 * nMissingBroadcasts      # raise negative flag
                elif (Ns - Bq) % L == 0 and Bq > Ns:       # case 3: positive broadcast frame mismatch between node `k` and node `q`
                    nExtraBroadcasts = int(np.abs((Ns - Bq) / L))
                    print(f'[b+] Buffer overflow at current node`s B_{neighs[idxq]+1} buffer | +{nExtraBroadcasts} broadcasts(s)')
                    bufferFlags[idxq] = +1 * nExtraBroadcasts       # raise positive flag
                else:
                    if (Ns - Bq) % L != 0 and np.abs(ik - lastExpectedIter) < 10:
                        print('[b!] This is the last iteration -- not enough samples anymore due to cumulated SROs effect, skip update.')
                        bufferFlags[idxq] = np.NaN   # raise "end of signal" flag
                    else:
                        raise ValueError(f'Unexpected buffer size ({Bq} samples, with L={L} and N={Ns}) for neighbor node q={neighs[idxq]+1}.')
                # Build current buffer
                if N - Bq > 0:
                    zCurrBuffer = np.concatenate((zPreviousk[-(N - Bq):, idxq], zBufferk[idxq]), axis=0)
                else:   # edge case: no overlap between consecutive frames
                    zCurrBuffer = zBufferk[idxq]

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


def fill_buffers_freq_domain(k, neighbourNodes, zBuffer, zLocalK):
    """Fills neighbors nodes' buffers, using frequency domain data.
    Data comes from compression via function `danse_compression_freq_domain`.
    
        Parameters
    ----------
    k : int
        Current node index.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (complex)
        Compressed signals buffers for each node and its neighbours.
    zLocal : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [N x 1] np.ndarrays (complex)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Loop over neighbors of `k`
    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]                 # network-wide index of node `q` (one of node `k`'s neighbors)
        idxKforNeighborQ = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighborQ = idxKforNeighborQ[0]      # node `k`'s "neighbor index", from node `q`'s perspective
        # Fill in neighbor's buffer
        zBuffer[q][idxKforNeighborQ] = zLocalK
        
    return zBuffer


def fill_buffers_td_few_samples(k, neighbourNodes, lk, zBuffer, zLocalK, L):
    """Fill in buffers -- simulating broadcast of compressed signals
    from one node (`k`) to its neighbours.
    
    Parameters
    ----------
    k : int
        Current node index.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    lk : [numNodes x 1] list (int)
        Broadcast index per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (float)
        Compressed signals buffers for each node and its neighbours.
    zLocalK : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.
    L : int
        Broadcast chunk length.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (float)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Check for sample-per-sample broadcast scheme (`L==1`)
    if isinstance(zLocalK, float):
        zLocalK = [zLocalK]
        if L != 1:
            raise ValueError(f'Incoherence: float `zLocalK` but L = {L}')

    # Loop over neighbors of node `k`
    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]                 # network-wide index of node `q` (one of node `k`'s neighbors)
        idxKforNeighborQ = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighborQ = idxKforNeighborQ[0]      # node `k`'s "neighbor index", from node `q`'s perspective
        # Only broadcast the `L` last samples of local compressed signals
        zBuffer[q][idxKforNeighborQ] = np.concatenate((zBuffer[q][idxKforNeighborQ], zLocalK[-L:]), axis=0)
        
    return zBuffer


def fill_buffers_td_whole_chunk(k, neighbourNodes, zBuffer, zLocalK):
    """Fill in buffers -- simulating broadcast of compressed signals
    from one node (`k`) to its neighbours, in the time-domain, chunk-wise
    broadcasting scheme.
    
    Parameters
    ----------
    k : int
        Current node index.
    N : int
        Buffer size.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of empty (`len()==0`) np.ndarrays
        Compressed signals buffers for each node and its neighbours.
    zLocalK : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (float)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Loop over neighbors of node `k`
    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]                 # network-wide index of node `q` (one of node `k`'s neighbors)
        idxKforNeighborQ = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighborQ = idxKforNeighborQ[0]      # node `k`'s "neighbor index", from node `q`'s perspective
        # Fill in neighbor's buffer
        zBuffer[q][idxKforNeighborQ] = zLocalK
        
    return zBuffer


def events_parser(events, startUpdates, printouts=False):
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


def get_events_matrix(timeInstants, N, Ns, L, bd):
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
    bd : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.

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
        precision = int(np.ceil(np.abs(np.log10(np.mean(deltas) / 1e7))))  # allowing computer precision errors down to 1e-4*mean delta.
        if len(np.unique(np.round(deltas, precision))) > 1:
            raise ValueError(f'[NOT IMPLEMENTED] Clock jitter detected: {len(np.unique(deltas))} different sample intervals detected for node {k+1}.')
        fs[k] = np.round(1 / np.unique(np.round(deltas, precision))[0], 3)  # np.round(): not going below 1PPM precision for typical fs >= 8 kHz

    # Total signal duration [s] per node (after truncation during signal generation)
    Ttot = timeInstants[-1, :]
    
    # Get expected DANSE update instants
    numUpdatesInTtot = np.floor(Ttot * fs / Ns)   # expected number of DANSE update per node over total signal length
    updateInstants = [np.arange(np.ceil(N / Ns), int(numUpdatesInTtot[k])) * Ns/fs[k] for k in range(nNodes)]  # expected DANSE update instants
    #                               ^ note that we only start updating when we have enough samples
    
    numBroadcastsInTtot = np.floor(Ttot * fs / L)   # expected number of broadcasts per node over total signal length
    # Get expected broadcast instants
    if 'wholeChunk' in bd:
        broadcastInstants = [np.arange(N/L, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]
        #                              ^ note that we only start broadcasting when we have enough samples to perform compression
    elif 'fewSamples' in bd:
        broadcastInstants = [np.arange(1, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]
        #                              ^ note that we start broadcasting sooner: when we have `L` samples, enough for linear convolution

    # Ensure that all nodes have broadcasted enough times to perform any update
    minWaitBeforeUpdate = np.amax([v[N // L - 1] for v in broadcastInstants])
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


def broadcast(t, k, fs, L, yk, w, n, neighbourNodes, lk, zBuffer,
            broadcastDomain,
            winWOLAanalysis,
            winWOLAsynthesis,
            winShift,
            previousTDfilterUpdate,
            updateTDfilterEvery, 
            wIRprevious,
            zTDpreviousFrame):
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
    yk : [N + L - 1 x nSensors] np.ndarray of floats
        Local sensor data chunk from node `k` at current time stamp.
    w : [N/2 x nSensors] np.ndarray of complex
        Filter coefficients used for compression.
    n : int
        FFT length.
    neighbourNodes : [nNodes x 1] list of [nNeighbours[n] x 1] lists of ints
        Network indices of neighbours, per node.
    lk : [nNodes x 1] list of ints
        Broadcast index per node.
    zBuffer : [nNodes x 1] list of [Nneighbors x 1] lists of np.ndarrays (float)
        For each node, buffers at previous iteration.
    broadcastDomain : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    winWOLAanalysis : [`n` x 1] np.ndarray (float)
        WOLA analysis window (time-domain to WOLA-domain).
    winWOLAsynthesis : [`n` x 1] np.ndarray (float)
        WOLA synthesis window (WOLA-domain to time-domain).
    winShift : int
        Sample shift between adjacent windows.
    previousTDfilterUpdate : float
        Instant at which the previous time-domain filters update occurred [s]
        (only used if `broadcastDomain=='fewSamples_td'`).
    updateTDfilterEvery : float
        Duration of pause between two consecutive time-domain filter updates.
        (only used if `broadcastDomain=='fewSamples_td'`).
    wIRprevious : [2*`n` - 1 x nSensors] np.ndarray (float)
        Time-domain filters used for the previous broadcast.
        (only used if `broadcastDomain=='fewSamples_td'`).
    zTDpreviousFrame : [`L` x 1] np.ndarray (complex)
        Previously broadcasted frame for current node (`k`).
        (only used if `broadcastDomain=='wholeChunk_td'`).
    
    Returns
    -------
    zBuffer : [nNodes x 1] list of [Nneighbors x 1] lists of np.ndarrays (float or complex)
        For each node, buffers at current iteration.
        # TODO: other outputs
    """
    # Check inputs
    if np.round(t * fs) != np.round(t * fs, 9):
        raise ValueError('Unexpected time instant: does not correspond to a specific sample.')

    if len(yk) < n:
        print('Cannot perform compression: not enough local signals samples.')

    else:
        if broadcastDomain == 'wholeChunk_fd':
            # Frequency-domain chunk-wise broadcasting
            zLocalHat, _ = danse_compression_whole_chunk(yk, w,
                                        winWOLAanalysis, winWOLAsynthesis)  # local compressed signals (freq.-domain)

            zBuffer = fill_buffers_freq_domain(k, neighbourNodes, zBuffer, zLocalHat) 

            wIR = None      # chunk-wise broadcasting: no IR filter 
            zLocal = None   # FD BC -- no `zLocal` variable computed
            
        elif broadcastDomain == 'wholeChunk_td':
            # Time-domain chunk-wise broadcasting
            _, zLocal = danse_compression_whole_chunk(yk, w,
                                        winWOLAanalysis, winWOLAsynthesis,
                                        zqPrevious=zTDpreviousFrame)  # local compressed signals (time-domain)

            zBuffer = fill_buffers_td_whole_chunk(k, neighbourNodes, zBuffer, zLocal[:(n // 2)])
            
            wIR = None  # chunk-wise broadcasting: no IR filter 
        
        elif broadcastDomain == 'fewSamples_td':
            # Time-domain broadcasting, `L` samples at a time,
            # via linear-convolution approximation of WOLA filtering process

            # Only update filter every so often
            updateBroadcastFilter = False
            if np.abs(t - previousTDfilterUpdate) >= updateTDfilterEvery:
                updateBroadcastFilter = True
                previousTDfilterUpdate = t

            zLocal, wIR = danse_compression_few_samples(yk, w, n, L,
                                            wIRprevious,
                                            winWOLAanalysis, winWOLAsynthesis, winShift,
                                            updateBroadcastFilter=updateBroadcastFilter)  # local compressed signals

            zBuffer = fill_buffers_td_few_samples(k, neighbourNodes, lk, zBuffer, zLocal, L)

        # if lk[k] > 1000 and k == 1:
        #     print(yk[-1])
        #     print(t)
        #     stop = 1

        lk[k] += 1  # increment local broadcast index

    return zBuffer, wIR, previousTDfilterUpdate, zLocal


def spatial_covariance_matrix_update(y, Ryy, Rnn, beta, vad):
    """Helper function: performs the spatial covariance matrices updates.
    
    Parameters
    ----------
    y : [N x M] np.ndarray (real or complex)
        Current input data chunk (if complex: in the frequency domain).
    Ryy : [N x M x M] np.ndarray (real or complex)
        Previous Ryy matrices (for each time frame /or/ each frequency line).
    Rnn : [N x M x M] np.ndarray (real or complex)
        Previous Rnn matrices (for each time frame /or/ each frequency line).
    beta : float (0 <= beta <= 1)
        Exponential averaging forgetting factor.
    vad : bool
        If True (=1), Ryy is updated. Otherwise, Rnn is updated.
    
    Returns
    -------
    Ryy : [N x M x M] np.ndarray (real or complex)
        New Ryy matrices (for each time frame /or/ each frequency line).
    Rnn : [N x M x M] np.ndarray (real or complex)
        New Rnn matrices (for each time frame /or/ each frequency line).
    yyH : [N x M x M] np.ndarray (real or complex)
        Instantaneous correlation outer product.
    """
    yyH = np.einsum('ij,ik->ijk', y, y.conj())

    if vad:
        Ryy = beta * Ryy + (1 - beta) * yyH  # update signal + noise matrix
    else:     
        Rnn = beta * Rnn + (1 - beta) * yyH  # update noise-only matrix

    return Ryy, Rnn, yyH


def local_chunk_for_broadcast(y, t, fs, bd, N, k, lk):
    """Extract correct chunk of local signals for broadcasting.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
        Time-domain locally recorded signal (at `Mk` sensors).
    t : float
        Current time instant [s].
    fs : int or float
        Transmitting node's sampling frequency [Hz].
    bd : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    N : int
        Frame size (= FFT size in DANSE).

    Returns
    -------
    chunk : [N x Mk] np.ndarray (float)
        Time chunk of local sensor signals.
    """

    idxEnd = int(np.floor(np.round(t * fs, 5)))  # np.round() used to avoid issues with previous rounding/precision errors (see Word journal week 32, THU, 2022)
    # if bd in ['wholeChunk_fd', 'wholeChunk_td']:
    idxBeg = np.amax([idxEnd - N, 0])   # don't go into negative sample indices!
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed
    if idxEnd - idxBeg < N:
        chunk = np.concatenate((np.zeros((N - chunk.shape[0], chunk.shape[1])), chunk))
    # elif bd == 'fewSamples_td':
    #     raise ValueError('[NOT YET IMPLEMENTED]')   # TODO: do that

    return chunk


def local_chunk_for_update(y, t, fs, bd, N, Ns, L):
    """Extract correct chunk of local signals for DANSE updates.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
        Time-domain locally recorded signal (at `Mk` sensors).
    t : float
        Current time instant [s].
    fs : int or float
        Transmitting node's sampling frequency [Hz].
    bd : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    N : int
        Frame size (= FFT size in DANSE).
    L : int
        Broadcast length.

    Returns
    -------
    chunk : [N x Mk] np.ndarray (float)
        Time chunk of local sensor signals.
    idxBeg : int
        Start index of chunk (w.r.t. `y`).
    idxEnd : int
        End index of chunk (w.r.t. `y`).
    """

    # Broadcast scheme: block-wise, in freq.-domain
    # <or> Broadcast scheme: few samples at a time, in time-domain
    if bd in ['wholeChunk_fd', 'fewSamples_td']:
        idxEnd = int(np.floor(np.round(t * fs, 5)))
    # Broadcast scheme: block-wise, in time-domain
    elif bd == 'wholeChunk_td':
        idxEnd = int(np.floor(np.round(t * fs, 5))) - Ns     # `Ns` samples delay due to time-domain WOLA

    idxBeg = np.amax([idxEnd - N, 0])       # don't go into negative sample indices!
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed (occurs at algorithm's startup)
    if idxEnd - idxBeg < N:
        chunk = np.concatenate((np.zeros((N - chunk.shape[0], chunk.shape[1])), chunk))

    return chunk, idxBeg, idxEnd


from doctest import master
from tracemalloc import start
from turtle import numinput
import numpy as np
import scipy, time, datetime
import scipy.signal
from numba import njit
from . import classes
import matplotlib.pyplot as plt


"""
References:
 - [1]: Bertrand, Alexander, and Marc Moonen. "Distributed adaptive node-specific 
        signal estimation in fully connected sensor networks—Part I: 
        Sequential node updating." IEEE Transactions on Signal Processing 
        58.10 (2010): 5277-5291.
 - [2]: Bertrand, Alexander, and Marc Moonen. "Distributed adaptive node-specific 
        signal estimation in fully connected sensor networks—Part II: 
        Simultaneous and Asynchronous Node Updating." IEEE Transactions on 
        Signal Processing 58.10 (2010): 5292-5305.
 - [3]: Ruiz, Santiago, Toon Van Waterschoot, and Marc Moonen. "Distributed 
        combined acoustic echo cancellation and noise reduction in wireless 
        acoustic sensor and actuator networks." IEEE/ACM Transactions on Audio, 
        Speech, and Language Processing (2022).
"""


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
    win = settings.stftWin[:, np.newaxis]

    # Define useful quantities
    frameSize = settings.stftWinLength
    nNewSamplesPerFrame = settings.stftEffectiveFrameLen
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



def danse_sequential(yin, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD):
    """Wrapper for Sequential-Node-Updating DANSE [1].

    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray (real)
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray (binary/boolean)
        Voice Activity Detector.

    Returns
    -------
    d : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes.
    """

    # Initialization (extracting useful quantities)
    rng, win, frameSize, nNewSamplesPerFrame, numIterations, _, neighbourNodes = danse_init(yin, settings, asc)
    fftscale = 1.0 / win.sum()
    ifftscale = win.sum()

    # Compute frame-wise VAD
    oVADframes = np.zeros(numIterations, dtype=bool)
    for i in range(numIterations):
        VADinFrame = oVAD[i * nNewSamplesPerFrame : i * nNewSamplesPerFrame + frameSize]
        nZeros = sum(VADinFrame == 0)
        oVADframes[i] = nZeros <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

    # Initialize matrices
    y = []              # local sensor observations
    w = []              # filter coefficients
    z = []              # compressed observations from neighboring nodes
    ytilde = []         # local + remote observations 
    ytilde_hat = []     # local + remote observations (freq. domain)
    Rnn = []            # per-node list of noise-only autocorrelation matrices
    ryd = []            # per-node list of cross-correlation vectors
    Ryy = []            # per-node list of signal+noise autocorrelation matrices
    srosEst = []        # per-node list of SROs estimates
    dimYTilde = np.zeros(asc.numNodes, dtype=int)  # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    numFreqLines = int(frameSize / 2 + 1)
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        w.append(rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k])) +\
            1j * rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k])))
        y.append(np.zeros((frameSize, numIterations, sum(asc.sensorToNodeTags == k + 1))))
        z.append(np.zeros((frameSize, numIterations, len(neighbourNodes[k])), dtype=complex))
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytilde_hat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        #
        # slice = np.zeros((dimYTilde[k], dimYTilde[k]), dtype=complex)       # single autocorrelation matrix init (zeros)
        slice = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnn.append(np.tile(slice, (numFreqLines, 1, 1)))                    # noise only
        Ryy.append(np.tile(slice, (numFreqLines, 1, 1)))                    # speech + noise
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        srosEst.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))    # SROs estimates
    # Filter coefficients update flag for each frequency
    startUpdates = np.array([False for _ in range(numFreqLines)])
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
    # Desired signal estimate [frames x frequencies x nodes]
    d = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)

    u = 1       # init updating node number
    for i in range(numIterations):
        print(f'Sequential DANSE -- Iteration {i + 1}/{numIterations} -- Updating node {u}...')
        for k in range(asc.numNodes):
            # Select time samples for current frame
            idxSamplesFrame = np.arange(i * nNewSamplesPerFrame, i * nNewSamplesPerFrame + frameSize)
            # Collect local observations
            y[k][:, i, :] = yin[idxSamplesFrame][:, asc.sensorToNodeTags == k + 1]
            # Build complete observation vectors
            ytilde_curr = y[k][:, i, :]
            for idx, q in enumerate(neighbourNodes[k]):
                # Identify remote sensors
                yq = yin[idxSamplesFrame][:, asc.sensorToNodeTags == q + 1]
                # Go to frequency domain
                yq_hat = fftscale * np.fft.fft(yq * win, frameSize, axis=0)  # np.fft.fft: frequencies ordered from DC to Nyquist, then -Nyquist to -DC (https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)
                # Keep only positive frequencies
                yq_hat = yq_hat[:numFreqLines]
                # Compress using filter coefficients
                zq_hat = np.einsum('ij,ij->i', w[q][:, i, :yq_hat.shape[1]].conj(), yq_hat)  # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                # Format for IFFT 
                # zq_hat = np.concatenate((zq_hat, np.flip(zq_hat.conj())), axis=0)
                zq_hat[0] = zq_hat[0].real      # Set DC to real value
                zq_hat[-1] = zq_hat[-1].real    # Set Nyquist to real value
                zq_hat = np.concatenate((zq_hat, np.flip(zq_hat[:-1].conj())[:-1]))
                # Back to time-domain
                z[k][:, i, idx] = ifftscale * np.fft.ifft(zq_hat, frameSize)
                zq = z[k][:, i, idx][:, np.newaxis]  # necessary for np.concatenate
                ytilde_curr = np.concatenate((ytilde_curr, zq), axis=1)

            ytilde[k][:, i, :] = ytilde_curr    # <-- complete observation vectors
            # Go to frequency domain
            ytilde_hat_curr = fftscale * np.fft.fft(ytilde[k][:, i, :] * win, frameSize, axis=0)
            # Keep only positive frequencies
            ytilde_hat[k][:, i, :] = ytilde_hat_curr[:numFreqLines, :]
            
            # Count autocorrelation matrices updates
            if oVADframes[i]:
                Ryy[k] = settings.expAvgBeta * Ryy[k] + (1 - settings.expAvgBeta) *\
                    np.einsum('ij,ik->ijk', ytilde_hat[k][:, i, :], ytilde_hat[k][:, i, :].conj())  # update signal + noise matrix
                numUpdatesRyy[k] += 1
            else:     
                Rnn[k] = settings.expAvgBeta * Rnn[k] + (1 - settings.expAvgBeta) *\
                    np.einsum('ij,ik->ijk', ytilde_hat[k][:, i, :], ytilde_hat[k][:, i, :].conj())  # update signal + noise matrix
                numUpdatesRnn[k] += 1

            # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
            if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
                startUpdates[k] = True

            if startUpdates[k] and u == k + 1:
                if settings.performGEVD:    
                    w[k][:, i + 1, :], Qmat = perform_gevd_noforloop(Ryy[k], Rnn[k], settings.GEVDrank, settings.referenceSensor)
                else:
                    raise ValueError('Not yet implemented')     # TODO
        #             # Reference sensor selection vector
        #             Evect = np.zeros((dimYTilde[k],))
        #             Evect[settings.referenceSensor] = 1
        #             # Cross-correlation matrix update 
        #             ryd[k][kappa, :] = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect
        #             # Update node-specific parameters of node k
        #             w[k][kappa, i + 1, :] = np.linalg.inv(Ryy[k][kappa, :, :]) @ ryd[k][kappa, :]
            else:
                # Do not update the filter coefficients
                w[k][:, i + 1, :] = w[k][:, i, :]

            # Compute desired signal estimate
            d[:, i, k] = np.einsum('ij,ij->i', w[k][:, i + 1, :].conj(), ytilde_hat[k][:, i, :])  # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850

        # Update updating node index
        u = (u % asc.numNodes) + 1

    return d


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
    zq_hat[0] = zq_hat[0].real      # Set DC to real value
    zq_hat[-1] = zq_hat[-1].real    # Set Nyquist to real value
    zq_hat = np.concatenate((zq_hat, np.flip(zq_hat[:-1].conj())[:-1]))
    # Back to time-domain
    zq = ifftscale * np.fft.ifft(zq_hat, len(win))
    # zq = zq[:, np.newaxis]

    return zq


def danse_simultaneous(yin, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD, timeInstants):
    """Wrapper for Simultaneous-Node-Updating DANSE (rs-DANSE) [2].

    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray (real)
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray (binary/boolean)
        Voice Activity Detector.
    timeInstants : [Nt x Nn] np.ndarray 
        Time instants corresponding to the samples of each of the Nn nodes in the network. 

    Returns
    -------
    d : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes -- using full-observations vectors (also data coming from neighbors).
    dLocal : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes -- using only local observations (not data coming from neighbors).
    """
    
    # Initialization (extracting useful quantities)
    rng, win, frameSize, nExpectedNewSamplesPerFrame, numIterations, _, neighbourNodes = danse_init(yin, settings, asc)
    fftscale = 1 / win.sum()

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, settings.referenceSensor]     # reference clock


    # ---------------------- Arrays initialization ----------------------
    lk = np.zeros(asc.numNodes, dtype=int)                      # node-specific broadcast index
    i = np.zeros(asc.numNodes, dtype=int)                       # !node-specific! DANSE iteration index
    nReadyForBroadcast = np.zeros(asc.numNodes, dtype=int)      # node-specific number of samples lined up for broadcast 
    nNewLocalSamples = np.zeros(asc.numNodes, dtype=int)        # node-specific number of new local samples since the last DANSE iteration 
    lastSampleIdx = np.full(shape=(asc.numNodes,), fill_value=-1)           # last sample index lined up for broadcast
    #
    wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
    wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
    Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
    Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
    Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
    Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
    ryd = []                                        # cross-correlation between observations and estimations
    ytilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                 # local full observation vectors, frequency-domain
    z = []                                          # current-iteration compressed signals used in DANSE update
    zPrevious = []                                  # previous-iteration compressed signals used in DANSE update
    zPreviousBuffer = []                            # previous-iteration "incoming signals from other nodes" buffer
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
    numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
    for k in range(asc.numNodes):
        dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
        dimYTilde[k] = dimYLocal[k] + len(neighbourNodes[k])
        wLocal.append(settings.initialWeightsAmplitude * (rng.random(size=(numFreqLines, numIterations + 1, dimYLocal[k])) +\
            1j * rng.random(size=(numFreqLines, numIterations + 1, dimYLocal[k]))))
        wTilde.append(settings.initialWeightsAmplitude * (rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k])) +\
            1j * rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k]))))
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytildeHat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        #
        sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
        Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
        Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
        Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros(len(neighbourNodes[k])))    # init all buffer flags at 0 (assuming no over- or under-flow)
        #
        z.append(np.empty((frameSize, 0), dtype=float))
        zPrevious.append(np.empty((frameSize, 0), dtype=float))
        zPreviousBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
        zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
    # Desired signal estimate [frames x frequencies x nodes]
    d = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
    dLocal = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)

    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
    # Important instants
    idxBroadcasts = [[] for _ in range(asc.numNodes)]
    idxUpdates = [[] for _ in range(asc.numNodes)]
    # Booleans
    startedCompression = np.full(shape=(asc.numNodes,), fill_value=False)   # when True, indicates that the compression of local observations has started
    broadcastSignals = np.full(shape=(asc.numNodes,), fill_value=False)     # when True, broadcast signals to neighbor nodes
    startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
    # ------------------------------------------------------------------

    # # Initialize figure object
    # fig, axes = plt.subplots(asc.numNodes,1)
    # # to run GUI event loop
    # plt.ion()
    # plt.show()

    t0 = time.perf_counter()    # global timing
    for idxt, tMaster in enumerate(masterClock):    # loop over master clock instants
        for k in range(asc.numNodes):               # loop over nodes
            # ============== Pre-DANSE time processing ============== 
            if k == settings.referenceSensor:
                # Special processing for reference sensor
                nReadyForBroadcast[k] += 1
                nNewLocalSamples[k] += 1
                idxCurrSample = idxt                    # number of samples accumulated at master-time `t` at node `k`
            else:
                passedInstants = timeInstants[timeInstants[:, k] <= tMaster, k]     # list of passed time stamps at node `k`
                idxCurrSample = len(passedInstants) - 1     # number of samples accumulated at master-time `t` at node `k`
                if lastSampleIdx[k] < idxCurrSample:
                    nReadyForBroadcast[k] += idxCurrSample - lastSampleIdx[k]       # can be more than 1 when SROs are present...
                    nNewLocalSamples[k] += idxCurrSample - lastSampleIdx[k]         # can be more than 1 when SROs are present...
                else:
                    pass    # if there is no new sample at node `k`, don't increment the node-specific numbers of samples
            lastSampleIdx[k] = idxCurrSample    # record current sample index at node `k`
            
            # Raise or do not raise flag for compressing + broadcasting
            if not startedCompression[k]:   # first broadcast -- need `frameSize` sample to perform compression in freq.-domain
                if nReadyForBroadcast[k] == frameSize:
                    broadcastSignals[k] = True
                    startedCompression[k] = True
            elif nReadyForBroadcast[k] == settings.broadcastLength:     # not the first broadcast -- need `settings.broadcastLength` new samples to perform compression in freq.-domain
                broadcastSignals[k] = True
            else:   # not time to broadcast
                broadcastSignals[k] = False
        
            if broadcastSignals[k]: 
                if k == settings.referenceSensor and lk[k] % 100 == 0:
                    print(f'tMaster = {np.round(tMaster, 4)}s(/{np.round(masterClock[-1], 2)}s): Ref. node {k+1}`s {lk[k]}^th broadcast to its neighbors')
                idxBroadcasts[k].append(idxt)
                # ~~~~~~~~~~~~~~ Time to broadcast! ~~~~~~~~~~~~~~
                # Extract current data chunk
                yinCurr = yin[(lastSampleIdx[k] - frameSize + 1):(lastSampleIdx[k] + 1), asc.sensorToNodeTags == k+1]
                zLocal = danse_compression(yinCurr, wTilde[k][:, i[k], :yinCurr.shape[-1]], win)        # local compressed signals
                for idxq in range(len(neighbourNodes[k])):
                    q = neighbourNodes[k][idxq]     # actual index of neighbor (from whole WASN perspective)
                    idxKforNeighbor = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
                    idxKforNeighbor = idxKforNeighbor[0]
                    # Fill in neighbors' buffers with the L = `settings.broadcastLength` last samples of local compressed signals
                    if lk[k] == 0:
                        # Broadcast the `nExpectedNewSamplesPerFrame` last samples of compressed signal (1st broadcast period, no "old" samples)
                        zBufferCurr = np.concatenate((zBuffer[q][idxKforNeighbor], zLocal), axis=0)
                    else:
                        # Only broadcast the L = `settings.broadcastLength` last samples of local compressed signals
                        zBufferCurr = np.concatenate((zBuffer[q][idxKforNeighbor], zLocal[-settings.broadcastLength:]), axis=0)
                    zBuffer[q][idxKforNeighbor] = zBufferCurr   # necessary for `np.concatenate()` afterwards
                nReadyForBroadcast[k] = 0               # reset number of samples ready for broadcast
                lk[k] += 1                              # increment broadcast index
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                pass    # no broadcast at this instant
            # ========================================================

        # Must reset the `k`-loop here to ensure that all nodes have broadcasted what they had to broadcast
        for k in range(asc.numNodes):               # loop over nodes
            if nNewLocalSamples[k] == nExpectedNewSamplesPerFrame:  # TODO: consider the case where there is an SRO and nNewLocalSamples[k] >= nExpectedNewSamplesPerFrame
                if lastSampleIdx[k] < frameSize - 1:
                    nNewLocalSamples[k] = 0     # if there are not enough samples to perform the `frameSize`-points FFT, we reset the counter (first iteration, when the node has just arrived in the WASN)
                else:
                    # ~~~~~~~~~~~~~~ Time to update the filter coefficients! ~~~~~~~~~~~~~~
                    t0update = time.perf_counter()
                    idxUpdates[k].append(idxt)

                    # Gather local observation vector
                    idxStartChunk = lastSampleIdx[k] - frameSize + 1
                    idxEndChunk = lastSampleIdx[k] + 1
                    yLocalCurr = yin[idxStartChunk:idxEndChunk, asc.sensorToNodeTags == k+1]  # local sensor observations ("$\mathbf{y}_k$" in [1])
                    
                    # Process buffers
                    for idxq in range(len(neighbourNodes[k])):
                        Bq = len(zBuffer[k][idxq])  # buffer size
                        if i[k] == 0:
                            if Bq == frameSize: 
                                zCurrBuffer = zBuffer[k][idxq]
                            else:
                                # There is an SRO between node k and q...
                                if Bq == 0:     # case I
                                    # Node q has not yet transmitted any data to node k, while node k already has reached its first update instant
                                    # Response from node k: skip this update. All buffers should have at least some samples inside before updating.
                                    skipUpdate = True
                                # raise ValueError('Scenario not yet considered / implemented (1st iteration, Bq/neq N).')    # TODO
                        else:
                            if Bq == nExpectedNewSamplesPerFrame:                                 # case 1: no broadcast frame mismatch between node `k` and node `q`
                                zCurrBuffer = np.concatenate((zPrevious[k][-(frameSize - Bq):, idxq], zBuffer[k][idxq]), axis=0)
                            elif Bq == frameSize - settings.broadcastLength:     # case 2: positive broadcast frame mismatch between node `k` and node `q`
                                # TODO
                                print(f'Buffer underflow (node k={k+1}`s B_{neighbourNodes[k][idxq]+1} buffer).')
                                if i[k] == 0:
                                    raise ValueError(f'Buffer underflow occured at first DANSE iteration (node k={k+1}`s B_{neighbourNodes[k][idxq]+1} buffer).')
                                bufferFlags[k][idxq] = -1      # raise negative flag
                                # Use previous iteration's buffer
                                zUnderFlow = np.concatenate((zPreviousBuffer[k][idxq][-settings.broadcastLength:, np.newaxis], zBuffer[k][idxq][:, np.newaxis]), axis=1)
                                z = np.concatenate((z, zUnderFlow), axis=0)
                            elif Bq == frameSize + settings.broadcastLength:     # case 3: negative broadcast frame mismatch between node `k` and node `q`
                                # TODO
                                print(f'Buffer overflow (node k={k+1}`s B_{neighbourNodes[k][idxq]+1} buffer).')
                                bufferFlags[k][idxq] = 1       # raise positive flag
                                # Discard L = `settings.broadcastLength` oldest samples in buffer
                                z = np.concatenate((z, zBuffer[k][idxq][settings.broadcastLength:, np.newaxis]), axis=1)
                            else:
                                raise ValueError(f'Node k={k+1}: Unexpected buffer size for neighbor node q={neighbourNodes[k][idxq]+1}.')
                        # Stack compressed signals
                        z[k] = np.concatenate((z[k], zCurrBuffer[:, np.newaxis]), axis=1)
                    
                    # Build full available observation vector
                    yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                    ytilde[k][:, i[k], :] = yTildeCurr
                    # Go to frequency domain
                    ytildeHatCurr = fftscale * np.fft.fft(ytilde[k][:, i[k], :] * win, frameSize, axis=0)
                    # Local observations only
                    yHat = ytildeHatCurr[:numFreqLines, :dimYLocal[k]]
                    # Keep only positive frequencies
                    ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]
                    
                    # Compute VAD
                    VADinFrame = oVAD[idxStartChunk:idxEndChunk]
                    oVADframes[i[k]] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

                    # Count autocorrelation matrices updates
                    yyHtilde = np.einsum('ij,ik->ijk', ytildeHat[k][:, i[k], :], ytildeHat[k][:, i[k], :].conj())
                    yyHlocal = np.einsum('ij,ik->ijk', yHat, yHat.conj())
                    if oVADframes[i[k]]:
                        Ryytilde[k] = settings.expAvgBeta * Ryytilde[k] + (1 - settings.expAvgBeta) * yyHtilde  # update WIDE signal + noise matrix
                        Ryylocal[k] = settings.expAvgBeta * Ryylocal[k] + (1 - settings.expAvgBeta) * yyHlocal  # update LOCAL signal + noise matrix
                        numUpdatesRyy[k] += 1
                    else:     
                        Rnntilde[k] = settings.expAvgBeta * Rnntilde[k] + (1 - settings.expAvgBeta) * yyHtilde  # update WIDE noise-only matrix
                        Rnnlocal[k] = settings.expAvgBeta * Rnnlocal[k] + (1 - settings.expAvgBeta) * yyHlocal  # update LOCAL noise-only matrix
                        numUpdatesRnn[k] += 1

                    # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
                    if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
                        startUpdates[k] = True

                    if startUpdates[k]:
                        # No `for`-loop versions
                        if settings.performGEVD:    # GEVD update
                            wTilde[k][:, i[k] + 1, :], _ = perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
                            wLocal[k][:, i[k] + 1, :], _ = perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)
                        else:                       # regular update (no GEVD)
                            raise ValueError('Not yet implemented')     # TODO
                    else:
                        # Do not update the filter coefficients
                        wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                        wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]

                    # Compute desired signal estimate
                    d[:, i[k], k] = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                    dLocal[:, i[k], k] = np.einsum('ij,ij->i', wLocal[k][:, i[k] + 1, :].conj(), yHat)   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850

                    # Reset counters
                    nNewLocalSamples[k] = 0
                    # Reset z vectors
                    zPrevious[k] = z[k]
                    z[k] = np.empty((frameSize, 0), dtype=float)
                    # Update buffers status for node `k`
                    zPreviousBuffer[k] = zBuffer[k]                                         # save previous buffer
                    zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]      # reset current buffer
                    #
                    print(f'tMaster = {np.round(tMaster, 4)}s(/{np.round(masterClock[-1], 2)}s): Node {k+1} DANSE iteration #{i[k]+1} (took {int(1e3 * (time.perf_counter() - t0update))}ms)')
                    # Increment DANSE iteration index
                    i[k] += 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('\nSimultaneous DANSE processing all done.')
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=time.perf_counter() - t0))}s.')

    stop = 1

    return d, dLocal

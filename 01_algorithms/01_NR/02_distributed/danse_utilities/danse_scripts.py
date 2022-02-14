
from multiprocessing.sharedctypes import Value
import numpy as np
import scipy, copy
import scipy.signal
from . import classes


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


def perform_gevd(Ryy, Rnn, rank=1, refSensorIdx=0):
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
    # Perform generalized eigenvalue decomposition
    sigma, Xmat = scipy.linalg.eigh(Ryy, Rnn)
    Qmat = np.linalg.inv(Xmat.conj().T)
    # Sort eigenvalues in descending order
    idx = np.flip(np.argsort(sigma))
    GEVLs_yy = np.flip(np.sort(sigma))
    Sigma_yy = np.diag(GEVLs_yy)
    Qmat = Qmat[:, idx]
    # Estimate speech covariance matrix
    diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
    diagveig = np.append(diagveig, np.zeros(Sigma_yy.shape[0] - rank))
    # LMMSE weights
    w = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect 
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
        w.append(rng.random(size=(numFreqLines, numIterations, dimYTilde[k])) +\
            1j * rng.random(size=(numFreqLines, numIterations, dimYTilde[k])))
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
    goodAutocorrMatrices = np.array([False for _ in range(numFreqLines)])
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    # Desired signal estimate [frames x frequencies x nodes]
    d = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)

    u = 1       # init updating node number
    for i in range(numIterations):
        print(f'DANSE -- Iteration {i + 1}/{numIterations} -- Updating node #{u}...')
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
                zq_hat = np.einsum('ij,ij->i', w[q][:, i, :yq_hat.shape[1]].conj(), yq_hat)  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850
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
                numUpdatesRyy[k] += 1
            else:     
                numUpdatesRnn[k] += 1
            # Loop over frequency lines
            for kappa in range(numFreqLines):
                # Autocorrelation matrices update -- eq.(46) in ref.[1] /and-or/ eq.(20) in ref.[3].
                if oVADframes[i]:
                    Ryy[k][kappa, :, :] = settings.expAvgBeta * Ryy[k][kappa, :, :] + \
                        (1 - settings.expAvgBeta) * np.outer(ytilde_hat[k][kappa, i, :], ytilde_hat[k][kappa, i, :].conj())  # update signal + noise matrix
                else:
                    Rnn[k][kappa, :, :] = settings.expAvgBeta * Rnn[k][kappa, :, :] + \
                        (1 - settings.expAvgBeta) * np.outer(ytilde_hat[k][kappa, i, :], ytilde_hat[k][kappa, i, :].conj())   # update noise-only matrix

                #Check quality of autocorrelations estimates
                if not goodAutocorrMatrices[kappa]:
                    goodAutocorrMatrices[kappa] = check_autocorr_est(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                            numUpdatesRyy[k], numUpdatesRnn[k], settings.minNumAutocorrUpdates)

                if goodAutocorrMatrices[kappa] and u == k + 1:
                    if settings.performGEVD:
                        w[k][kappa, i, :], Qmat = perform_gevd(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                                                                settings.GEVDrank, settings.referenceSensor)
                    else:
                        # Reference sensor selection vector
                        Evect = np.zeros((dimYTilde[k],))
                        Evect[settings.referenceSensor] = 1
                        # Cross-correlation matrix update 
                        ryd[k][kappa, :] = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect
                        # Update node-specific parameters of node k
                        w[k][kappa, i, :] = np.linalg.inv(Ryy[k][kappa, :, :]) @ ryd[k][kappa, :]
                elif i > 1:
                    # Do not update the filter coefficients
                    w[k][kappa, i, :] = w[k][kappa, i - 1, :]

            # Compute desired signal estimate
            d[:, i, k] = np.einsum('ij,ij->i', w[k][:, i, :].conj(), ytilde_hat[k][:, i, :])  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850

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
    yq_hat = yq_hat[:len(win) / 2]
    # Compress using filter coefficients
    zq_hat = np.einsum('ij,ij->i', w.conj(), yq_hat)  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850
    # Format for IFFT 
    zq_hat[0] = zq_hat[0].real      # Set DC to real value
    zq_hat[-1] = zq_hat[-1].real    # Set Nyquist to real value
    zq_hat = np.concatenate((zq_hat, np.flip(zq_hat[:-1].conj())[:-1]))
    # Back to time-domain
    z = ifftscale * np.fft.ifft(zq_hat, len(win))
    zq = z[:, np.newaxis]  # necessary for np.concatenate

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
        STFT representation of the desired signal at each of the Nn nodes.
    """
    
    # Initialization (extracting useful quantities)
    rng, win, frameSize, _, numIterations, numBroadcasts, neighbourNodes = danse_init(yin, settings, asc)
    fftscale = 1.0 / win.sum()
    ifftscale = win.sum()

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, settings.referenceSensor]     # reference clock

    lk = np.zeros(asc.numNodes)                                         # node-specific broadcast index
    i = 0                                                               # DANSE iteration index
    nSamplesLinedUpForBroadcast = np.zeros(asc.numNodes, dtype=int)     # node-specific number of samples lined up for broadcast 
    nLocalSamplesSinceLastIter = np.zeros(asc.numNodes, dtype=int)      # node-specific number of new local samples since the last DANSE iteration 
    lastSampleIdx = np.zeros(asc.numNodes, dtype=int)                   # last sample index lined up for broadcast
    #
    w = []                                          # filter coefficients
    Ryy = []
    Rnn = []
    ryd = []
    ytilde = []
    ytilde_hat = []
    z = []                                          # compressed signals - time domain (/!\ -- NOT THE SAME as in `danse_sequential`!)
    zBufferEmpty = []                               # empty "incoming signals" buffers
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)
    numFreqLines = int(frameSize / 2 + 1)
    # Filter coefficients update flag for each frequency
    goodAutocorrMatrices = np.array([False for _ in range(numFreqLines)])
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        w.append(rng.random(size=(numFreqLines, numIterations, dimYTilde[k])) +\
            1j * rng.random(size=(numFreqLines, numIterations, dimYTilde[k])))
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytilde_hat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        z.append(np.zeros((frameSize, numBroadcasts)))
        #
        slice = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnn.append(np.tile(slice, (numFreqLines, 1, 1)))                    # noise only
        Ryy.append(np.tile(slice, (numFreqLines, 1, 1)))                    # speech + noise
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros(len(neighbourNodes[k])))    # init all buffer flags at 0 (assuming no over- or under-flow)
        #
        zBufferEmpty.append([])
        for q in neighbourNodes[k]:
            zBufferEmpty[k].append(np.array([]))     # one empty array per neighbor (for each node) -- TODO: Test that out...
    zPreviousBuffer = copy.copy(zBufferEmpty)       # "previous DANSE iteration" buffers
    zBuffer = copy.copy(zBufferEmpty)               # "current DANSE iteration" buffers
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)

    for idxt, tMaster in enumerate(masterClock):  # loop over master clock instants

        # =============================== Pre-DANSE time processing =============================== 
        for k in range(asc.numNodes):   # loop over nodes

            if k == settings.referenceSensor:
                # Special processing for reference sensor
                nSamplesLinedUpForBroadcast[k] += 1
                nLocalSamplesSinceLastIter[k] += 1
                idxCurrSample = idxt                    # number of samples accumulated at master-time `t` at node `k`
            else:
                passedInstants = timeInstants[timeInstants[:, k] <= tMaster, k]     # list of passed time stamps at node `k`
                idxCurrSample = len(passedInstants)     # number of samples accumulated at master-time `t` at node `k`
                if lastSampleIdx[k] < idxCurrSample:
                    nSamplesLinedUpForBroadcast[k] += idxCurrSample - lastSampleIdx[k]     # can be more than 1...
                    nLocalSamplesSinceLastIter[k] += idxCurrSample - lastSampleIdx[k]      # can be more than 1...
                else:
                    pass    # if there is no new sample at node `k`, don't increment the node-specific numbers of samples
            lastSampleIdx[k] = idxCurrSample    # record current sample at node `k`

            if nSamplesLinedUpForBroadcast[k] == settings.broadcastLength:  
                # ~~~~~~~~~~~~~~ Time to broadcast! ~~~~~~~~~~~~~~
                # Extract current data chunk
                yinCurr = yin[np.amax([0, lastSampleIdx[k] - frameSize]):lastSampleIdx[k], asc.sensorToNodeTags == k]
                zLocal = danse_compression(yinCurr, w[k][:, i, :], win)        # local compressed signals
                z[k][:, lk[k]] = zLocal                                     # store them
                for q in neighbourNodes[k]:
                    # Fill in neighbors' buffers with the L = `settings.broadcastLength` last samples of local compressed signals
                    zBuffer[k][q] = np.concatenate((zBuffer[k][q], zLocal[-settings.broadcastLength:]), axis=0)     # TODO: reset (empty-out) buffer at some point
                settings.broadcastLength[k] = 0     # reset broadcast buffer length
                lk[k] += 1                          # increment broadcast index
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if nLocalSamplesSinceLastIter[k] == frameSize:
                print(f'DANSE iteration {i+1}...')
                # ~~~~~~~~~~~~~~ Time to update the filter coefficients! ~~~~~~~~~~~~~~
                # Gather local observation vector
                yLocalCurr = yin[(idxCurrSample - frameSize):idxCurrSample, asc.sensorToNodeTags == k]  # local sensor observations ("$\mathbf{y}_k$" in [1])
                # Process buffers
                zFromBuffers = np.empty((0, frameSize), dtype=float)
                for q in neighbourNodes[k]:
                    Bq = len(zBuffer[k][q])
                    
                    if Bq == frameSize:     # case 1: no broadcast frame mismatch between node `k` and node `q`
                        zFromBuffers = np.concatenate((zFromBuffers, zBuffer[k][q]), axis=0)    # TODO: check axis

                    if Bq == frameSize - settings.broadcastLength:     # case 2: positive broadcast frame mismatch between node `k` and node `q`
                        if i == 0:
                            raise ValueError(f'Buffer underflow occured at first DANSE iteration (node k={k+1}`s B_{q+1} buffer).')
                        bufferFlags[k][q] = -1      # raise negative flag
                        # Use previous iteration's buffer
                        zUnderFlow = np.concatenate((zPreviousBuffer[k][q][-settings.broadcastLength:], zBuffer[k][q]), axis=1) # TODO: check axis
                        zFromBuffers = np.concatenate((zFromBuffers, zUnderFlow), axis=0)   # TODO: check axis

                    if Bq == frameSize + settings.broadcastLength:     # case 3: negative broadcast frame mismatch between node `k` and node `q`
                        bufferFlags[k][q] = 1       # raise positive flag
                        # Discard L = `settings.broadcastLength` oldest samples in buffer
                        zFromBuffers = np.concatenate((zFromBuffers, zBuffer[k][q][settings.broadcastLength:]), axis=0)    # TODO: check axis

                    else:
                        raise ValueError(f'Node k={k+1}: Unexpected buffer size for neighbor node q={q+1}.')    # TODO: ensure that we never get that error
                        
                # Build full available observation vector
                yTildeCurr = np.concatenate((yLocalCurr, zFromBuffers))
                ytilde[k][:, i, :] = yTildeCurr
                # Go to frequency domain
                ytilde_hat_curr = fftscale * np.fft.fft(ytilde[k][:, i, :] * win, frameSize, axis=0)
                # Keep only positive frequencies
                ytilde_hat[k][:, i, :] = ytilde_hat_curr[:numFreqLines, :]
                # Compute VAD
                VADinFrame = oVAD[(idxCurrSample - frameSize):idxCurrSample]
                oVADframes[i] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1
                # Count autocorrelation matrices updates
                if oVADframes[i]:
                    numUpdatesRyy[k] += 1
                else:     
                    numUpdatesRnn[k] += 1
                # Loop over frequency lines
                for kappa in range(numFreqLines):
                    # Autocorrelation matrices update -- eq.(46) in ref.[1] /and-or/ eq.(20) in ref.[3].
                    if oVADframes[i]:
                        Ryy[k][kappa, :, :] = settings.expAvgBeta * Ryy[k][kappa, :, :] + \
                            (1 - settings.expAvgBeta) * np.outer(ytilde_hat[k][kappa, i, :], ytilde_hat[k][kappa, i, :].conj())  # update signal + noise matrix
                    else:
                        Rnn[k][kappa, :, :] = settings.expAvgBeta * Rnn[k][kappa, :, :] + \
                            (1 - settings.expAvgBeta) * np.outer(ytilde_hat[k][kappa, i, :], ytilde_hat[k][kappa, i, :].conj())   # update noise-only matrix

                    #Check quality of autocorrelations estimates
                    if not goodAutocorrMatrices[kappa]:
                        goodAutocorrMatrices[kappa] = check_autocorr_est(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                                numUpdatesRyy[k], numUpdatesRnn[k], settings.minNumAutocorrUpdates)

                    if goodAutocorrMatrices[kappa]:
                        if settings.performGEVD:
                            w[k][kappa, i, :], Qmat = perform_gevd(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                                                                    settings.GEVDrank, settings.referenceSensor)
                        else:
                            # Reference sensor selection vector
                            Evect = np.zeros((dimYTilde[k],))
                            Evect[settings.referenceSensor] = 1
                            # Cross-correlation matrix update 
                            ryd[k][kappa, :] = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect
                            # Update node-specific parameters of node k
                            w[k][kappa, i, :] = np.linalg.inv(Ryy[k][kappa, :, :]) @ ryd[k][kappa, :]
                    elif i > 1:
                        # Do not update the filter coefficients
                        w[k][kappa, i, :] = w[k][kappa, i - 1, :]
                
                # Increment DANSE iteration index
                i += 1
                # Update buffers status for node `k`
                zPreviousBuffer[k] = copy.copy(zBuffer[k])    # TODO make sure this is the correct way to copy objects in this case
                zBuffer[k] = copy.copy(zBufferEmpty[k])       # TODO make sure this is the correct way to copy objects in this case
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # =========================================================================================

    print('Simultaneous DANSE processing all done.')

    stop = 1

    return 0

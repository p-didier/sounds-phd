import numpy as np
import time
from . import classes

"""
References:
 - [1]: Bertrand, Alexander, and Marc Moonen. "Distributed adaptive node-specific 
        signal estimation in fully connected sensor networks—Part I: 
        Sequential node updating." IEEE Transactions on Signal Processing 
        58.10 (2010): 5277-5291.
 - [2]: Ruiz, Santiago, Toon Van Waterschoot, and Marc Moonen. "Distributed 
        combined acoustic echo cancellation and noise reduction in wireless 
        acoustic sensor and actuator networks." IEEE/ACM Transactions on Audio, 
        Speech, and Language Processing (2022).
"""

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

def danse_sequential(y_STFT, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD):
    """Wrapper for Sequential-Node-Updating DANSE.
    Parameters
    ----------
    y_STFT : [Nf x Nt x Ns] np.ndarray
        The microphone signals in the STFT domain ([freq bins x time frames x sensor]).
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray
        Voice Activity Detector output per time frame.

    Returns
    -------
    d : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes.
    wkk : [Nn x 1] list of [Nf x N(k)] nd.arrays (complex)
        Local filter coefficients.
    gkmk : [Nn x 1] list of [Nf x N(-k)] nd.arrays (complex)
        Filter coefficients applied to data incoming from neighbor nodes.  
    ytilde : [Nn x 1] list of [Nt x 1] lists of [Nf x N(k)+N(-k)] nd.arrays (complex)
        Full observations (local signals & compressed remote signals), for all TF bins.  
    """
    # # Interpret settings
    # olaOverlap = settings.OLAoverlap
    # if not settings.useOLA:
    #     olaOverlap = 0

    # Define random generator
    rng = np.random.default_rng(settings.randSeed)

    # Extract useful variables
    numFreqLines, numTimeFrames = y_STFT.shape[0], y_STFT.shape[1]
    danseBlockSize = settings.timeBtwConsecUpdates * asc.samplingFreq   # DANSE iteration block size [samples]
    danseB = int(danseBlockSize / settings.stftEffectiveFrameLen)               # num. of frames within 1 DANSE iteration block
    # # WOLA window
    # if settings.WOLAwindow == 'sqrthann':
    #     wolaWin = np.sqrt(np.hanning(danseB))

    # Divide sensor data per nodes
    y = []
    for k in range(1, asc.numNodes + 1):
        y.append(y_STFT[:, :, asc.sensorToNodeTags == k])
    # Identify neighbours of each node
    neighbourNodes = []
    allNodeIdx = np.arange(asc.numNodes)
    for k in range(asc.numNodes):
        neighbourNodes.append(np.delete(allNodeIdx, k))   # Option 1) - FULLY-CONNECTED WASN

    # Initialize loop
    wkk      = []   # filter coefficients applied to local signals
    gkmk     = []   # filter coefficients applied to incoming signals
    wk_tilde = []   # all filter coefficients
    for k in range(asc.numNodes):
        wkkSingleFrame = settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, asc.numSensorPerNode[k])) +\
                1j * rng.random(size=(numFreqLines, asc.numSensorPerNode[k]))
            ) # random values
        wkkAllFrames = np.tile(wkkSingleFrame, (numTimeFrames, 1, 1))
        wkkAllFrames = np.transpose(wkkAllFrames, [1,0,2])
        wkk.append(wkkAllFrames)  
        #
        gkmkSingleFrame = settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, len(neighbourNodes[k]))) +\
                1j * rng.random(size=(numFreqLines, len(neighbourNodes[k])))
            ) # random values
        gkmkAllFrames = np.tile(gkmkSingleFrame, (numTimeFrames, 1, 1))
        gkmkAllFrames = np.transpose(gkmkAllFrames, [1,0,2])
        gkmk.append(gkmkAllFrames)
        # wkk.append(np.ones((numFreqLines, asc.numSensorPerNode[k]), dtype=complex))  # ones only
        # gkmk.append(np.ones((numFreqLines, len(neighbourNodes[k])), dtype=complex))  # ones only
        wk_tilde.append(np.concatenate((wkk[-1], gkmk[-1]), axis=-1))

    z = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)   # compressed local signals
    d = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)   # init desired signal estimate
    # Init autocorrelation matrices ([TODO] FOR NOW: storing them all, for each node (unnecessary))
    Ryy = []
    Rnn = []
    ryd = []
    ytilde = []
    dimYTilde = np.zeros(asc.numNodes, dtype=int)
    for k in range(asc.numNodes):
        dimYTilde[k] = y[k].shape[-1] + len(neighbourNodes[k])              # dimension of \tilde{y}_k
        slice = np.zeros((dimYTilde[k], dimYTilde[k]), dtype=complex)       # single autocorrelation matrix init
        Ryy.append(np.tile(slice, (numFreqLines, 1, 1)))                    # speech + noise
        Rnn.append(np.tile(slice, (numFreqLines, 1, 1)))                    # noise only
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        ytilde.append(np.zeros((numFreqLines, numTimeFrames, dimYTilde[k]), dtype=complex))
    
    # Filter coefficients update flag for each frequency
    goodAutocorrMatrices = np.array([False for _ in range(numFreqLines)])
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)

    # # Overlap-add idea
    # nFramesPerOLA = 4
    # nOverlappingFrames = nFramesPerOLA / 2

    tStartGlobal = time.perf_counter()    # time computation
    # Loop over time frames
    for l in range(numTimeFrames):
        print(f'DANSE -- Time frame {l + 1}/{numTimeFrames}...')
        # Loop over nodes in network 
        for k in range(asc.numNodes):

            # Compress node k's signals 
            z[:, l, k] = np.einsum('ij,ij->i', wkk[k][:, l, :].conj(), y[k][:, l, :])  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850
            
            # Retrieve compressed observations from other nodes (q =/= k)
            ytilde_curr = y[k][:, l, :]     # Full "observations vector" at node k and time l
            for q in neighbourNodes[k]:
                z[:, l, q] = np.einsum('ij,ij->i', wkk[q][:, l, :].conj(), y[q][:, l, :])  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850
                # Build available vectors at node k
                zq = z[:, l, q]
                zq = zq[:, np.newaxis]  # necessary for np.concatenate
                ytilde_curr = np.concatenate((ytilde_curr, zq), axis=1)
            ytilde[k][:, l, :] = ytilde_curr

            # Count autocorrelation matrices updates
            if oVAD[l]:
                numUpdatesRyy[k] += 1
            else:     
                numUpdatesRnn[k] += 1

            # Loop over frequency bins
            for kappa in range(numFreqLines):
            
                # Autocorrelation matrices update -- eq.(46) in ref.[1] /and-or/ eq.(20) in ref.[2].
                if oVAD[l]:
                    Ryy[k][kappa, :, :] = settings.expAvgBeta * Ryy[k][kappa, :, :] + \
                        (1 - settings.expAvgBeta) * np.outer(ytilde[k][kappa, l, :], ytilde[k][kappa, l, :].conj())  # update signal + noise matrix
                else:
                    Rnn[k][kappa, :, :] = settings.expAvgBeta * Rnn[k][kappa, :, :] + \
                        (1 - settings.expAvgBeta) * np.outer(ytilde[k][kappa, l, :], ytilde[k][kappa, l, :].conj())   # update noise-only matrix

                #Check quality of autocorrelations estimates
                if not goodAutocorrMatrices[kappa]:
                    goodAutocorrMatrices[kappa] = check_autocorr_est(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                            numUpdatesRyy[k], numUpdatesRnn[k], settings.minNumAutocorrUpdates)

                if goodAutocorrMatrices[kappa] and l % danseB == 0:
                    # Cross-correlation matrix update 
                    Evect = np.zeros((dimYTilde[k],))
                    Evect[settings.referenceSensor] = 1    # reference sensor
                    ryd[k][kappa, :] = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect
                    # Update node-specific parameters of node k
                    wk_tilde[k][kappa, l, :] = np.linalg.pinv(Ryy[k][kappa, :, :]) @ ryd[k][kappa, :]
                elif l > 1:
                    # Do not update the filter coefficients
                    wk_tilde[k][kappa, l, :] = wk_tilde[k][kappa, l - 1, :]

                # Distinguish filter coefficients applied...
                wkk[k][kappa, l, :]  = wk_tilde[k][kappa, l, :y[k].shape[-1]]  #...to the local signals
                gkmk[k][kappa, l, :] = wk_tilde[k][kappa, l, y[k].shape[-1]:]  #...to the received signals

            # if l > nFramesPerOLA:
            # Compute desired signal estimate
            d[:, l, k] = np.einsum('ij,ij->i', wk_tilde[k][:, l, :].conj(), ytilde[k][:, l, :])  # vectorized way to do things https://stackoverflow.com/a/15622926/16870850

            stop = 1

    print(f'DANSE computations completed in {np.round(time.perf_counter() - tStartGlobal, 2)} s.')

    return d, wkk, gkmk, z



def danse_sequential_old(y_STFT, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD):
    """Wrapper for Sequential-Node-Updating DANSE.
    Parameters
    ----------
    y_STFT : [Nf x Nt x Ns] np.ndarray
        The microphone signals in the STFT domain ([freq bins x time frames x sensor]).
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray
        Voice Activity Detector output per time frame.
    Returns
    -------
    d : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes. 
    wkk : [Nn x 1] list of [Nf x N(k)] nd.arrays (complex)
        Local filter coefficients. 
    gkmk : [Nn x 1] list of [Nf x N(-k)] nd.arrays (complex)
        Filter coefficients applied to data incoming from neighbor nodes. 
    """

    # Define random generator
    rng = np.random.default_rng(settings.randSeed)

    # Extract useful variables
    numFreqLines, numTimeFrames = y_STFT.shape[0], y_STFT.shape[1]
    danseBlockSize = settings.timeBtwConsecUpdates * asc.samplingFreq   # DANSE iteration block size [samples]
    danseB = int(danseBlockSize / settings.stftWinLength)               # num. of frames within 1 DANSE iteration block
    # # WOLA window
    # if settings.WOLAwindow == 'sqrthann':
    #     wolaWin = np.sqrt(np.hanning(danseB))

    # Divide sensor data per nodes
    y = []
    for k in range(1, asc.numNodes + 1):
        y.append(y_STFT[:, :, asc.sensorToNodeTags == k])
    # Identify neighbours of each node
    neighbourNodes = []
    allNodeIdx = np.arange(asc.numNodes)
    for k in range(asc.numNodes):
        neighbourNodes.append(np.delete(allNodeIdx, k))   # Option 1) - FULLY-CONNECTED WASN

    # Initialize loop
    wkk  = []   # filter coefficients applied to local signals
    gkmk = []   # filter coefficients applied to incoming signals
    for k in range(asc.numNodes):
        wkk.append(settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, asc.numSensorPerNode[k])) +\
            1j * rng.random(size=(numFreqLines, asc.numSensorPerNode[k]))
            ))  # random values
        gkmk.append(settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, len(neighbourNodes[k]))) +\
            1j * rng.random(size=(numFreqLines, len(neighbourNodes[k])))
            ))  # random values
    z = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)               # compressed local signals
    d = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)   # init desired signal estimate
    # Init autocorrelation matrices ([TODO] FOR NOW: storing them all, for each node (unnecessary))
    Ryy = []
    Rnn = []
    dimYTilde = np.zeros(asc.numNodes, dtype=int)
    for k in range(asc.numNodes):
        dimYTilde[k] = y[k].shape[-1] + len(neighbourNodes[k])          # dimension of \tilde{y}_k
        slice = np.zeros((dimYTilde[k], dimYTilde[k]), dtype=complex)   # single autocorrelation matrix init
        Ryy.append(np.tile(slice, (numFreqLines, 1, 1)))                # speech + noise
        Rnn.append(np.tile(slice, (numFreqLines, 1, 1)))                # noise only

    # Filter coefficients update flag for each frequency
    updateWeights = np.array([False for _ in range(numFreqLines)])
    # Autocorrelation matrices update counters
    numUpdatesRyy = 0
    numUpdatesRnn = 0

    i = 0   # DANSE iteration index
    stopCriterion = False
    tStartGlobal = time.perf_counter()    # time computation
    while not stopCriterion:
        tStart = time.perf_counter()    # time computation
        # Loop over nodes in network 
        for k in range(asc.numNodes):
            # Identify indices of new sensor observations -- corresponding to (approximatively) <settings.timeBtwConsecUpdates> s of signal
            framesCurrIter = np.arange(i * danseB, i * danseB + danseB, dtype=int)
            
            # Init available vectors at node k
            y_tilde = np.zeros((numFreqLines, len(framesCurrIter), dimYTilde[k]), dtype=complex) 
            # Loop over frequency bins
            for kappa in range(numFreqLines):
                # Loop over time frames in current DANSE iteration block
                for idx in range(len(framesCurrIter)):

                    # Compress local signals 
                    l = framesCurrIter[idx]   # actual frame index
                    z[kappa, l, k] = np.dot(wkk[k][kappa, :].conj(), y[k][kappa, l, :])

                    # ... Broadcast compressed observations to other nodes ...

                    # Retrieve compressed observations from other nodes
                    y_tilde_curr = y[k][kappa, l, :]
                    for q in neighbourNodes[k]:
                        z[kappa, l, q] = np.dot(wkk[q][kappa, :].conj(), y[q][kappa, l, :])
                        # Build available vectors at node k
                        y_tilde_curr = np.append(y_tilde_curr, z[kappa, l, q])
                    y_tilde[kappa, idx, :] = y_tilde_curr
                
                    # Autocorrelation matrices update -- eq.(46) in ref.[1] /and-or/ eq.(20) in ref.[2].
                    if oVAD[l]:
                        Ryy[k][kappa, :, :] = settings.expAvgBeta * Ryy[k][kappa, :, :] + \
                            (1 - settings.expAvgBeta) * np.outer(y_tilde[kappa, idx, :], y_tilde[kappa, idx, :].conj())  # update signal + noise matrix
                        numUpdatesRyy += 1
                    else:
                        Rnn[k][kappa, :, :] = settings.expAvgBeta * Rnn[k][kappa, :, :] + \
                            (1 - settings.expAvgBeta) * np.outer(y_tilde[kappa, idx, :], y_tilde[kappa, idx, :].conj())   # update noise-only matrix
                        numUpdatesRnn += 1

                #Check quality of autocorrelations estimates
                if not updateWeights[kappa]:
                    updateWeights[kappa] = check_autocorr_est(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :],
                                                            numUpdatesRyy, numUpdatesRnn, settings.minNumAutocorrUpdates)

                if updateWeights[kappa]:
                    # Cross-correlation matrix update 
                    Evect = np.zeros((dimYTilde[k],))
                    Evect[settings.referenceSensor] = 1    # reference sensor
                    ryd = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect

                    # Update node-specific parameters of node k
                    wk_tilde_kappa = np.linalg.pinv(Ryy[k][kappa, :, :]) @ ryd
                    wkk[k][kappa, :] = wk_tilde_kappa[:y[k].shape[-1]]
                    gkmk[k][kappa, :] = wk_tilde_kappa[y[k].shape[-1]:]

                    # Compute desired signal estimate
                    d[kappa, framesCurrIter, k] += wk_tilde_kappa.conj().T @ y_tilde[kappa, :, :].T

                    # # Compute desired signal estimate
                    # d[kappa, framesCurrIter, k] = \
                    #         wkk[k][kappa, :].conj().T @ y[k][kappa, framesCurrIter, :].T\
                    #         + gkmk[k][kappa, :].conj().T @ z[kappa, framesCurrIter, :][:, [n-1 for n in neighbourNodes[k]]].T
                    #         # + gkmk[k][kappa, :].conj().T @ y_tilde[kappa, :, y[k].shape[-1]:].T
                else:
                    d[kappa, framesCurrIter, k] += y[k][kappa, framesCurrIter, settings.referenceSensor]

        print(f'DANSE iteration {i + 1} done in {np.round(time.perf_counter() - tStart, 2)} s...')
        i += 1
        stopCriterion = i * danseB + danseB >= numTimeFrames

    print(f'DANSE computations completed after {i} iterations and {np.round(time.perf_counter() - tStartGlobal, 2)} s.')

    return d, wkk, gkmk
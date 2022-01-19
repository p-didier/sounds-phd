import numpy as np
import time
from . import classes           # <-- classes for DANSE


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
    """

    # Define random generator
    rng = np.random.default_rng(settings.randSeed)

    # Extract useful variables
    numFreqLines, numTimeFrames = y_STFT.shape[0], y_STFT.shape[1]

    # Divide sensor data per nodes
    y = []
    for k in range(1, asc.numNodes + 1):
        y.append(y_STFT[:, :, asc.sensorToNodeTags == k])

    # DANSE iteration length
    danseBlockSize = settings.timeBtwConsecUpdates * asc.samplingFreq   # [samples]
    danseB = int(danseBlockSize / settings.stftWinLength)               # num. of frames within 1 DANSE iteration block

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
        for k in range(asc.numNodes):

            # Identify indices of new sensor observations -- corresponding to (approximatively) <settings.timeBtwConsecUpdates> s of signal
            framesCurrIter = np.arange(i * danseB, (i + 1) * danseB)

            # Loop over frequency bins
            y_tilde = np.zeros((numFreqLines, len(framesCurrIter), dimYTilde[k]), dtype=complex) # Init available vectors at node k
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

                #Check quality of covariance estimates
                if not updateWeights[kappa]:
                    # Full rank matrices
                    if np.linalg.matrix_rank(np.squeeze(Ryy[k][kappa, :, :])) == dimYTilde[k] and\
                        np.linalg.matrix_rank(np.squeeze(Rnn[k][kappa, :, :])) == dimYTilde[k]:
                        # Sufficient number of updates to get reasonable result
                        if numUpdatesRyy > settings.minNumAutocorr and\
                             numUpdatesRnn > settings.minNumAutocorr:
                            updateWeights[kappa] = True

                if updateWeights[kappa]:
                    # Cross-correlation matrix update 
                    Evect = np.zeros((dimYTilde[k],))
                    Evect[0] = 1    # reference sensor
                    ryd = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ Evect

                    # Update node-specific parameters of node k
                    wk_tilde_kappa = np.linalg.pinv(Ryy[k][kappa, :, :]) @ ryd
                    wkk[k][kappa, :] = wk_tilde_kappa[:y[k].shape[-1]]
                    gkmk[k][kappa, :] = wk_tilde_kappa[y[k].shape[-1]:]

                # Compute desired signal estimate
                d[kappa, framesCurrIter, k] = \
                        wkk[k][kappa, :].conj().T @ y[k][kappa, framesCurrIter, :].T\
                        + gkmk[k][kappa, :].conj().T @ z[kappa, framesCurrIter, :][:, [n-1 for n in neighbourNodes[k]]].T

        print(f'DANSE iteration {i + 1} done in {np.round(time.perf_counter() - tStart, 2)} s...')
        i += 1
        stopCriterion = (i + 1) * danseB - 1 >= numTimeFrames

    print(f'DANSE computations completed after {i} iterations and {np.round(time.perf_counter() - tStartGlobal, 2)} s.')

    return d
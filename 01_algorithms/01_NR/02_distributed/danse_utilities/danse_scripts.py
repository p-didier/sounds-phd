import numpy as np
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
    for idxNode in range(1, asc.numNodes + 1):
        y.append(y_STFT[:, :, asc.sensorToNodeTags == idxNode])

    # DANSE iteration length
    danseBlockSize = settings.timeBtwConsecUpdates * asc.samplingFreq   # [samples]
    danseB = int(danseBlockSize / settings.stftWinLength)               # num. of frames within 1 DANSE iteration block

    # Identify neighbours of each node
    neighbourNodes = []
    allNodeIdx = np.arange(start=1, stop=asc.numNodes + 1)
    for idxNode in range(asc.numNodes):
        neighbourNodes.append(np.delete(allNodeIdx, idxNode))   # Option 1) - FULLY-CONNECTED WASN

    # Initialize loop
    wkk  = []   # filter coefficients applied to local signals
    gkmk = []   # filter coefficients applied to incoming signals
    for idxNode in range(asc.numNodes):
        wkk.append(settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, asc.numSensorPerNode[idxNode])) +\
            1j * rng.random(size=(numFreqLines, asc.numSensorPerNode[idxNode]))
            ))  # random values
        gkmk.append(settings.initialWeightsAmplitude * (
                rng.random(size=(numFreqLines, len(neighbourNodes[idxNode]))) +\
            1j * rng.random(size=(numFreqLines, len(neighbourNodes[idxNode])))
            ))  # random values
    z = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)               # compressed local signals
    d = np.zeros((numFreqLines, numTimeFrames, asc.numNodes), dtype=complex)   # init desired signal estimate

    i = 0   # DANSE iteration index
    stopCriterion = False
    while not stopCriterion:
        print(f'DANSE iteration {i + 1}...')
        for k in range(asc.numNodes):

            # Identify indices of new sensor observations -- corresponding to (approximatively) <settings.timeBtwConsecUpdates> s of signal
            frames_i = np.arange(i * danseB, (i + 1) * danseB)

            dimYTilde = y[k].shape[-1] + len(neighbourNodes[k])  # dimension of \tilde{y}_k vector

            # Loop over frequency bins
            y_tilde = np.zeros((numFreqLines, len(frames_i), dimYTilde), dtype=complex) # Init available vectors at node k
            for kappa in range(numFreqLines):

                # Init autocorrelation matrix (for current frequency bin)
                Ryy = np.eye(dimYTilde, dtype=complex)            # observed signals autocorrelation
                Rnn = np.eye(dimYTilde, dtype=complex)            # noise autocorrelation

                # Loop over time frames in current DANSE iteration block
                for idx in range(len(frames_i)):

                    # Compress local signals 
                    l = frames_i[idx]   # actual frame index
                    z[kappa, l, k] = np.dot(wkk[k][kappa, :].conj().T, y[k][kappa, l, :])

                    # ... Broadcast compressed observations to other nodes ...

                    # Retrieve compressed observations from other nodes
                    y_tilde_curr = y[k][kappa, l, :]
                    for q in neighbourNodes[k]:
                        l = frames_i[idx]   # actual frame index
                        z[kappa, l, q - 1] = np.dot(wkk[q - 1][kappa, :].conj().T, y[q - 1][kappa, l, :])
                        # Build available vectors at node k
                        y_tilde_curr = np.append(y_tilde_curr, z[kappa, l, q - 1])
                    y_tilde[kappa, idx, :] = y_tilde_curr
                
                    # Autocorrelation matrices update -- eq.(46) in ref.[1] /and-or/ eq.(20) in ref.[2].
                    if oVAD[l]:
                        Ryy = settings.expAvgBeta * Ryy + \
                            (1 - settings.expAvgBeta) * np.outer(y_tilde[kappa, idx, :], y_tilde[kappa, idx, :].conj())  # update signal + noise matrix
                    else:
                        Rnn = settings.expAvgBeta * Rnn + \
                            (1 - settings.expAvgBeta) * np.outer(y_tilde[kappa, idx, :], y_tilde[kappa, idx, :].conj())   # update noise-only matrix

                Ryy /= len(frames_i)    # normalize average
                Rnn /= len(frames_i)

                # Cross-correlation matrix update 
                Evect = np.zeros((dimYTilde,))
                Evect[0] = 1
                ryd = (Ryy - Rnn) @ Evect

                # Update node-specific parameters of node k
                wk_tilde_kappa = np.linalg.pinv(Ryy) @ ryd
                wkk[k][kappa, :] = wk_tilde_kappa[:y[k].shape[-1]]
                gkmk[k][kappa, :] = wk_tilde_kappa[y[k].shape[-1]:]

                # Compute desired signal estimate
                d[kappa, frames_i, k] = \
                        wkk[k][kappa, :].conj().T @ y[k][kappa, frames_i, :].T\
                        + gkmk[k][kappa, :].conj().T @ z[kappa, frames_i, :][:, [n-1 for n in neighbourNodes[k]]].T

        i += 1
        stopCriterion = (i + 1) * danseB - 1 >= numTimeFrames

    print(f'DANSE computations completed after {i} iterations.')

    return d
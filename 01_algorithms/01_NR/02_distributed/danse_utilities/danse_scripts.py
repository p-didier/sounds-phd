
import numpy as np
import time, datetime
from . import classes
from . import danse_subfcns as subs
from . import danse_plots as dplt

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
    rng, win, frameSize, nNewSamplesPerFrame, numIterations, _, neighbourNodes = subs.danse_init(yin, settings, asc)
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
                    w[k][:, i + 1, :], Qmat = subs.perform_gevd_noforloop(Ryy[k], Rnn[k], settings.GEVDrank, settings.referenceSensor)
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


def danse_simultaneous(yin, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD, timeInstants, masterClockNodeIdx):
    """Wrapper for Simultaneous-Node-Updating DANSE (rs-DANSE) [2].

    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray of floats
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray of booleans or binary ints
        Voice Activity Detector.
    timeInstants : [Nt x Nn] np.ndarray of floats
        Time instants corresponding to the samples of each of the Nn nodes in the network. 
    masterClockNodeIdx : int
        Index of node to be used as "master clock" (0 ppm SRO). 

    Returns
    -------
    d : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using full-observations vectors (also data coming from neighbors).
    dLocal : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using only local observations (not data coming from neighbors).
        -Note: if `settings.computeLocalEstimate == False`, then `dLocal` is output as an all-zeros array.
    """
    
    # Initialization (extracting useful quantities)
    rng, win, frameSize, nExpectedNewSamplesPerFrame, numIterations, _, neighbourNodes = subs.danse_init(yin, settings, asc)
    fftscale = 1 / win.sum()
    ifftscale = win.sum()

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, masterClockNodeIdx]     # reference clock

    # ---------------------- Arrays initialization ----------------------
    lk = np.zeros(asc.numNodes, dtype=int)                      # node-specific broadcast index
    i = np.zeros(asc.numNodes, dtype=int)                       # !node-specific! DANSE iteration index
    nReadyForBroadcast = np.zeros(asc.numNodes, dtype=int)      # node-specific number of samples lined up for broadcast 
    nNewLocalSamples = np.zeros(asc.numNodes, dtype=int)        # node-specific number of new local samples since the last DANSE iteration 
    lastSampleIdx = np.full(shape=(asc.numNodes,), fill_value=-1)           # last sample index lined up for broadcast
    #
    wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
    Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
    Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
    ryd = []                                        # cross-correlation between observations and estimations
    ytilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                  # local full observation vectors, frequency-domain
    z = []                                          # current-iteration compressed signals used in DANSE update
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    bufferLengths = []                              # node-specific number of samples in each buffer
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
    numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
    if settings.computeLocalEstimate:
        wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
        Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
        Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
        dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)

    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1)+ len(neighbourNodes[k])
        # wTilde.append(settings.initialWeightsAmplitude * (rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k])) +\
        #     1j * rng.random(size=(numFreqLines, numIterations + 1, dimYTilde[k]))))
        # wTilde.append(settings.initialWeightsAmplitude * np.ones((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex))   # ones
        wtmp = np.zeros((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex)
        wtmp[:, :, 0] = 1
        wTilde.append(wtmp)   # zeros
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytildeHat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        #
        sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
        Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))    # init all buffer flags at 0 (assuming no over- or under-flow)
        bufferLengths.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))
        #
        z.append(np.empty((frameSize, 0), dtype=float))
        zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
        #
        if settings.computeLocalEstimate:
            dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
            sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            wLocal.append(settings.initialWeightsAmplitude * (rng.random(size=(numFreqLines, numIterations + 1, dimYLocal[k])) +\
                1j * rng.random(size=(numFreqLines, numIterations + 1, dimYLocal[k]))))
            Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
            Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
    # Desired signal estimate [frames x frequencies x nodes]
    dhat = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
    d = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhat`
    dhatLocal = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)
    dLocal = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatLocal`

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

    t0 = time.perf_counter()    # global timing
    for idxt, tMaster in enumerate(masterClock):    # loop over master clock instants

        for k in range(asc.numNodes):
            # Update sample counts at new instant `tMaster`
            lastSampleIdx[k], nReadyForBroadcast[k], nNewLocalSamples[k] = subs.count_samples(
                timeInstants[:, k],
                tMaster,
                lastSampleIdx[k],
                nReadyForBroadcast[k],
                nNewLocalSamples[k])
            
            # Check whether flag for compressing + broadcasting should be raised
            broadcastSignals, startedCompression[k] = subs.broadcast_flag_raising(
                startedCompression[k],
                nReadyForBroadcast[k],
                frameSize,
                settings.broadcastLength)

            if broadcastSignals: 
                # # Inform user
                # if k == 0 and lk[k] % 1000 == 0:
                #     print(f'tMaster = {np.round(tMaster, 4)}s: Node 1`s {lk[k]}^th broadcast to its neighbors')

                # Extract current data chunk
                yinCurr = yin[(lastSampleIdx[k] - frameSize + 1):(lastSampleIdx[k] + 1), asc.sensorToNodeTags == k+1]
                # Compress current data chunk in the frequency domain
                zLocal = subs.danse_compression(yinCurr, wTilde[k][:, i[k], :yinCurr.shape[-1]], settings.danseWindow[:, np.newaxis])        # local compressed signals
                # Loop over node `k`'s neighbours and fill their buffers
                zBuffer = subs.fill_buffers(k, neighbourNodes, lk, zBuffer, zLocal, settings.broadcastLength)
                nReadyForBroadcast[k] = 0       # reset number of samples ready for broadcast
                idxBroadcasts[k].append(idxt)   # save master clock index where broadcast happened
                lk[k] += 1                      # increment broadcast index

        # Reset the `k`-loop here to ensure that all nodes have broadcasted what they had to broadcast before counting buffered samples
        for k in range(asc.numNodes):
            # Record buffer lengths
            bufferLengths[k][idxt, :] = np.array([len(buffer) for buffer in zBuffer[k]])

        # Reset the `k`-loop here to ensure that all nodes have broadcasted what they had to broadcast before further processing
        for k in range(asc.numNodes):  # loop over nodes

            # Checks: can we update?
            emptyBuffer = any([len(buff) == 0 for buff in zBuffer[k]])   # check whether local buffers are full
            enoughSamples = nNewLocalSamples[k] >= nExpectedNewSamplesPerFrame

            if enoughSamples and not emptyBuffer:

                if nNewLocalSamples[k] > nExpectedNewSamplesPerFrame:
                    stop = 1    # TODO deal with that case
                
                if lastSampleIdx[k] >= frameSize - 1:
                    # ~~~~~~~~~~~~~~ Time to update the filter coefficients! ~~~~~~~~~~~~~~
                    t0update = time.perf_counter()
                    idxUpdates[k].append(idxt)
                    
                    # Gather local observation vector
                    idxStartChunk = np.amax([0, lastSampleIdx[k] - frameSize + 1])
                    idxEndChunk = lastSampleIdx[k] + 1
                    yLocalCurr = yin[idxStartChunk:idxEndChunk, asc.sensorToNodeTags == k+1]  # local sensor observations ("$\mathbf{y}_k$" in [1])

                    # Process buffers
                    z[k], bufferFlags[k][idxt, :] = subs.process_incoming_signals_buffers(
                        zBuffer[k],
                        z[k],
                        neighbourNodes[k],
                        i[k],
                        frameSize,
                        N=nExpectedNewSamplesPerFrame,
                        L=settings.broadcastLength,
                        lastExpectedIter=numIterations - 1)

                    # Wipe local buffers
                    zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]
                    
                    # Build full available observation vector
                    yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                    ytilde[k][:, i[k], :] = yTildeCurr
                    # Go to frequency domain
                    ytildeHatCurr = fftscale * np.fft.fft(ytilde[k][:, i[k], :] * win, frameSize, axis=0)
                    # Keep only positive frequencies
                    ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]
                    if settings.computeLocalEstimate:
                        # Local observations only
                        yHat = ytildeHatCurr[:numFreqLines, :dimYLocal[k]]
                    
                    # Compute VAD
                    VADinFrame = oVAD[idxStartChunk:idxEndChunk]
                    oVADframes[i[k]] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

                    # Count autocorrelation matrices updates
                    yyHtilde = np.einsum('ij,ik->ijk', ytildeHat[k][:, i[k], :], ytildeHat[k][:, i[k], :].conj())
                    if oVADframes[i[k]]:
                        Ryytilde[k] = settings.expAvgBeta * Ryytilde[k] + (1 - settings.expAvgBeta) * yyHtilde  # update WIDE signal + noise matrix
                        numUpdatesRyy[k] += 1
                    else:     
                        Rnntilde[k] = settings.expAvgBeta * Rnntilde[k] + (1 - settings.expAvgBeta) * yyHtilde  # update WIDE noise-only matrix
                        numUpdatesRnn[k] += 1
                        
                    if settings.computeLocalEstimate:
                        yyHlocal = np.einsum('ij,ik->ijk', yHat, yHat.conj())
                        if oVADframes[i[k]]:
                            Ryylocal[k] = settings.expAvgBeta * Ryylocal[k] + (1 - settings.expAvgBeta) * yyHlocal  # update LOCAL signal + noise matrix
                        else:     
                            Rnnlocal[k] = settings.expAvgBeta * Rnnlocal[k] + (1 - settings.expAvgBeta) * yyHlocal  # update LOCAL noise-only matrix

                    # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
                    if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
                        startUpdates[k] = True

                    if startUpdates[k]:
                        # No `for`-loop versions
                        if settings.performGEVD:    # GEVD update
                            wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
                            if settings.computeLocalEstimate:
                                wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)
                        else:                       # regular update (no GEVD)
                            raise ValueError('Not yet implemented')     # TODO
                    else:
                        # Do not update the filter coefficients
                        wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                        if settings.computeLocalEstimate:
                            wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # ----- Compute desired signal chunk estimate (in frequency domain) -----
                    dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                    dhat[:, i[k], k] = dhatCurr
                    if settings.computeLocalEstimate:
                        dhatLocalCurr = np.einsum('ij,ij->i', wLocal[k][:, i[k] + 1, :].conj(), yHat)   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                        dhatLocal[:, i[k], k] = dhatLocalCurr
                    # -----------------------------------------------------------------------

                    # -------------------- Transform back to time domain --------------------
                    dhatCurr[0] = dhatCurr[0].real      # Set DC to real value
                    dhatCurr[-1] = dhatCurr[-1].real    # Set Nyquist to real value
                    dhatCurr = np.concatenate((dhatCurr, np.flip(dhatCurr[:-1].conj())[:-1]))
                    # Back to time-domain
                    dChunk = ifftscale * np.fft.ifft(dhatCurr, len(win))
                    d[idxStartChunk:idxEndChunk, k] += np.real_if_close(dChunk)   # overlap and add construction of output time-domain signal

                    #
                    if settings.computeLocalEstimate:
                        dhatLocalCurr[0] = dhatLocalCurr[0].real      # Set DC to real value
                        dhatLocalCurr[-1] = dhatLocalCurr[-1].real    # Set Nyquist to real value
                        dhatLocalCurr = np.concatenate((dhatLocalCurr, np.flip(dhatLocalCurr[:-1].conj())[:-1]))
                        # Back to time-domain
                        dLocalChunk = ifftscale * np.fft.ifft(dhatLocalCurr, len(win))
                        dLocal[idxStartChunk:idxEndChunk, k] += dLocalChunk   # overlap and add construction of output time-domain signal
                    # -----------------------------------------------------------------------

                    # Inform user
                    print(f'[{np.round(tMaster / masterClock[-1] * 100, 1)}%] Simult. DANSE - t={np.round(tMaster, 4)}s(/{np.round(masterClock[-1], 2)}s): Node {k+1}, i={i[k]+1} ({int(1e3 * (time.perf_counter() - t0update))}ms)')
                    # Increment DANSE iteration index
                    i[k] += 1

                # Reset local samples counter
                nNewLocalSamples[k] = 0

    print('\nSimultaneous DANSE processing all done.')
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=time.perf_counter() - t0))}s.')


    if 0:
        dplt.buffer_sizes_evolution_plot(asc, bufferLengths, neighbourNodes, masterClock, nExpectedNewSamplesPerFrame, bufferFlags)

    stop = 1

    return d, dLocal

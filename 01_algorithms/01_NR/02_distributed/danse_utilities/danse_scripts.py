
from cProfile import label
from doctest import master
import numpy as np
import time, datetime
from . import classes
from . import danse_subfcns as subs
import matplotlib.pyplot as plt
from pyinstrument import Profiler

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
"""


def danse_sequential(yin, asc, settings: classes.ProgramSettings, oVAD):
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
    rng, win, _, frameSize, nNewSamplesPerFrame, numIterations, _, neighbourNodes = subs.danse_init(yin, settings, asc)
    win = win[:, np.newaxis]
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
        if i % 10 == 0:
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
                    # for kappa in range(numFreqLines):
                    #     w[k][kappa, i + 1, :], Qmat = subs.perform_gevd(Ryy[k][kappa, :, :], Rnn[k][kappa, :, :], settings.GEVDrank, settings.referenceSensor)
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

    # Hard-coded parameters
    printingInterval = 2            # [s] minimum time between two "% done" printouts
    alphaExternalFilters = 0.5      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html

    # Profiling
    profiler = Profiler()
    profiler.start()
    
    # Initialization (extracting/defining useful quantities)
    _, winWOLAanalysis, winWOLAsynthesis, frameSize, Ns, numIterations, _, neighbourNodes = subs.danse_init(yin, settings, asc)

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, masterClockNodeIdx]     # reference clock

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Arrays initialization ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    lk = np.zeros(asc.numNodes, dtype=int)          # node-specific broadcast index
    i = np.zeros(asc.numNodes, dtype=int)           # !node-specific! DANSE iteration index
    #
    wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
    wTildeExternal = []                             # external filter coefficients - used for broadcasting only, updated every `settings.timeBtwExternalFiltUpdates` seconds
    wTildeExternalTarget = []                       # target external filter coefficients - used for broadcasting only, updated every `settings.timeBtwExternalFiltUpdates` seconds
    Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
    Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
    ryd = []                                        # cross-correlation between observations and estimations
    ytilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                  # local full observation vectors, frequency-domain
    z = []                                          # current-iteration compressed signals used in DANSE update
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    bufferLengths = []                              # node-specific number of samples in each buffer
    phaseShiftFactors = []                          # phase-shift factors for SRO compensation (only used if `settings.compensateSROs == True`)
    SROsEstimates = []                              # SRO estimates per node (for each neighbor)
    residualSROs = []                               # residual SROs, for each node, across DANSE iterations
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
    numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
    if settings.computeLocalEstimate:
        wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
        Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
        Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
        dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
        
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        wtmp = np.zeros((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex)
        wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTilde.append(wtmp)
        wtmp = np.ones((numFreqLines, sum(asc.sensorToNodeTags == k + 1)), dtype=complex)
        wTildeExternal.append(wtmp)
        wTildeExternalTarget.append(wtmp)
        #
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
        if settings.broadcastDomain == 't':
            z.append(np.empty((frameSize, 0), dtype=float))
        elif settings.broadcastDomain == 'f':
            z.append(np.empty((numFreqLines, 0), dtype=complex))
        zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
        #
        phaseShiftFactors.append(np.zeros(dimYTilde[k]))   # initiate phase shift factors as 0's (no phase shift)
        SROsEstimates.append(np.zeros(len(neighbourNodes[k])))
        residualSROs.append(np.zeros((dimYTilde[k], numIterations)))
        #
        if settings.computeLocalEstimate:
            dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
            wtmp = np.zeros((numFreqLines, numIterations + 1, dimYLocal[k]), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wLocal.append(wtmp)
            sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
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
    # Booleans
    startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Arrays initialization ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # Prepare events to be considered in main `for`-loop
    eventsMatrix, fs = subs.get_events_matrix(timeInstants,
                                            frameSize,
                                            Ns,
                                            settings.broadcastLength,
                                            asc.nodeLinks
                                            )

    # Extra variables -- TEMPORARY: TODO -- to be treated and integrated more neatly
    zFull = []
    a = []
    for k in range(asc.numNodes):
        zFull.append(np.empty((0,)))
        a.append(np.zeros(len(neighbourNodes[k])))
    SROsEstimatesThroughTime = np.zeros(numIterations)

    # External filter updates (for broadcasting)
    lastExternalFiltUpdateInstant = np.zeros(asc.numNodes)   # [s]

    tprint = -printingInterval      # printouts timing

    t0 = time.perf_counter()        # loop timing
    # Loop over time instants when one or more event(s) occur(s)
    for events in eventsMatrix:

        # Parse event matrix (+ inform user)
        t, eventTypes, nodesConcerned = subs.events_parser(events, startUpdates, printouts=settings.printouts.events_parser)

        if t > tprint + printingInterval and settings.printouts.danseProgress:
            print(f'----- t = {np.round(t, 2)}s | {np.round(t / masterClock[-1] * 100)}% done -----')
            tprint = t

        # Loop over events
        for idxEvent in range(len(eventTypes)):

            # Node index
            k = int(nodesConcerned[idxEvent])
            event = eventTypes[idxEvent]

            if event == 'broadcast':        # <-- data compression + broadcast to neighbour nodes
                
                # Extract current local data chunk
                idxEndChunk = int(np.floor(t * fs[k]))
                if settings.broadcastDomain == 't':
                    idxBegChunk = np.amax([idxEndChunk - 2 * frameSize, 0])   # don't go into negative sample indices!
                    yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]
                    # Pad zeros at beginning if needed
                    if idxEndChunk - idxBegChunk < 2 * frameSize:   # Using 2*N to keep power of 2 for FFT
                        yLocalCurr = np.concatenate((np.zeros((2 * frameSize - yLocalCurr.shape[0], yLocalCurr.shape[1])), yLocalCurr))
                elif settings.broadcastDomain == 'f':
                    idxBegChunk = np.amax([idxEndChunk - frameSize, 0])   # don't go into negative sample indices!
                    yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]
                    # Pad zeros at beginning if needed
                    if idxEndChunk - idxBegChunk < frameSize:
                        yLocalCurr = np.concatenate((np.zeros((frameSize - yLocalCurr.shape[0], yLocalCurr.shape[1])), yLocalCurr))
                    
                # Perform broadcast -- update all buffers in network
                zBuffer = subs.broadcast(
                            t,
                            k,
                            fs[k],
                            settings.broadcastLength,
                            yLocalCurr,
                            wTildeExternal[k],
                            frameSize,
                            neighbourNodes,
                            lk,
                            zBuffer,
                            settings.broadcastDomain,
                        )
                    
            elif event == 'update':         # <-- DANSE filter update

                skipUpdate = False  # flag to skip update if needed

                # Extract current local data chunk
                idxEndChunk = int(np.floor(t * fs[k]))
                idxBegChunk = idxEndChunk - frameSize     # <N> samples long chunk
                yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]

                # Compute VAD
                VADinFrame = oVAD[idxBegChunk:idxEndChunk]
                oVADframes[i[k]] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1
                if oVADframes[i[k]]:    # Count number of spatial covariance matrices updates
                    numUpdatesRyy[k] += 1
                else:
                    numUpdatesRnn[k] += 1
                
                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Build local observations vector ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                # Process buffers
                z[k], bufferFlags = subs.process_incoming_signals_buffers(
                                    zBuffer[k],
                                    z[k],
                                    neighbourNodes[k],
                                    i[k],
                                    frameSize,
                                    Ns=Ns,
                                    L=settings.broadcastLength,
                                    lastExpectedIter=numIterations - 1,
                                    broadcastDomain=settings.broadcastDomain)

                if settings.broadcastDomain == 't':
                    zFull[k] = np.concatenate((zFull[k], z[k][Ns:, 0]))     # selecting the first neighbor (index 0)

                # Wipe local buffers
                zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]

                if settings.broadcastDomain == 't':
                    # Build full available observation vector
                    yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                    ytilde[k][:, i[k], :] = yTildeCurr
                    # Go to frequency domain
                    ytildeHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(ytilde[k][:, i[k], :] * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                    ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]      # Keep only positive frequencies
                elif settings.broadcastDomain == 'f':
                    # Broadcasting done in frequency-domain
                    yLocalHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yLocalCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                    ytildeHat[k][:, i[k], :] = np.concatenate((yLocalHatCurr[:numFreqLines, :], 1 / winWOLAanalysis.sum() * z[k]), axis=1)
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Build local observations vector ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

                # Compensate SROs
                if settings.compensateSROs:
                    # Account for buffer flags
                    extraPhaseShiftFactor = np.zeros(dimYTilde[k])
                    for q in range(len(neighbourNodes[k])):
                        if not np.isnan(bufferFlags[q]):
                            extraPhaseShiftFactor[yLocalCurr.shape[-1] + q] = bufferFlags[q] * settings.broadcastLength
                            # ↑↑↑ if `bufferFlags[q] == 0`, `extraPhaseShiftFactor = 0` and no additional phase shift is applied
                        else:
                            # From `process_incoming_signals_buffers`: "Not enough samples anymore due to cumulated SROs effect, skip update"
                            skipUpdate = True
                    # Complete phase shift factors
                    phaseShiftFactors[k] += extraPhaseShiftFactor
                    # Apply phase shift factors
                    ytildeHat[k][:, i[k], :] *= np.exp(-1 * 1j * 2 * np.pi / frameSize * np.outer(np.arange(numFreqLines), phaseShiftFactors[k]))

                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Spatial covariance matrices updates ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                Ryytilde[k], Rnntilde[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :],
                                                Ryytilde[k], Rnntilde[k], settings.expAvgBeta, oVADframes[i[k]])
                if settings.computeLocalEstimate:
                    # Local observations only
                    Ryylocal[k], Rnnlocal[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :dimYLocal[k]],
                                                    Ryylocal[k], Rnnlocal[k], settings.expAvgBeta, oVADframes[i[k]])
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Spatial covariance matrices updates ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                
                # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
                if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
                    startUpdates[k] = True

                if startUpdates[k] and not settings.bypassFilterUpdates and not skipUpdate:
                    # No `for`-loop versions
                    if settings.performGEVD:    # GEVD update
                        wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
                        if settings.computeLocalEstimate:
                            wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)

                    else:                       # regular update (no GEVD)
                        wTilde[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryytilde[k], Rnntilde[k], settings.referenceSensor)
                        if settings.computeLocalEstimate:
                            wLocal[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryylocal[k], Rnnlocal[k], settings.referenceSensor)

                else:
                    # Do not update the filter coefficients
                    wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                    if settings.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
                    if skipUpdate:
                        print(f'Node {k+1}: {i[k] + 1}^th update skipped.')
                if settings.bypassFilterUpdates:
                    print('!! User-forced bypass of filter coefficients updates !!')


                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Update external filters (for broadcasting)  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                wTildeExternal[k] = settings.expAvgBeta * wTildeExternal[k] + (1 - settings.expAvgBeta) * wTildeExternalTarget[k]
                # Update targets
                if t - lastExternalFiltUpdateInstant[k] >= settings.timeBtwExternalFiltUpdates:
                    wTildeExternalTarget[k] = (1 - alphaExternalFilters) * wTildeExternalTarget[k] + alphaExternalFilters * wTilde[k][:, i[k] + 1, :yLocalCurr.shape[-1]]
                    # Update last external filter update instant [s]
                    lastExternalFiltUpdateInstant[k] = t
                    if settings.printouts.externalFilterUpdates:    # inform user
                        print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {settings.timeBtwExternalFiltUpdates}s)')
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update external filters (for broadcasting)  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


                # Update SRO estimates
                if settings.estimateSROs:

                    residualSROs[k][:, i[k]] = subs.residual_sro_estimation(
                                        resPos=wTilde[k][:, i[k] + 1, :],         # a posteriori estimate
                                        resPri=wTilde[k][:, i[k], :],             # a priori estimate
                                        nLocalSensors=yLocalCurr.shape[-1],
                                        N=frameSize,
                                        Ns=Ns
                                        )

                    resSROneighbors = residualSROs[k][asc.numSensorPerNode[k]:, i[k]]
                    tmp = numFreqLines * resSROneighbors / (1 + resSROneighbors)  # in Taewoong's slides: $N \Delta\hat{\varepsilon}_k / (1 + \Delta\hat{\varepsilon}_k)$
                    a[k] += tmp
                    SROsEstimates[k] += a[k]

                    if k == 0:  # save evolution of SRO estimates
                        SROsEstimatesThroughTime[i[k]] = SROsEstimates[k][0]
                else:
                    SROsEstimates[k] = (settings.SROsppm[k] - settings.SROsppm[neighbourNodes[k]]) * 1e-6    # no data-based dynamic SRO estimation: use oracle knowledge

                # Update phase shifts for SRO compensation
                if settings.compensateSROs:
                    for q in range(len(neighbourNodes[k])):
                        # Increment phase shift factor recursively....
                        phaseShiftFactors[k][yLocalCurr.shape[-1] + q] += SROsEstimates[k][q] * Ns   
                        #                               ↑↑↑↑ `yLocalCurr.shape[-1] + q` because first `yLocalCurr.shape[-1]` indices are linked to the local sensors

                # ----- Compute desired signal chunk estimate -----
                dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dhat[:, i[k], k] = dhatCurr
                # Transform back to time domain (WOLA processing)
                dChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatCurr, frameSize)
                d[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunk)   # overlap and add construction of output time-domain signal
                
                if settings.computeLocalEstimate:
                    # Local observations only
                    dhatLocalCurr = np.einsum('ij,ij->i', wLocal[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :dimYLocal[k]])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                    dhatLocal[:, i[k], k] = dhatLocalCurr
                    # Transform back to time domain (WOLA processing)
                    dLocalChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatLocalCurr, frameSize)
                    dLocal[idxBegChunk:idxEndChunk, k] += np.real_if_close(dLocalChunk)   # overlap and add construction of output time-domain signal
                # -----------------------------------------------------------------------
                
                # Increment DANSE iteration index
                i[k] += 1

    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=dur))}s.')
    print(f'(Real-time processing factor: {np.round(masterClock[-1] / dur, 4)})')

    # Export empty array if local desired signal estimate was not computed
    if (dLocal == 0).all():
        dLocal = np.array([])

    # Profiling
    profiler.stop()
    profiler.print()

    # fig = plt.figure(figsize=(8,4))
    # ax = fig.add_subplot(111)
    # ax.plot(SROsEstimatesThroughTime)
    # ax.grid()
    # ax.set_xlabel('DANSE iteration index')
    # plt.tight_layout()	
    # plt.show()

    # fig = plt.figure(figsize=(6,4))
    # for ii in range(len(residualSROs)):
    #     ax = fig.add_subplot(len(residualSROs) * 100 + 10 + ii + 1)
    #     for jj in range(residualSROs[ii].shape[0]):
    #         if jj >= asc.numSensorPerNode[ii]:
    #             lab = f'Neighbor #{jj+1 - asc.numSensorPerNode[ii]}'
    #             ax.plot(residualSROs[ii][jj, :].T * 10 ** 6, label=lab)
    #             # Show actual relative SRO
    #             trueSRO = (SROsEstimates[ii][jj - asc.numSensorPerNode[ii]]) * 10 ** 6
    #             ax.hlines(trueSRO, xmin=0, xmax=residualSROs[ii].shape[1], color='k')
    #     ax.grid()
    #     ax.set_ylabel('Residual SRO [ppm]')
    #     ax.set_title(f"Node {ii+1}'s estimates")
    #     ax.legend()
    #     # ax.set_ylim([-100, 100])
    # ax.set_xlabel('DANSE iteration i')
    # plt.tight_layout()	
    # plt.show()

    # stop = 1

    return d, dLocal

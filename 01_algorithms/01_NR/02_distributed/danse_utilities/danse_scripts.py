
import numpy as np
import time, datetime
from . import classes
from . import danse_subfcns as subs
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import copy

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


def danse_simultaneous(yin, asc: classes.AcousticScenario, settings: classes.ProgramSettings,
        oVAD, timeInstants, masterClockNodeIdx,
        referenceSpeechOnly=None):
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
    sroData : danse_subfcns.SROdata object
        Data on SRO estimation / compensation (see danse_subfcns.sro_subfcns module for details)
    """

    # Hard-coded parameters
    printingInterval = 2            # [s] minimum time between two "% done" printouts
    alphaExternalFilters = 0.5      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html
    # alphaExternalFilters = 0      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html

    # Profiling
    profiler = Profiler()
    profiler.start()
    
    # Initialization (extracting/defining useful quantities)
    _, winWOLAanalysis, winWOLAsynthesis,\
        frameSize, Ns, numIterations, _,\
            neighbourNodes = subs.danse_init(yin, settings, asc)

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, masterClockNodeIdx]     # reference clock

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Arrays initialization ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    lk = np.zeros(asc.numNodes, dtype=int)          # node-specific broadcast index
    i = np.zeros(asc.numNodes, dtype=int)           # !node-specific! DANSE iteration index
    #
    wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
    wTildeExternal = []                             # external filter coefficients - used for broadcasting only, updated every `settings.timeBtwExternalFiltUpdates` seconds
    wTildeExternalTarget = []                       # target external filter coefficients - used for broadcasting only, updated every `settings.timeBtwExternalFiltUpdates` 
    wIR = []                                        # time-domain filters, for compressing and broadcasting 
    Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
    Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
    yyH = []                                        # instantaneous autocorrelation product
    yyHuncomp = []                                  # instantaneous autocorrelation product with no SRO compensation (`CohDrift`, `loop='open'` SRO estimation setting)
    ryd = []                                        # cross-correlation between observations and estimations
    ytilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                  # local full observation vectors, frequency-domain
    ytildeHatCentr = []                             # local full observation vectors, frequency-domain, as if centralized processing
    ytildeHatUncomp = []                            # local full observation vectors, frequency-domain, SRO-uncompensated
    z = []                                          # current-iteration compressed signals used in DANSE update
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    zLocal = []                                     # last local frame (entire frame) of compressed signal (used for time-domain chunk-wise broadcast: overlap-add)
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    bufferFlagsAccumulated = []                     # buffer flags (0, -1, or +1) accumulated through time - for when buffers over- or under-flow
    bufferLengths = []                              # node-specific number of samples in each buffer
    phaseShiftFactors = []                          # phase-shift factors for SRO compensation (only used if `settings.compensateSROs == True`)
    a = []
    tauSROsEstimates = []                           # SRO-induced time shift estimates per node (for each neighbor)
    SROsResiduals = []                              # SRO residuals per node (for each neighbor)
    SROsEstimates = []                              # SRO estimates per node (for each neighbor)
    SROsEstimatesAccumulated = []
    residualSROs = []                               # residual SROs, for each node, across DANSE iterations
    avgCohProdDWACD = []                            # average coherence product coming out of DWACD processing (SRO estimation)
    avgProdResiduals = []                           # average residuals product coming out of filter-shift processing (SRO estimation)
    avgProdResidualsRyy = []                        # average residuals product coming out of filter-shift processing (SRO estimation) - for Ryy only
    avgProdResidualsRnn = []                        # average residuals product coming out of filter-shift processing (SRO estimation) - for Rnn only
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
    numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
    if settings.computeLocalEstimate:
        wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
        Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
        Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
        dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
    if settings.computeCentralizedEstimate:
        wCentr = []                                     # filter coefficients - using all observations (centralized processing)
        RnnCentr = []                                   # autocorrelation matrix when VAD=0 - using all observations (centralized processing)
        RyyCentr = []                                   # autocorrelation matrix when VAD=1 - using all observations (centralized processing)
        
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        wtmp = np.zeros((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex)
        wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTilde.append(wtmp)
        wtmp = np.zeros((numFreqLines, sum(asc.sensorToNodeTags == k + 1)), dtype=complex)
        wtmp[:, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTildeExternal.append(wtmp)
        wTildeExternalTarget.append(wtmp)
        wtmp = np.zeros((frameSize, sum(asc.sensorToNodeTags == k + 1)))
        wtmp[0, 0] = 1   # initialize time-domain (real-valued) filter as Dirac for first sensor signal
        wIR.append(wtmp)
        #
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytildeHat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        ytildeHatCentr.append(np.zeros((numFreqLines, numIterations, asc.numSensors), dtype=complex))
        ytildeHatUncomp.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        #
        sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
        Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
        yyH.append(np.zeros((numIterations, numFreqLines, dimYTilde[k], dimYTilde[k]), dtype=complex))
        yyHuncomp.append(np.zeros((numIterations, numFreqLines, dimYTilde[k], dimYTilde[k]), dtype=complex))
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros((numIterations, len(neighbourNodes[k]))))    # init all buffer flags at 0 (assuming no over- or under-flow)
        bufferFlagsAccumulated.append(np.zeros((numIterations, len(neighbourNodes[k]))))
        bufferLengths.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))
        #
        if settings.broadcastDomain == 'wholeChunk_td':
            z.append(np.empty((frameSize, 0), dtype=float))
        elif settings.broadcastDomain == 'wholeChunk_fd':
            z.append(np.empty((numFreqLines, 0), dtype=complex))
        elif settings.broadcastDomain == 'fewSamples_td':
            z.append(np.empty((frameSize, 0), dtype=float))
        #
        zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
        zLocal.append(np.array([])) 
        # SRO stuff vvv
        phaseShiftFactors.append(np.zeros(dimYTilde[k]))   # initiate phase shift factors as 0's (no phase shift)
        a.append(np.zeros(dimYTilde[k]))   # 
        tauSROsEstimates.append(np.zeros(len(neighbourNodes[k])))
        SROsResiduals.append(np.zeros(len(neighbourNodes[k])))
        SROsEstimates.append(np.zeros(len(neighbourNodes[k])))
        SROsEstimatesAccumulated.append(np.zeros(len(neighbourNodes[k])))
        residualSROs.append(np.zeros((dimYTilde[k], numIterations)))
        avgCohProdDWACD.append(np.zeros((len(neighbourNodes[k]), frameSize), dtype=complex))
        avgProdResiduals.append(np.zeros((frameSize, len(neighbourNodes[k])), dtype=complex))
        avgProdResidualsRyy.append(np.zeros((frameSize, len(neighbourNodes[k])), dtype=complex))
        avgProdResidualsRnn.append(np.zeros((frameSize, len(neighbourNodes[k])), dtype=complex))
        # If local estimate is to be computed vvv
        if settings.computeLocalEstimate:
            dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
            wtmp = np.zeros((numFreqLines, numIterations + 1, dimYLocal[k]), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wLocal.append(wtmp)
            sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
            Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
        # If centralized estimate is to be computed vvv
        if settings.computeCentralizedEstimate:
            wtmp = np.zeros((numFreqLines, numIterations + 1, asc.numSensors), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wCentr.append(wtmp)
            sliceCentr = np.finfo(float).eps * np.eye(asc.numSensors, dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            RnnCentr.append(np.tile(sliceCentr, (numFreqLines, 1, 1)))                    # noise only
            RyyCentr.append(np.tile(sliceCentr, (numFreqLines, 1, 1)))                    # speech + noise
    # Desired signal estimate [frames x frequencies x nodes]
    dhat = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
    d = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhat`
    if settings.computeLocalEstimate:
        dhatLocal = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)
        dLocal = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatLocal`
        dLocal2 = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatLocal`
    else:
        dhatLocal, dLocal = [], []
    if settings.computeCentralizedEstimate:
        dhatCentr = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using all observations (centralized processing)
        dCentr = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatCentr`
    else:
        dhatCentr, dCentr = [], []
    zReconstructed = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `z` (last neighbor, for each node)
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
    nInternalFilterUpdates = np.zeros(asc.numNodes)
    # Booleans
    startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
    # --- SRO estimation ---
    # Expected number of SRO estimate updates via DWACD during total signal length
    nDWACDSROupdates = numIterations // settings.asynchronicity.dwacd.nFiltUpdatePerSeg
    # DANSE filter update indices corresponding to DWACD SRO estimate updates
    dwacdSROupdateIndices = (settings.asynchronicity.dwacd.temp_dist +\
                            settings.asynchronicity.dwacd.seg_len +\
                            np.arange(nDWACDSROupdates) * settings.asynchronicity.dwacd.seg_shift)\
                            // settings.stftEffectiveFrameLen
    # DANSE filter update indices corresponding to "Filter-shift" SRO estimate updates
    cohDriftSROupdateIndices = np.arange(start=settings.asynchronicity.cohDriftMethod.startAfterNupdates +\
                            settings.asynchronicity.cohDriftMethod.estEvery,
                            stop=numIterations,
                            step=settings.asynchronicity.cohDriftMethod.estEvery)
    # DWACD segments (i.e., SRO estimate update) counters (for each node)
    dwacdSegIdx = np.zeros(asc.numNodes, dtype=int)
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Arrays initialization ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # Prepare events to be considered in main `for`-loop
    eventsMatrix, fs = subs.get_events_matrix(timeInstants,
                                            frameSize,
                                            Ns,
                                            settings.broadcastLength,
                                            settings.broadcastDomain
                                            )

    # Extra variables -- TEMPORARY: TODO: -- to be treated and integrated more neatly
    SROresidualThroughTime = []
    SROestimateThroughTime = []
    numPSFupdates = []
    phaseShiftFactorThroughTime = np.zeros((numIterations))
    zFull = []
    for k in range(asc.numNodes):
        SROresidualThroughTime.append(np.zeros(numIterations))
        SROestimateThroughTime.append(np.zeros(numIterations))
        numPSFupdates.append(np.zeros(len(neighbourNodes[k])))
        zFull.append(np.empty((0,)))
    dLocalChunks = np.zeros((frameSize, numIterations, asc.numNodes))
    dHatLocalChunks = np.zeros((numFreqLines, numIterations, asc.numNodes))

    # External filter updates (for broadcasting)
    lastExternalFiltUpdateInstant = np.zeros(asc.numNodes)   # [s]
    previousTDfilterUpdate = np.zeros(asc.numNodes)

    tprint = -printingInterval      # printouts timing

    
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
    if referenceSpeechOnly is not None:
        referenceSpeechOnly = np.concatenate((np.zeros((Ns, referenceSpeechOnly.shape[-1])), referenceSpeechOnly), axis=0)
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
    # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 

    t0 = time.perf_counter()        # loop timing
    # Loop over time instants when one or more event(s) occur(s)
    for idxInstant, events in enumerate(eventsMatrix):

        # Parse event matrix (+ inform user)
        t, eventTypes, nodesConcerned = subs.events_parser(events, startUpdates, printouts=settings.printouts.events_parser)

        if t > tprint + printingInterval and settings.printouts.danseProgress:
            print(f'----- t = {np.round(t, 2)}s | {np.round(t / masterClock[-1] * 100)}% done -----')
            tprint = t

        eventIndices = np.arange(len(eventTypes))

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Deal with broadcasts first ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idxBroadcastEvent in eventIndices[[True if ii == 'broadcast' else False for ii in eventTypes]]:

            # Node index
            k = int(nodesConcerned[idxBroadcastEvent])# Extract current local data chunk

            yLocalForBC, _, _ = subs.local_chunk_for_broadcast(yin[:, asc.sensorToNodeTags == k+1],
                                                        t,
                                                        fs[k],
                                                        settings.broadcastDomain,
                                                        frameSize, k, lk[k])

            # Perform broadcast -- update all relevant buffers in network
            zBuffer, wIR[k], previousTDfilterUpdate[k], zLocal[k] = subs.broadcast(
                        t,
                        k,
                        fs[k],
                        settings.broadcastLength,
                        yLocalForBC,
                        wTildeExternal[k],
                        frameSize,
                        neighbourNodes,
                        lk,
                        zBuffer,
                        settings.broadcastDomain,
                        winWOLAanalysis,
                        winWOLAsynthesis,
                        winShift=int(frameSize * (1 - settings.stftFrameOvlp)),
                        previousTDfilterUpdate=previousTDfilterUpdate[k],
                        updateTDfilterEvery=settings.updateTDfilterEvery,
                        wIRprevious=wIR[k],
                        zTDpreviousFrame=zLocal[k],
                    )
            

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Deal with local estimates next ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idxUpdateEvent in eventIndices[[True if ii == 'update' else False for ii in eventTypes]]:
            
            # Node index
            k = int(nodesConcerned[idxUpdateEvent])
            
            # Extract current local data chunk !!! USING local_chunk_for_broadcast() BECAUSE NO NEED FOR Ns DELAY HERE !!!
            yLocalCurr, idxBegChunk, idxEndChunk = subs.local_chunk_for_broadcast(yin[:, asc.sensorToNodeTags == k+1],
                                                        t,
                                                        fs[k],
                                                        settings.broadcastDomain,
                                                        frameSize, k, lk[k])

            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 
            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 
            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 
            if settings.computeLocalEstimate:
                yLocalHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yLocalCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)

                # Local observations only
                dhatLocalCurr = np.einsum('ij,ij->i', wLocal[k][:, i[k], :].conj(), yLocalHatCurr[:numFreqLines, :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                # dhatLocalCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k], :dimYLocal[k]].conj(), yLocalHatCurr[:numFreqLines, :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dhatLocal[:, i[k], k] = dhatLocalCurr
                # Transform back to time domain (WOLA processing)
                dLocalChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatLocalCurr, frameSize)
                if i[k] == 0 and idxEndChunk < frameSize:   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
                    dLocal[idxBegChunk:idxEndChunk, k] += np.real_if_close(dLocalChunk[-idxEndChunk:])   # overlap and add construction of output time-domain signal
                else:
                    dLocal[idxBegChunk:idxEndChunk, k] += np.real_if_close(dLocalChunk)   # overlap and add construction of output time-domain signal

                # Local observations only
                # dhatLocalCurr2 = np.einsum('ij,ij->i', wTilde[k][:, i[k], :dimYLocal[k]].conj(), yLocalHatCurr[:numFreqLines, :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dhatLocalCurr2 = np.einsum('ij,ij->i', wLocal[k][:, i[k], :].conj(), yLocalHatCurr[:numFreqLines, :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dHatLocalChunks[:, i[k], k] = dhatLocalCurr2 # <-----------
                # Transform back to time domain (WOLA processing)
                dLocalChunk2 = np.real_if_close(winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatLocalCurr2, frameSize))
                dLocalChunks[:, i[k], k] = dLocalChunk2 # <-----------
                if i[k] == 0 and idxEndChunk < frameSize:   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
                    dLocal2[idxBegChunk:idxEndChunk, k] += dLocalChunk2[-idxEndChunk:]   # overlap and add construction of output time-domain signal
                else:
                    dLocal2[idxBegChunk:idxEndChunk, k] += dLocalChunk2   # overlap and add construction of output time-domain signal
            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 
            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 
            # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING # COMPUTE LOCAL ESTIMATE BEFORE ANYTHING 



        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Deal with updates next ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idxUpdateEvent in eventIndices[[True if ii == 'update' else False for ii in eventTypes]]:

            # Node index
            k = int(nodesConcerned[idxUpdateEvent])
            
            skipUpdate = False  # flag to skip update if needed

            # Extract current local data chunk
            yLocalCurr, idxBegChunk, idxEndChunk = subs.local_chunk_for_update(yin[:, asc.sensorToNodeTags == k+1],
                                                        t,
                                                        fs[k],
                                                        settings.broadcastDomain,
                                                        frameSize,
                                                        Ns,
                                                        settings.broadcastLength)
            
            if settings.computeCentralizedEstimate:
                yCentrCurr, _, _ = subs.local_chunk_for_update(yin,
                                                            t,
                                                            fs[k],
                                                            settings.broadcastDomain,
                                                            frameSize,
                                                            Ns,
                                                            settings.broadcastLength)

            # Compute VAD
            VADinFrame = oVAD[np.amax([idxBegChunk, 0]):idxEndChunk]
            oVADframes[i[k]] = sum(VADinFrame == 0) <= len(VADinFrame) / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1
            if oVADframes[i[k]]:    # Count number of spatial covariance matrices updates
                numUpdatesRyy[k] += 1
            else:
                numUpdatesRnn[k] += 1

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Build local observations vector ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # Process buffers
            z[k], bufferFlags[k][i[k], :] = subs.process_incoming_signals_buffers(
                                zBuffer[k],
                                z[k],
                                neighbourNodes[k],
                                i[k],
                                frameSize,
                                Ns=Ns,
                                L=settings.broadcastLength,
                                lastExpectedIter=numIterations - 1,
                                broadcastDomain=settings.broadcastDomain)

            # Accumulate flags
            bufferFlagsAccumulated[k][i[k]:, :] = bufferFlagsAccumulated[k][i[k] - 1, :] + bufferFlags[k][i[k], :]

            # Save full z vector for DEBUGGING PURPOSES
            if k == 0:
                zFull[k] = np.concatenate((zFull[k], z[k][Ns:, 0]))

            # Wipe local buffers
            zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]

            if settings.broadcastDomain == 'wholeChunk_fd':
                # Broadcasting done in frequency-domain
                yLocalHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yLocalCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                ytildeHat[k][:, i[k], :] = np.concatenate((yLocalHatCurr[:numFreqLines, :], 1 / winWOLAanalysis.sum() * z[k]), axis=1)
            elif settings.broadcastDomain in ['wholeChunk_td', 'fewSamples_td']:
                # Build full available observation vector
                yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                ytilde[k][:, i[k], :] = yTildeCurr
                # Go to frequency domain
                ytildeHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(ytilde[k][:, i[k], :] * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]      # Keep only positive frequencies

            # Centralized estimate
            if settings.computeCentralizedEstimate:
                if settings.broadcastDomain != 'wholeChunk_fd':
                    raise ValueError('NOT YET IMPLEMENTED FOR CENTRALIZED ESTIMATE')
                ytildeHatCentrCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yCentrCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                ytildeHatCentr[k][:, i[k], :] = ytildeHatCentrCurr[:numFreqLines, :]

            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
            if referenceSpeechOnly is not None:
                for q in range(len(neighbourNodes[k])):
                    # referenceSpeechOnlyCurr = referenceSpeechOnly[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == q + 1]
                    # referenceSpeechOnlyCurr = referenceSpeechOnlyCurr[:,0]  # select 1st sensor only
                    # referenceSpeechOnlyCurr = referenceSpeechOnlyCurr[:, np.newaxis]  # select 1st sensor only
                    # #
                    # refCurrHat = 1 / winWOLAanalysis.sum() * np.fft.fft(referenceSpeechOnlyCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                    # # vvv for debugging only (TD not needed otherwise, here)
                    # refCurrBackToTD = np.real_if_close(winWOLAsynthesis.sum() * winWOLAsynthesis[:, np.newaxis] * subs.back_to_time_domain(refCurrHat[:numFreqLines, :], frameSize))
                    # ytildeHat[k][:, i[k], dimYLocal[k] + q] = np.squeeze(refCurrHat[:numFreqLines])      # Keep only positive frequencies
                    # ytildeHat[k][:, i[k], dimYLocal[k] + q] = dHatLocalChunks[:, i[k], neighbourNodes[k][q]]
                    tmp = 1 / winWOLAanalysis.sum() * np.fft.fft(dLocal2[idxBegChunk:idxEndChunk, neighbourNodes[k][q]] * winWOLAanalysis, frameSize, axis=0)
                    ytildeHat[k][:, i[k], dimYLocal[k] + q] = tmp[:numFreqLines]
            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING 
            # # DEBUGGING DEBUGGING DEBUGGING DEBUGGING
            # # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Build local observations vector ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            if t > 4 and oVADframes[i[k]]:
                stop = 1

            # Save uncompensated \tilde{y} (for coherence-drift-based SRO estimation)
            ytildeHatUncomp[k][:, i[k], :] = copy.copy(ytildeHat[k][:, i[k], :])
            yyHuncomp[k][i[k], :, :, :] = np.einsum('ij,ik->ijk', ytildeHatUncomp[k][:, i[k], :], ytildeHatUncomp[k][:, i[k], :].conj())
            # Compensate SROs
            if settings.asynchronicity.compensateSROs:
                # Account for buffer flags
                extraPhaseShiftFactor = np.zeros(dimYTilde[k])
                for q in range(len(neighbourNodes[k])):
                    if not np.isnan(bufferFlags[k][i[k], q]):
                        extraPhaseShiftFactor[yLocalCurr.shape[-1] + q] = bufferFlags[k][i[k], q] * settings.broadcastLength
                        # ↑↑↑ if `bufferFlags[k][i[k], q] == 0`, `extraPhaseShiftFactor = 0` and no additional phase shift is applied
                    else:
                        # From `process_incoming_signals_buffers`: "Not enough samples anymore due to cumulated SROs effect, skip update"
                        skipUpdate = True
                # Complete phase shift factors
                phaseShiftFactors[k] += extraPhaseShiftFactor
                # Save for plotting
                if k == 0:
                    phaseShiftFactorThroughTime[i[k]:] = phaseShiftFactors[k][yLocalCurr.shape[-1] + q]
                # Apply phase shift factors
                ytildeHat[k][:, i[k], :] *= np.exp(-1 * 1j * 2 * np.pi / frameSize * np.outer(np.arange(numFreqLines), phaseShiftFactors[k]))


            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Spatial covariance matrices updates ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            Ryytilde[k], Rnntilde[k], yyH[k][i[k], :, :, :] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :],
                                            Ryytilde[k], Rnntilde[k], settings.expAvgBeta, oVADframes[i[k]])
            if settings.computeLocalEstimate:
                # Local observations only
                Ryylocal[k], Rnnlocal[k], _ = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :dimYLocal[k]],
                                                Ryylocal[k], Rnnlocal[k], settings.expAvgBeta, oVADframes[i[k]])
            if settings.computeCentralizedEstimate:
                # All observations
                RyyCentr[k], RnnCentr[k], _ = subs.spatial_covariance_matrix_update(ytildeHatCentr[k][:, i[k], :],
                                                RyyCentr[k], RnnCentr[k], settings.expAvgBeta, oVADframes[i[k]])
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Spatial covariance matrices updates ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            
            # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
            if (not startUpdates[k]) and (numUpdatesRyy[k] >= minNumAutocorrUpdates) and (numUpdatesRnn[k] >= minNumAutocorrUpdates):
                startUpdates[k] = True
            # elif not startUpdates[k]:
            #     # As long as we have not started DANSE updates,
            #     # prevent updating of the time-domain filters used for broadcasting.
            #     previousTDfilterUpdate[k] = t

            if startUpdates[k] and not settings.bypassFilterUpdates and not skipUpdate:
                # No `for`-loop versions 
                if settings.performGEVD:    # GEVD update
                    wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
                    if settings.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)
                    if settings.computeCentralizedEstimate:
                        wCentr[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(RyyCentr[k], RnnCentr[k], settings.GEVDrank, settings.referenceSensor)

                else:                       # regular update (no GEVD)
                    wTilde[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryytilde[k], Rnntilde[k], settings.referenceSensor)
                    if settings.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryylocal[k], Rnnlocal[k], settings.referenceSensor)
                    if settings.computeCentralizedEstimate:
                        wCentr[k][:, i[k] + 1, :] = subs.perform_update_noforloop(RyyCentr[k], RnnCentr[k], settings.referenceSensor)
                # Count the number of internal filter updates
                nInternalFilterUpdates[k] += 1  
            else:
                # Do not update the filter coefficients
                wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                if settings.computeLocalEstimate:
                    wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
                if settings.computeCentralizedEstimate:
                    wCentr[k][:, i[k] + 1, :] = wCentr[k][:, i[k], :]
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
            
            # DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG
            # DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG
            # DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG
            # DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG
            wTildeExternal[k] = wTilde[k][:, i[k] + 1, :yLocalCurr.shape[-1]]
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update external filters (for broadcasting)  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Update SRO estimates  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            if settings.asynchronicity.estimateSROs == 'CohDrift':
                
                ld = settings.asynchronicity.cohDriftMethod.segLength

                if i[k] in cohDriftSROupdateIndices:

                    flagFirstSROEstimate = False
                    if i[k] == np.amin(cohDriftSROupdateIndices):
                        flagFirstSROEstimate = True     # let `cohdrift_sro_estimation()` know that this is the 1st SRO estimation round

                    # Residuals method
                    for q in range(len(neighbourNodes[k])):

                        idxq = yLocalCurr.shape[-1] + q     # index of the compressed signal from node `q` inside `yyH`
                        if settings.asynchronicity.cohDriftMethod.loop == 'closed':
                            # Use SRO-compensated correlation matrix entries (closed-loop SRO est. + comp.)
                            cohPosteriori = (yyH[k][i[k], :, 0, idxq]
                                                    / np.sqrt(yyH[k][i[k], :, 0, 0] * yyH[k][i[k], :, idxq, idxq]))     # a posteriori coherence
                            cohPriori = (yyH[k][i[k] - ld, :, 0, idxq]
                                                    / np.sqrt(yyH[k][i[k] - ld, :, 0, 0] * yyH[k][i[k] - ld, :, idxq, idxq]))     # a priori coherence
                        elif settings.asynchronicity.cohDriftMethod.loop == 'open':
                            # Use SRO-_un_compensated correlation matrix entries (open-loop SRO est. + comp.)
                            cohPosteriori = (yyHuncomp[k][i[k], :, 0, idxq]
                                                    / np.sqrt(yyHuncomp[k][i[k], :, 0, 0] * yyHuncomp[k][i[k], :, idxq, idxq]))     # a posteriori coherence
                            cohPriori = (yyHuncomp[k][i[k] - ld, :, 0, idxq]
                                                    / np.sqrt(yyHuncomp[k][i[k] - ld, :, 0, 0] * yyHuncomp[k][i[k] - ld, :, idxq, idxq]))     # a priori coherence

                        sroRes, apr = subs.cohdrift_sro_estimation(
                                            wPos=cohPosteriori,
                                            wPri=cohPriori,
                                            avg_res_prod=avgProdResiduals[k][:, q],
                                            Ns=Ns,
                                            ld=ld,
                                            method=settings.asynchronicity.cohDriftMethod.estimationMethod,
                                            alpha=settings.asynchronicity.cohDriftMethod.alpha,
                                            flagFirstSROEstimate=flagFirstSROEstimate,
                                            )
                    
                        SROsResiduals[k][q] = sroRes
                        avgProdResiduals[k][:, q] = apr

                SROresidualThroughTime[k][i[k]:] = SROsResiduals[k][0]  # save for plotting / debugging
                    
            elif settings.asynchronicity.estimateSROs == 'DWACD':
                # Dynamic Weighted-Average Coherence Drift [T. Gburrek, 2021]

                if i[k] in dwacdSROupdateIndices:

                    for q in range(len(neighbourNodes[k])):
                        sroEst, acp = subs.dwacd_sro_estimation(
                                sigSTFT=ytildeHatUncomp[k][:, :i[k]+1, yLocalCurr.shape[-1] + q],    # compressed signal from `q`-th neighbour
                                ref_sigSTFT=ytildeHatUncomp[k][:, :i[k]+1, 0],   # local sensor signal
                                activity_sig=oVADframes[:i[k]+1],    
                                activity_ref_sig=oVADframes[:i[k]+1],
                                paramsDWACD=settings.asynchronicity.dwacd,
                                seg_idx=dwacdSegIdx[k],
                                sro_est=SROsEstimates[k][q],     # previous SRO estimate
                                # sro_est=0,     # previous SRO estimate
                                avg_coh_prod=avgCohProdDWACD[k][q, :]
                        )
                        SROsEstimates[k][q] = sroEst
                        avgCohProdDWACD[k][q, :] = acp

                    dwacdSegIdx[k] += 1

            elif settings.asynchronicity.estimateSROs == 'Oracle':
                # no data-based dynamic SRO estimation: use oracle knowledge
                SROsEstimates[k] = (settings.asynchronicity.SROsppm[neighbourNodes[k]] -\
                                    settings.asynchronicity.SROsppm[k]) * 1e-6
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update SRO estimates  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # Update phase shifts for SRO compensation
            if settings.asynchronicity.compensateSROs:
                for q in range(len(neighbourNodes[k])):
                    if settings.asynchronicity.estimateSROs == 'CohDrift':
                        if settings.asynchronicity.cohDriftMethod.loop == 'closed':
                            # Increment estimate using SRO residual
                            SROsEstimates[k][q] += SROsResiduals[k][q] / (1 + SROsResiduals[k][q]) *\
                                settings.asynchronicity.cohDriftMethod.alphaEps
                        elif settings.asynchronicity.cohDriftMethod.loop == 'open':
                            # Use SRO "residual" as estimates
                            SROsEstimates[k][q] = SROsResiduals[k][q] / (1 + SROsResiduals[k][q])
                            # SROsEstimates[k][q] += bufferFlags[k][i[k], q] / Ns
                    # Save estimate evolution for later plotting
                    SROestimateThroughTime[k][i[k]:] = SROsEstimates[k][0]
                    # Increment phase shift factor recursively
                    phaseShiftFactors[k][yLocalCurr.shape[-1] + q] -= SROsEstimates[k][q] * Ns  # <-- valid directly for oracle SRO ``estimation''
                    # Keep track of number of phase shift factor updates
                    numPSFupdates[k][q] += 1

            # ----- Compute desired signal chunk estimate -----
            dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
            dhat[:, i[k], k] = dhatCurr
            # Transform back to time domain (WOLA processing)
            dChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatCurr, frameSize)
            if i[k] == 0 and idxEndChunk < frameSize:   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
                d[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunk[-idxEndChunk:])   # overlap and add construction of output time-domain signal
            else:
                d[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunk)   # overlap and add construction of output time-domain signal
            # ----- Compute desired signal chunk estimate [CENTRALIZED] -----
            if settings.computeCentralizedEstimate:
                dhatCurrCentr = np.einsum('ij,ij->i', wCentr[k][:, i[k] + 1, :].conj(), ytildeHatCentr[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dhatCentr[:, i[k], k] = dhatCurrCentr
                # Transform back to time domain (WOLA processing)
                dChunkCentr = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatCurrCentr, frameSize)
                if i[k] == 0 and idxEndChunk < frameSize:   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
                    dCentr[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunkCentr[-idxEndChunk:])   # overlap and add construction of output time-domain signal
                else:
                    dCentr[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunkCentr)   # overlap and add construction of output time-domain signal
            
            # Reconstruct (via WOLA synthesis) full TD version of z-signal
            zHatCurr = ytildeHat[k][:, i[k], -1]
            # Transform back to time domain (WOLA processing)
            zChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(zHatCurr, frameSize)
            if i[k] == 0 and idxEndChunk < frameSize:   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
                zReconstructed[idxBegChunk:idxEndChunk, k] += np.real_if_close(zChunk[-idxEndChunk:])   # overlap and add construction of output time-domain signal
            else:
                zReconstructed[idxBegChunk:idxEndChunk, k] += np.real_if_close(zChunk)   # overlap and add construction of output time-domain signal
            # -----------------------------------------------------------------------
            
            # Increment DANSE iteration index
            i[k] += 1


    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(zReconstructed[:, 1] + 2, label='$z_1$')
    # axes.plot(dLocal2[:, 0], label='$\hat{d}_{\mathrm{local}, 1}$ -- computed with wTilde[:localSensors]')
    # axes.plot(dLocal[:, 0] - 2, label='$\hat{d}_{\mathrm{local}, 1}$ -- computed with wLocal')
    # # ynodeq = yin[:, asc.sensorToNodeTags == 1]
    # # axes.plot(ynodeq[:, 0], '--')
    # axes.legend()
    # axes.grid()
    # plt.tight_layout()	
    # plt.show()

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(wLocal[0][100,:,0])
    # axes.plot(wTilde[0][100,:,0])
    # axes.legend()
    # axes.grid()
    # plt.tight_layout()
    # plt.show()


    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=dur))}s.')
    print(f'(Real-time processing factor: {np.round(masterClock[-1] / dur, 4)})')

    # Export empty array if local desired signal estimate was not computed
    if (dLocal == 0).all():
        dLocal = np.array([])

    # Prep SRO data for export
    sroData = classes.SROdata(estMethod=settings.asynchronicity.estimateSROs,
                            compensation=settings.asynchronicity.compensateSROs,
                            residuals=SROresidualThroughTime,
                            estimate=SROestimateThroughTime,
                            groundTruth=settings.asynchronicity.SROsppm / 1e6)

    # Profiling
    profiler.stop()
    profiler.print()


    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(asc.numNodes,1)
    # fig.set_size_inches(8.5, 3.5)
    # for k in range(asc.numNodes):
    #     axes[k].plot(masterClock, d[:,k], label='DANSE estimate')
    #     axes[k].plot(masterClock, dLocal[:,k], alpha=.5, label='Local estimate')
    #     axes[k].plot(masterClock, dCentr[:,k], alpha=.5, label='Centralized estimate')
    #     axes[k].grid()
    #     axes[k].legend()
    #     axes[k].set_title(f'Node {k+1}')
    # plt.tight_layout()
    # plt.show()

    # Debugging
    fig = sroData.plotSROdata(xaxistype='time', fs=fs[0], Ns=Ns)
    # fig = sroData.plotSROdata(xaxistype='iterations', fs=fs[0], Ns=Ns)
    # plt.show()

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(phaseShiftFactorThroughTime)
    # axes.grid()
    # plt.tight_layout()	
    # plt.show()

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(bufferFlags[0][:, 0])
    # axes.plot(bufferFlags[1][:, 0])
    # axes.grid()
    # plt.tight_layout()	
    # plt.show()

    # if 0:
    #     fig.savefig(f'U:/py/sounds-phd/01_algorithms/01_NR/02_distributed/res/forPresentations/202206_SROestimation/out_alphaeps{int(settings.asynchronicity.cohDriftMethod.alphaEps * 100)}.png')
    #     fig.savefig(f'U:/py/sounds-phd/01_algorithms/01_NR/02_distributed/res/forPresentations/202206_SROestimation/out_alphaeps{int(settings.asynchronicity.cohDriftMethod.alphaEps * 100)}.pdf')

    stop = 1

    return d, dLocal, dCentr, sroData

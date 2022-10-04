
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


def danse_simultaneous(yin, asc: classes.AcousticScenario, s: classes.ProgramSettings,
        oVAD, timeInstants, masterClockNodeIdx,
        referenceSpeechOnly=None):
    """Wrapper for Simultaneous-Node-Updating DANSE (rs-DANSE) [2].

    Parameters
    ----------
    yin : [Nt x settings.Ns] np.ndarray of floats
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    s : ProgramSettings object
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
    tStartForMetrics : [Nn x 1] np.ndarray (float)
        Start instants (per node) for the computation of speech enhancement metrics.
        --> Avoiding metric bias due to first DANSE iterations where the filters have not converged yet.
    """

    # Hard-coded parameters
    alphaExternalFilters = 0.5      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html

    # Profiling
    profiler = Profiler()
    profiler.start()
    
    # Initialization (extracting/defining useful quantities)
    _, winWOLAanalysis, winWOLAsynthesis, numIter, _, neighbourNodes = subs.danse_init(yin, s, asc)
    normFactWOLA = winWOLAanalysis.sum()
    # normFactWOLA = 1

    # Pre-process certain input parameters
    if referenceSpeechOnly is not None:
        referenceSpeechOnly = np.concatenate((np.zeros((s.Ns, referenceSpeechOnly.shape[-1])), referenceSpeechOnly), axis=0)

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
    yTilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                  # local full observation vectors, frequency-domain
    yTildeCentr = []                                # local full observation vectors, time-domain, as if centralized processing
    ytildeHatCentr = []                             # local full observation vectors, frequency-domain, as if centralized processing
    ytildeHatUncomp = []                            # local full observation vectors, frequency-domain, SRO-uncompensated
    z = []                                          # current-iteration compressed signals used in DANSE update
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    zLocal = []                                     # last local frame (entire frame) of compressed signal (used for time-domain chunk-wise broadcast: overlap-add)
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
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
    oVADframes = np.zeros(numIter)            # oracle VAD per time frame
    numFreqLines = int(s.DFTsize / 2 + 1)           # number of frequency lines (only positive frequencies)
    if s.computeLocalEstimate:
        wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
        Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
        Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
        dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
    if s.computeCentralizedEstimate:
        wCentr = []                                     # filter coefficients - using all observations (centralized processing)
        RnnCentr = []                                   # autocorrelation matrix when VAD=0 - using all observations (centralized processing)
        RyyCentr = []                                   # autocorrelation matrix when VAD=1 - using all observations (centralized processing)
        
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        wtmp = np.zeros((numFreqLines, numIter + 1, dimYTilde[k]), dtype=complex)
        wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTilde.append(wtmp)
        wtmp = np.zeros((numFreqLines, sum(asc.sensorToNodeTags == k + 1)), dtype=complex)
        wtmp[:, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTildeExternal.append(wtmp)
        wTildeExternalTarget.append(wtmp)
        wtmp = np.zeros((2 * s.DFTsize - 1, sum(asc.sensorToNodeTags == k + 1)))
        wtmp[s.DFTsize, 0] = 1   # initialize time-domain (real-valued) filter as Dirac for first sensor signal
        wIR.append(wtmp)
        #
        yTilde.append(np.zeros((s.DFTsize, numIter, dimYTilde[k])))
        ytildeHat.append(np.zeros((numFreqLines, numIter, dimYTilde[k]), dtype=complex))
        yTildeCentr.append(np.zeros((s.DFTsize, numIter, asc.numSensors)))
        ytildeHatCentr.append(np.zeros((numFreqLines, numIter, asc.numSensors), dtype=complex))
        ytildeHatUncomp.append(np.zeros((numFreqLines, numIter, dimYTilde[k]), dtype=complex))
        #
        rng = np.random.default_rng(s.randSeed)
        # sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        sliceTilde = np.finfo(float).eps * (rng.random((dimYTilde[k], dimYTilde[k])) + 1j * rng.random((dimYTilde[k], dimYTilde[k]))) 
        Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
        Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
        yyH.append(np.zeros((numIter, numFreqLines, dimYTilde[k], dimYTilde[k]), dtype=complex))
        yyHuncomp.append(np.zeros((numIter, numFreqLines, dimYTilde[k], dimYTilde[k]), dtype=complex))
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros((numIter, len(neighbourNodes[k]))))    # init all buffer flags at 0 (assuming no over- or under-flow)
        bufferLengths.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))
        #
        if s.broadcastDomain == 'wholeChunk_td':
            z.append(np.empty((s.DFTsize, 0), dtype=float))
        elif s.broadcastDomain == 'wholeChunk_fd':
            z.append(np.empty((numFreqLines, 0), dtype=complex))
        elif s.broadcastDomain == 'fewSamples_td':
            z.append(np.empty((s.DFTsize, 0), dtype=float))
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
        residualSROs.append(np.zeros((dimYTilde[k], numIter)))
        avgCohProdDWACD.append(np.zeros((len(neighbourNodes[k]), s.DFTsize), dtype=complex))
        avgProdResiduals.append(np.zeros((s.DFTsize, len(neighbourNodes[k])), dtype=complex))
        avgProdResidualsRyy.append(np.zeros((s.DFTsize, len(neighbourNodes[k])), dtype=complex))
        avgProdResidualsRnn.append(np.zeros((s.DFTsize, len(neighbourNodes[k])), dtype=complex))
        # If local estimate is to be computed vvv
        if s.computeLocalEstimate:
            dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
            wtmp = np.zeros((numFreqLines, numIter + 1, dimYLocal[k]), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wLocal.append(wtmp)
            sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
            Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
        # If centralized estimate is to be computed vvv
        if s.computeCentralizedEstimate:
            wtmp = np.zeros((numFreqLines, numIter + 1, asc.numSensors), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wCentr.append(wtmp)
            sliceCentr = np.finfo(float).eps * np.eye(asc.numSensors, dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            RnnCentr.append(np.tile(sliceCentr, (numFreqLines, 1, 1)))                    # noise only
            RyyCentr.append(np.tile(sliceCentr, (numFreqLines, 1, 1)))                    # speech + noise
    # Desired signal estimate [frames x frequencies x nodes]
    dhat = np.zeros((numFreqLines, numIter, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
    d = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhat`
    if s.computeLocalEstimate:
        dhatLocal = np.zeros((numFreqLines, numIter, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)
        dLocal = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatLocal`
    else:
        dhatLocal, dLocal = [], []
    if s.computeCentralizedEstimate:
        dhatCentr = np.zeros((numFreqLines, numIter, asc.numNodes), dtype=complex)   # using all observations (centralized processing)
        dCentr = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatCentr`
    else:
        dhatCentr, dCentr = [], []
    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
    nInternalFilterUpdates = np.zeros(asc.numNodes)
    # Booleans
    startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
    # --- SRO estimation ---
    # DANSE filter update indices corresponding to "Filter-shift" SRO estimate updates
    cohDriftSROupdateIndices = np.arange(start=s.asynchronicity.cohDriftMethod.startAfterNupdates +\
                            s.asynchronicity.cohDriftMethod.estEvery,
                            stop=numIter,
                            step=s.asynchronicity.cohDriftMethod.estEvery)
    flagIterations = [[] for _ in range(asc.numNodes)]
    flagInstants = [[] for _ in range(asc.numNodes)]
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Arrays initialization ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # Prepare events to be considered in main `for`-loop
    eventsMatrix, fs = subs.get_events_matrix(timeInstants,
                                            s.DFTsize,
                                            s.Ns,
                                            s.broadcastLength,
                                            s.broadcastDomain,
                                            efficient=s.efficiency.efficientSpSBC
                                            )

    # Extra variables TODO: -- to be treated and integrated more neatly
    SROresidualThroughTime = []
    SROestimateThroughTime = []
    phaseShiftFactorThroughTime = np.zeros((numIter))
    for k in range(asc.numNodes):
        SROresidualThroughTime.append(np.zeros(numIter))
        SROestimateThroughTime.append(np.zeros(numIter))
    tStartForMetrics = np.full(asc.numNodes, fill_value=None)  # start instant for the computation of speech enhancement metrics
    if s.efficiency.efficientSpSBC:
        lastBroadcastInstant = np.zeros(asc.numNodes)

    # External filter updates (for broadcasting)
    lastExternalFiltUpdateInstant = np.zeros(asc.numNodes)   # [s]
    previousTDfilterUpdate = np.zeros(asc.numNodes)

    tprint = -s.printouts.progressPrintingInterval      # printouts timing

    t0 = time.perf_counter()        # loop timing
    # Loop over time instants when one or more event(s) occur(s)
    for _, events in enumerate(eventsMatrix):

        # Parse event matrix (+ inform user)
        t, eventTypes, nodesConcerned = subs.events_parser(events, startUpdates,
                                    printouts=s.printouts.events_parser,
                                    doNotPrintBCs=s.printouts.events_parser_noBC)

        if t > tprint + s.printouts.progressPrintingInterval and s.printouts.danseProgress:
            print(f'----- t = {np.round(t, 2)}s | {np.round(t / masterClock[-1] * 100, 2)}% done -----')
            tprint = t

        eventIndices = np.arange(len(eventTypes))

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Deal with broadcasts first ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idxBroadcastEvent in eventIndices[[True if e == 'broadcast' else False for e in eventTypes]]:

            # Node index
            k = int(nodesConcerned[idxBroadcastEvent])  # Extract current local data chunk

            yLocalForBC = subs.local_chunk_for_broadcast(yin[:, asc.sensorToNodeTags == k+1],
                                                        t,
                                                        fs[k],
                                                        s.broadcastDomain,
                                                        s.DFTsize, k, lk[k])
            
            if s.efficiency.efficientSpSBC:
                # Count samples recorded since the last broadcast at node `k`
                nSamplesSinceLastBroadcast = ((timeInstants[:, k] > lastBroadcastInstant[k]) & (timeInstants[:, k] <= t)).sum()
                lastBroadcastInstant[k] = t
                currL = nSamplesSinceLastBroadcast
            else:
                currL = s.broadcastLength

            # Perform broadcast -- update all relevant buffers in network
            zBuffer, wIR[k], previousTDfilterUpdate[k], zLocal[k] = subs.broadcast(
                        t=t,
                        k=k,
                        fs=fs[k],
                        L=currL,
                        yk=yLocalForBC,
                        w=wTildeExternal[k],
                        n=s.DFTsize,
                        neighbourNodes=neighbourNodes,
                        lk=lk,
                        zBuffer=zBuffer,
                        broadcastDomain=s.broadcastDomain,
                        winWOLAanalysis=winWOLAanalysis,
                        winWOLAsynthesis=winWOLAsynthesis,
                        winShift=s.Ns,
                        previousTDfilterUpdate=previousTDfilterUpdate[k],
                        updateTDfilterEvery=s.updateTDfilterEvery,
                        wIRprevious=wIR[k],
                        zTDpreviousFrame=zLocal[k]
                        )

            if i[k] > 20:
                stop = 1

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Deal with updates next ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        for idxUpdateEvent in eventIndices[[True if e == 'update' else False for e in eventTypes]]:

            # Node index
            k = int(nodesConcerned[idxUpdateEvent])
            
            skipUpdate = False  # flag to skip update if needed

            # Extract current local data chunk
            yLocalCurr, idxBegChunk, idxEndChunk = subs.local_chunk_for_update(yin[:, asc.sensorToNodeTags == k+1],
                                                        t,
                                                        fs[k],
                                                        s.broadcastDomain,
                                                        s.DFTsize,
                                                        s.Ns,
                                                        s.broadcastLength)
            
            if s.computeCentralizedEstimate:
                # Extract current local data chunk
                yCentrCurr, _, _ = subs.local_chunk_for_update(yin,
                                                            t,
                                                            fs[k],
                                                            s.broadcastDomain,
                                                            s.DFTsize,
                                                            s.Ns,
                                                            s.broadcastLength)

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
                                s.DFTsize,
                                Ns=s.Ns,
                                L=s.broadcastLength,
                                lastExpectedIter=numIter - 1,
                                broadcastDomain=s.broadcastDomain,
                                t=t)

            # Wipe local buffers
            zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]

            if s.broadcastDomain == 'wholeChunk_fd':
                # Broadcasting done in frequency-domain
                yLocalHatCurr = 1 / normFactWOLA * np.fft.fft(yLocalCurr * winWOLAanalysis[:, np.newaxis], s.DFTsize, axis=0)
                ytildeHat[k][:, i[k], :] = np.concatenate((yLocalHatCurr[:numFreqLines, :], 1 / normFactWOLA * z[k]), axis=1)
            elif s.broadcastDomain in ['wholeChunk_td', 'fewSamples_td']:
                # Build full available observation vector
                yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                yTilde[k][:, i[k], :] = yTildeCurr
                # Go to frequency domain
                ytildeHatCurr = 1 / normFactWOLA * np.fft.fft(yTilde[k][:, i[k], :] * winWOLAanalysis[:, np.newaxis], s.DFTsize, axis=0)
                ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]      # Keep only positive frequencies

            # Centralized estimate ↓
            if s.computeCentralizedEstimate:
                if 'wholeChunk' not in s.broadcastDomain:
                    raise ValueError('NOT YET IMPLEMENTED FOR CENTRALIZED ESTIMATE')
                yTildeCentr[k][:, i[k], :] = yCentrCurr
                ytildeHatCentrCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yCentrCurr * winWOLAanalysis[:, np.newaxis], s.DFTsize, axis=0)
                ytildeHatCentr[k][:, i[k], :] = ytildeHatCentrCurr[:numFreqLines, :]

            # Account for buffer flags
            extraPhaseShiftFactor = np.zeros(dimYTilde[k])
            for q in range(len(neighbourNodes[k])):
                if not np.isnan(bufferFlags[k][i[k], q]):
                    extraPhaseShiftFactor[yLocalCurr.shape[-1] + q] = bufferFlags[k][i[k], q] * s.broadcastLength
                    # ↑↑↑ if `bufferFlags[k][i[k], q] == 0`, `extraPhaseShiftFactor = 0` and no additional phase shift is applied
                    if bufferFlags[k][i[k], q] != 0:
                        flagIterations[k].append(i[k])  # keep flagging iterations in memory
                        flagInstants[k].append(t)       # keep flagging instants in memory
                else:
                    # From `process_incoming_signals_buffers`: "Not enough samples anymore due to cumulated SROs effect, skip update"
                    skipUpdate = True
            # Save uncompensated \tilde{y} for coherence-drift-based SRO estimation
            ytildeHatUncomp[k][:, i[k], :] = copy.copy(ytildeHat[k][:, i[k], :])
            yyHuncomp[k][i[k], :, :, :] = np.einsum('ij,ik->ijk', ytildeHatUncomp[k][:, i[k], :], ytildeHatUncomp[k][:, i[k], :].conj())
            # Compensate SROs
            if s.asynchronicity.compensateSROs:
                # Complete phase shift factors
                phaseShiftFactors[k] += extraPhaseShiftFactor
                if k == 0:  # Save for plotting
                    phaseShiftFactorThroughTime[i[k]:] = phaseShiftFactors[k][yLocalCurr.shape[-1] + q]
                # Apply phase shift factors
                ytildeHat[k][:, i[k], :] *= np.exp(-1 * 1j * 2 * np.pi / s.DFTsize * np.outer(np.arange(numFreqLines), phaseShiftFactors[k]))


            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Spatial covariance matrices updates ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            Ryytilde[k], Rnntilde[k], yyH[k][i[k], :, :, :] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :],
                                            Ryytilde[k], Rnntilde[k], s.expAvgBeta, oVADframes[i[k]])
            if s.computeLocalEstimate:
                # Local observations only
                Ryylocal[k], Rnnlocal[k], _ = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :dimYLocal[k]],
                                                Ryylocal[k], Rnnlocal[k], s.expAvgBeta, oVADframes[i[k]])
            if s.computeCentralizedEstimate:
                # All observations
                RyyCentr[k], RnnCentr[k], _ = subs.spatial_covariance_matrix_update(ytildeHatCentr[k][:, i[k], :],
                                                RyyCentr[k], RnnCentr[k], s.expAvgBeta, oVADframes[i[k]])
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Spatial covariance matrices updates ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            
            # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
            if not startUpdates[k] and numUpdatesRyy[k] > minNumAutocorrUpdates and numUpdatesRnn[k] > minNumAutocorrUpdates:
                startUpdates[k] = True

            if startUpdates[k] and not s.bypassFilterUpdates and not skipUpdate:
                if k == s.referenceSensor and nInternalFilterUpdates[k] == 0:
                    # Save first update instant (for, e.g., SRO plot)
                    firstDANSEupdateRefSensor = t
                # No `for`-loop versions 
                if s.performGEVD:    # GEVD update
                    wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], s.GEVDrank, s.referenceSensor)
                    if s.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], s.GEVDrank, s.referenceSensor)
                    if s.computeCentralizedEstimate:
                        wCentr[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(RyyCentr[k], RnnCentr[k], s.GEVDrank, s.referenceSensor)

                else:                       # regular update (no GEVD)
                    wTilde[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryytilde[k], Rnntilde[k], s.referenceSensor)
                    if s.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryylocal[k], Rnnlocal[k], s.referenceSensor)
                    if s.computeCentralizedEstimate:
                        wCentr[k][:, i[k] + 1, :] = subs.perform_update_noforloop(RyyCentr[k], RnnCentr[k], s.referenceSensor)
                # Count the number of internal filter updates
                nInternalFilterUpdates[k] += 1  

                # Useful export for enhancement metrics computations
                if nInternalFilterUpdates[k] >= s.minFiltUpdatesForMetricsComputation and tStartForMetrics[k] is None:
                    if s.asynchronicity.compensateSROs and s.asynchronicity.estimateSROs == 'CohDrift':
                        # Make sure SRO compensation has started
                        if nInternalFilterUpdates[k] > s.asynchronicity.cohDriftMethod.startAfterNupdates:
                            tStartForMetrics[k] = t
                    else:
                        tStartForMetrics[k] = t
            else:
                # Do not update the filter coefficients
                wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                if s.computeLocalEstimate:
                    wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
                if s.computeCentralizedEstimate:
                    wCentr[k][:, i[k] + 1, :] = wCentr[k][:, i[k], :]
                if skipUpdate:
                    print(f'Node {k+1}: {i[k] + 1}^th update skipped.')
            if s.bypassFilterUpdates:
                print('!! User-forced bypass of filter coefficients updates !!')

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Update external filters (for broadcasting)  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            wTildeExternal[k] = s.expAvgBeta * wTildeExternal[k] + (1 - s.expAvgBeta) * wTildeExternalTarget[k]
            # Update targets
            if t - lastExternalFiltUpdateInstant[k] >= s.timeBtwExternalFiltUpdates:
                wTildeExternalTarget[k] = (1 - alphaExternalFilters) * wTildeExternalTarget[k] + alphaExternalFilters * wTilde[k][:, i[k] + 1, :yLocalCurr.shape[-1]]
                # Update last external filter update instant [s]
                lastExternalFiltUpdateInstant[k] = t
                if s.printouts.externalFilterUpdates:    # inform user
                    print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {s.timeBtwExternalFiltUpdates}s)')
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update external filters (for broadcasting)  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Update SRO estimates  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            sroOut, avgProdResiduals[k] = subs.update_sro_estimates(
                        s,  # settings
                        iter=i[k],
                        nLocalSensors=yLocalCurr.shape[-1],
                        cohDriftSROupdateIndices=cohDriftSROupdateIndices,
                        neighbourNodes=neighbourNodes[k],
                        yyH=yyH[k],
                        yyHuncomp=yyHuncomp[k],
                        avgProdRes=avgProdResiduals[k],
                        oracleSRO=s.asynchronicity.SROsppm[k],
                        bufferFlagPos=np.sum(bufferFlags[k][:(i[k] + 1), :], axis=0)\
                            * s.broadcastLength,
                        bufferFlagPri=np.sum(bufferFlags[k][:(i[k] - s.asynchronicity.cohDriftMethod.segLength + 1), :], axis=0)\
                            * s.broadcastLength,
                        )

            if s.asynchronicity.estimateSROs == 'CohDrift':
                SROsResiduals[k] = sroOut
                # Save through time (for plotting)
                SROresidualThroughTime[k][i[k]:] = sroOut[0]
            elif s.asynchronicity.estimateSROs == 'Oracle':
                SROsEstimates[k] = sroOut
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update SRO estimates  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # Update phase shifts for SRO compensation
            if s.asynchronicity.compensateSROs:
                for q in range(len(neighbourNodes[k])):
                    if s.asynchronicity.estimateSROs == 'CohDrift':
                        if s.asynchronicity.cohDriftMethod.loop == 'closed':
                            # Increment estimate using SRO residual
                            SROsEstimates[k][q] += SROsResiduals[k][q] / (1 + SROsResiduals[k][q]) *\
                                s.asynchronicity.cohDriftMethod.alphaEps
                        elif s.asynchronicity.cohDriftMethod.loop == 'open':
                            # Use SRO "residual" as estimates
                            SROsEstimates[k][q] = SROsResiduals[k][q] / (1 + SROsResiduals[k][q])
                    # Save estimate evolution for later plotting
                    SROestimateThroughTime[k][i[k]:] = SROsEstimates[k][0]
                    # Increment phase shift factor recursively
                    phaseShiftFactors[k][yLocalCurr.shape[-1] + q] -= SROsEstimates[k][q] * s.Ns  # <-- valid directly for oracle SRO ``estimation''

            # ----- Compute desired signal chunk estimate [DANSE, GLOBAL] -----
            dhat[:, i[k], k], tmp = subs.get_desired_signal(
                wTilde[k][:, i[k] + 1, :],
                ytildeHat[k][:, i[k], :],
                winWOLAsynthesis,
                d[idxBegChunk:idxEndChunk, k],
                normFactWOLA,
                winShift=s.Ns,
                desSigEstChunkLength=s.Ns,
                processingType=s.desSigProcessingType,
                yTD=yTilde[k][:, i[k], :]
                )
            if s.desSigProcessingType == 'wola':
                d[idxBegChunk:idxEndChunk, k] = tmp
            elif s.desSigProcessingType == 'conv':
                d[idxEndChunk - s.Ns:idxEndChunk, k] = tmp
            if s.computeLocalEstimate:
                # ----- Compute desired signal chunk estimate [LOCAL] -----
                dhatLocal[:, i[k], k], tmp = subs.get_desired_signal(
                    wLocal[k][:, i[k] + 1, :],
                    ytildeHat[k][:, i[k], :dimYLocal[k]],
                    winWOLAsynthesis,
                    dLocal[idxBegChunk:idxEndChunk, k],
                    normFactWOLA,
                    winShift=s.Ns,
                    desSigEstChunkLength=s.Ns,
                    processingType=s.desSigProcessingType,
                    yTD=yTilde[k][:, i[k], :dimYLocal[k]]
                    )
                if s.desSigProcessingType == 'wola':
                    dLocal[idxBegChunk:idxEndChunk, k] = tmp
                elif s.desSigProcessingType == 'conv':
                    dLocal[idxEndChunk - s.Ns:idxEndChunk, k] = tmp
            if s.computeCentralizedEstimate:
                # ----- Compute desired signal chunk estimate [CENTRALIZED] -----
                dhatCentr[:, i[k], k], tmp = subs.get_desired_signal(
                    wCentr[k][:, i[k] + 1, :],
                    ytildeHatCentr[k][:, i[k], :],
                    winWOLAsynthesis,
                    dCentr[idxBegChunk:idxEndChunk, k],
                    normFactWOLA,
                    winShift=s.Ns,
                    desSigEstChunkLength=s.Ns,
                    processingType=s.desSigProcessingType,
                    yTD=yTildeCentr[k][:, i[k], :]
                    )
                if s.desSigProcessingType == 'wola':
                    dCentr[idxBegChunk:idxEndChunk, k] = tmp
                elif s.desSigProcessingType == 'conv':
                    dCentr[idxEndChunk - s.Ns:idxEndChunk, k] = tmp
            
            # Increment DANSE iteration index
            i[k] += 1

    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=dur))}.')
    print(f'(Real-time processing factor: {np.round(masterClock[-1] / dur, 4)})')

    # Prep SRO data for export
    sroData = classes.SROdata(estMethod=s.asynchronicity.estimateSROs,
                            compensation=s.asynchronicity.compensateSROs,
                            residuals=SROresidualThroughTime,
                            estimate=SROestimateThroughTime,
                            groundTruth=s.asynchronicity.SROsppm / 1e6,
                            flagIterations=flagIterations,
                            neighbourIndex=np.array([n[0] for n in neighbourNodes]))

    # Profiling
    profiler.stop()
    if s.printouts.profiler:
        profiler.print()

    # Debugging
    fig = sroData.plotSROdata(xaxistype='both', fs=fs[0], Ns=s.Ns, firstUp=firstDANSEupdateRefSensor)
    # fig = sroData.plotSROdata(xaxistype='iterations', fs=fs[0], Ns=s.Ns)
    # plt.show(block=False)
    stop = 1

    return d, dLocal, dCentr, sroData, tStartForMetrics, firstDANSEupdateRefSensor

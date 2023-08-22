import os
import copy
import resampy
import numpy as np
import soundfile as sf
from numba import njit
import scipy.linalg as la
from utils.online_mwf import *
from utils.online_danse import *
from dataclasses import dataclass, field

@dataclass
class ScriptParameters:
    signalType: str = 'speech'  
    # ^^^ 'speech', 'noise_real', 'noise_complex', ...
    #     ... 'interrupt_noise_real', 'interrupt_noise_complex'.
    interruptionDuration: float = 0.1  # seconds
    interruptionPeriod: float = 0.5  # seconds
    targetSignalSpeechFile: str = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
    nSensors: int = 3
    nNodes: int = 3
    Mk: list[int] = field(default_factory=lambda: None)  # if None, randomly assign sensors to nodes
    selfNoisePower: float = 1
    minDuration: float = 1
    maxDuration: float = 10
    nDurationsBatch: int = 30
    fs: float = 8e3
    nMC: int = 10
    exportFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/figs/for20230823marcUpdate'
    taus: list[float] = field(default_factory=lambda: [2.])
    b: float = 0.1  # factor for determining beta from tau (online processing)
    toCompute: list[str] = field(default_factory=lambda: [
        'mwf_batch',
        'gevdmwf_batch',
        'danse_sim_batch',
        'gevddanse_sim_batch',
    ])
    seed: int = 0
    wolaParams: WOLAparameters = WOLAparameters(
        fs=fs,
    )
    VADwinLength: float = 0.02  # seconds
    VADenergyDecrease_dB: float = 10  # dB
    # Booleans vvvv
    randomDelays: bool = False
    showDeltaPerNode: bool = False
    useBatchModeFusionVectorsInOnlineDanse: bool = False
    ignoreFusionForSSNodes: bool = True  # in DANSE, ignore fusion vector for single-sensor nodes
    exportFigures: bool = True
    verbose: bool = True
    useVAD: bool = True  # use VAD for online processing of nonsstationary signals
    loadVadIfPossible: bool = True  # if True, load VAD from file if possible
    # Strings vvvv
    vadFilesFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/vad_files'

    def __post_init__(self):
        if any(['wola' in t for t in self.toCompute]) and\
            'complex' in self.signalType:
                raise ValueError('WOLA not implemented for complex-valued signals')
        self.durations = np.linspace(
            self.minDuration,
            self.maxDuration,
            self.nDurationsBatch,
            endpoint=True
        )
        if self.wolaParams.fs != self.fs:
            self.wolaParams.fs = self.fs
        if 'interrupt' in self.signalType and\
            self.minDuration <= self.interruptionPeriod:
            raise ValueError('`minDuration` should be > `interruptionPeriod`')


def compute_filter(
        noisySignals: np.ndarray,
        cleanSigs: np.ndarray,
        noiseOnlySigs: np.ndarray,
        type='mwf_batch',
        rank=1,
        channelToNodeMap=None,  # only used for DANSE
        wolaParams: WOLAparameters=WOLAparameters(),
        batchFusionInOnline=False,
        noSSfusion=True,
        verbose=True,
        vad=None,
    ):
    """
    [1] Santiago Ruiz, Toon van Waterschoot and Marc Moonen, "Distributed
    combined acoustic echo cancellation and noise reduction in wireless
    acoustic sensor and actuator networks" - 2022
    """

    if vad is None:
        nSamples = cleanSigs.shape[0]
        Ryy = 1 / nSamples * noisySignals.T @ noisySignals.conj()
        Rnn = 1 / nSamples * noiseOnlySigs.T @ noiseOnlySigs.conj()
    else:
        # Include VAD knowledge in batch-mode estimates of `Ryy` and `Rnn`.
        # Only keep frames where all sensors VADs are active.
        vadBool = np.all(vad.astype(bool), axis=1)
        Ryy = 1 / np.sum(vadBool) *\
            noisySignals[vadBool, :].T @ noisySignals[vadBool, :].conj()
        Rnn = 1 / np.sum(~vadBool) *\
            noisySignals[~vadBool, :].T @ noisySignals[~vadBool, :].conj()

    nSensors = cleanSigs.shape[1]

    if type == 'mwf_batch':
        RyyInv = np.linalg.inv(Ryy)
        if np.iscomplex(cleanSigs).any():
            w = np.zeros((nSensors, nSensors), dtype=np.complex128)
        else:
            w = np.zeros((nSensors, nSensors))
        for n in range(nSensors):
            e = np.zeros(nSensors)  # Selection vector
            e[n] = 1
            ryd = (Ryy - Rnn) @ e
            w[:, n] = RyyInv @ ryd
    elif type == 'mwf_online':
        w = run_online_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='regular',
            p=wolaParams,
            verbose=verbose,
            vad=vad
        )
    elif type == 'mwf_wola':
        w = run_wola_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='regular',
            p=wolaParams,
            verbose=verbose,
            vad=vad
        )
    elif type == 'gevdmwf_batch':
        sigma, Xmat = la.eigh(Ryy, Rnn)
        idx = np.flip(np.argsort(sigma))
        sigma = sigma[idx]
        Xmat = Xmat[:, idx]
        Qmat = np.linalg.inv(Xmat.T.conj())
        Dmat = np.zeros((nSensors, nSensors))
        Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
        w = Xmat @ Dmat @ Qmat.T.conj()   # see eq. (24) in [1]
    elif type == 'gevdmwf_online':
        w = run_online_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='gevd',
            rank=rank,
            p=wolaParams,
            verbose=verbose,
            vad=vad
        )
    elif type == 'gevdmwf_wola':
        w = run_wola_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='gevd',
            rank=rank,
            p=wolaParams,
            verbose=verbose,
            vad=vad
        )
    elif 'danse' in type:
        kwargsBatch = {
            'x': cleanSigs,
            'n': noiseOnlySigs,
            'referenceSensorIdx': 0,
            'channelToNodeMap': channelToNodeMap,
            'filterType': 'gevd' if 'gevd' in type else 'regular',
            'rank': rank,
            'verbose': verbose,
            'vad': vad
        }
        if 'sim' in type:
            kwargsBatch['nodeUpdatingStrategy'] = 'simultaneous'
        else:
            kwargsBatch['nodeUpdatingStrategy'] = 'sequential'

        # Keyword-arguments for online-mode DANSE function
        kwargsOnline = copy.deepcopy(kwargsBatch)
        kwargsOnline['p'] = wolaParams
        kwargsOnline['ignoreFusionForSSNodes'] = noSSfusion
    
        if batchFusionInOnline:
            wBatchNetWide = run_danse(**kwargsBatch)
            kwargsOnline['batchModeNetWideFilters'] = wBatchNetWide
        
        if 'wola' in type:
            w = run_wola_danse(**kwargsOnline)
        elif 'online' in type:
            w = run_online_danse(**kwargsOnline)
        else:
            w = run_danse(**kwargsBatch)
    return w


def run_danse(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        referenceSensorIdx=0,
        verbose=True,
        vad=None
    ):

    maxIter = 100
    tol = 1e-9
    # Get noisy signal
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    # Determine data type (complex or real)
    myDtype = np.complex128 if np.iscomplex(x).any() else np.float64
    # Initialize
    w = []
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        wCurr = np.zeros((nSensorsPerNode + nNodes - 1, maxIter), dtype=myDtype)
        wCurr[referenceSensorIdx, :] = 1
        w.append(wCurr)
    idxNodes = np.arange(nNodes)
    idxUpdatingNode = 0
    wNet = np.zeros((x.shape[1], maxIter, nNodes), dtype=myDtype)

    for k in range(nNodes):
        # Determine reference sensor index
        idxRef = np.where(channelToNodeMap == k)[0][referenceSensorIdx]
        wNet[idxRef, 0, k] = 1  # set reference sensor weight to 1
    # Run DANSE
    if nodeUpdatingStrategy == 'sequential':
        label = 'Batch DANSE [seq NU]'
    else:
        label = 'Batch DANSE [sim NU]'
    if filterType == 'gevd':
        label += ' [GEVD]'
    for iter in range(maxIter):
        if verbose:
            print(f'{label} iteration {iter+1}/{maxIter}')
        # Compute fused signals from all sensors
        fusedSignals = np.zeros((x.shape[0], nNodes), dtype=myDtype)
        fusedSignalsNoiseOnly = np.zeros((x.shape[0], nNodes), dtype=myDtype)
        for q in range(nNodes):
            yq = y[:, channelToNodeMap == q]
            fusedSignals[:, q] = yq @ w[q][:yq.shape[1], iter].conj()
            nq = n[:, channelToNodeMap == q]
            fusedSignalsNoiseOnly[:, q] = nq @ w[q][:nq.shape[1], iter].conj()
            
        for k in range(nNodes):
            # Get y tilde
            yTilde: np.ndarray = np.concatenate((
                y[:, channelToNodeMap == k],
                fusedSignals[:, idxNodes != k]
            ), axis=1)
            nTilde: np.ndarray = np.concatenate((
                n[:, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, idxNodes != k]
            ), axis=1)

            # Compute covariance matrices
            if vad is None:
                Ryy = 1 / yTilde.shape[0] * yTilde.T @ yTilde.conj()
                Rnn = 1 / nTilde.shape[0] * nTilde.T @ nTilde.conj()
            else:
                vadBool = np.all(vad.astype(bool), axis=1)
                Ryy = 1 / np.sum(vadBool) *\
                    yTilde[vadBool, :].T @ yTilde[vadBool, :].conj()
                Rnn = 1 / np.sum(~vadBool) *\
                    yTilde[~vadBool, :].T @ yTilde[~vadBool, :].conj()
            
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False

            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    e = np.zeros(w[k].shape[0])
                    e[referenceSensorIdx] = 1  # selection vector
                    ryd = (Ryy - Rnn) @ e
                    w[k][:, iter + 1] = np.linalg.inv(Ryy) @ ryd
                elif filterType == 'gevd':
                    sigma, Xmat = la.eigh(Ryy, Rnn)
                    idx = np.flip(np.argsort(sigma))
                    sigma = sigma[idx]
                    Xmat = Xmat[:, idx]
                    Qmat = np.linalg.inv(Xmat.T.conj())
                    Dmat = np.zeros((Ryy.shape[0], Ryy.shape[0]))
                    Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
                    e = np.zeros(Ryy.shape[0])
                    e[referenceSensorIdx] = 1
                    w[k][:, iter + 1] = Xmat @ Dmat @ Qmat.T.conj() @ e
            else:
                w[k][: , iter + 1] = w[k][:, iter]  # keep old filter
            
        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes

        # Compute network-wide filters
        for k in range(nNodes):
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(x.shape[1]):
                # Node index corresponding to channel `m`
                currNode = channelToNodeMap[m]
                # Count channel index within node
                c = channelCount[currNode]
                if currNode == k:
                    # Use local filter coefficient
                    wNet[m, iter + 1, k] = w[currNode][c, iter + 1]
                else:
                    nChannels_k = np.sum(channelToNodeMap == k)
                    gkq = w[k][nChannels_k + neighborCount, iter + 1]
                    wNet[m, iter + 1, k] = w[currNode][c, iter] * gkq
                channelCount[currNode] += 1

                if currNode != k and c == np.sum(channelToNodeMap == currNode) - 1:
                    neighborCount += 1
        
        # Check convergence
        if iter > 0:
            diff = 0
            for k in range(nNodes):
                diff += np.mean(np.abs(w[k][:, iter + 1] - w[k][:, iter]))
            if diff < tol:
                print(f'Convergence reached after {iter+1} iterations')
                break

    # Format for output
    wOut = np.zeros((x.shape[1], iter + 2, nNodes), dtype=myDtype)
    for k in range(nNodes):
        wOut[:, :, k] = wNet[:, :(iter + 2), k]

    return wOut


def get_clean_signals(
        p: ScriptParameters,
        scalings,
        fsTarget,
        maxDelay=0.1
    ):

    dur = np.amax(p.durations)
    # Load target signal
    vad = None
    if p.signalType == 'speech':
        latentSignal, fs = sf.read(p.targetSignalSpeechFile)
        if fs != fsTarget:
            # Resample
            latentSignal = resampy.resample(
                latentSignal,
                fs,
                fsTarget
            )
        # Truncate
        latentSignal = latentSignal[:int(dur * fsTarget)]
        if p.useVAD:
            # Get VAD
            vad = get_vad(
                x=latentSignal,
                tw=p.VADwinLength,
                eFactdB=p.VADenergyDecrease_dB,
                fs=fsTarget,
                loadIfPossible=p.loadVadIfPossible,
                vadFilesFolder=p.vadFilesFolder
            )
            # Normalize power (including VAD)
            latentSignal /= get_power(
                latentSignal[np.squeeze(vad).astype(bool)]
            )
        else:
            # Normalize power
            latentSignal /= get_power(latentSignal)
    elif 'noise' in p.signalType:
        if 'real' in p.signalType:
            # Generate noise signal (real-valued)
            latentSignal = np.random.normal(size=int(dur * fsTarget))
        elif 'complex' in p.signalType:
            # Generate noise signal (complex-valued)
            latentSignal = np.random.normal(size=int(dur * fsTarget)) +\
                1j * np.random.normal(size=int(dur * fsTarget))
        
        if 'interrupt' in p.signalType:
            # Add `p.interruptionDuration`-long interruptions every
            # `p.interruptionPeriod` seconds 
            nInterruptions = int(dur / p.interruptionPeriod)
            vad = np.ones((latentSignal.shape[0], 1))
            for k in range(nInterruptions):
                idxStart = int(
                    ((k + 1) * p.interruptionPeriod - p.interruptionDuration) *\
                        fsTarget
                )
                idxEnd = int(idxStart + p.interruptionDuration * fsTarget)
                latentSignal[idxStart:idxEnd] = 0
                vad[idxStart:idxEnd, 0] = 0
            # Normalize power (using VAD)
            latentSignal /= get_power(latentSignal[np.squeeze(vad).astype(bool)])
        else:
            # Normalize power
            latentSignal /= get_power(latentSignal)
    
    # Generate clean signals
    mat = np.tile(latentSignal, (p.nSensors, 1)).T
    # Random scalings
    cleanSigs = mat @ np.diag(scalings)  # rank-1 matrix (scalings only)
    # Random delays
    if p.randomDelays:
        for n in range(p.nSensors):
            delay = np.random.randint(0, int(maxDelay * fsTarget))
            cleanSigs[:, n] = np.roll(cleanSigs[:, n], delay)

    return cleanSigs, latentSignal, vad


def get_power(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def get_filters(
        cleanSigs,
        noiseSignals,
        channelToNodeMap,
        gevdRank=1,
        toCompute=[''],
        wolaParams: WOLAparameters=WOLAparameters(),
        taus=[2.],
        durations=[1.],
        b=0.1,
        batchFusionInOnline=False,
        noSSfusion=True,
        verbose=True,
        vad=None
    ):
    """Compute filters."""
    kwargs = {
        'channelToNodeMap': channelToNodeMap,
        'rank': gevdRank,
        'wolaParams': wolaParams,
        'verbose': verbose,
        'batchFusionInOnline': batchFusionInOnline,
        'noSSfusion': noSSfusion,
    }
    filters = {}
    for filterType in toCompute:

        if verbose:
            print(f'Computing {filterType} filter(s)')

        if 'online' in filterType or 'wola' in filterType:
            # Only need to compute for longest duration
            # + computing for several tau's
            nSamples = int(np.amax(durations) * wolaParams.fs)
            noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]
            kwargs['noisySignals'] = noisySignals
            kwargs['cleanSigs'] = cleanSigs[:nSamples, :]
            kwargs['noiseOnlySigs'] = noiseSignals[:nSamples, :]
            kwargs['vad'] = vad[:nSamples, :] if vad is not None else None
            currFiltsAll = []
            # ^^^ shape: (nTaus, nSensors, nIters, nNodes) for 'online'
            # & (nTaus, nSensors, nIters, nFrequencies, nNodes) for 'wola'
            for idxTau in range(len(taus)):
                # Set beta based on tau
                if 'online' in filterType:
                    normFact = kwargs['wolaParams'].nfft
                else:
                    normFact = kwargs['wolaParams'].hop
                kwargs['wolaParams'].betaMwf = np.exp(
                    np.log(b) /\
                        (taus[idxTau] * kwargs['wolaParams'].fs / normFact)
                )
                kwargs['wolaParams'].betaDanse = kwargs['wolaParams'].betaMwf

                currFiltsAll.append(
                    compute_filter(type=filterType, **kwargs)
                )
        
        else:  # <-- batch-mode
            # Only need to compute for a single dummy tau
            # + computing for several p.durations
            currFiltsAll = []
            # ^^^ shape: (nDurations, nSensors, nNodes)
            for dur in durations:
                nSamples = int(np.amax(dur) * wolaParams.fs)
                noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]
                kwargs['noisySignals'] = noisySignals
                kwargs['cleanSigs'] = cleanSigs[:nSamples, :]
                kwargs['noiseOnlySigs'] = noiseSignals[:nSamples, :]
                kwargs['vad'] = vad[:nSamples, :] if vad is not None else None

                currFiltsAll.append(
                    compute_filter(type=filterType, **kwargs)
                )
            
            if 'danse' in filterType:
                # Ensure adequate array dimensions
                maxNiter = np.amax([f.shape[1] for f in currFiltsAll])
                for k in range(len(currFiltsAll)):
                    if currFiltsAll[k].shape[1] < maxNiter:
                        # Pad by repeating the converged value as many times
                        # as needed to reach `maxNiter`
                        toPad = currFiltsAll[k][:, -1, :]
                        currFiltsAll[k] = np.concatenate((
                            currFiltsAll[k],
                            toPad[:, np.newaxis, :].repeat(
                                maxNiter - currFiltsAll[k].shape[1],
                                axis=1
                            )
                        ), axis=1)
        filters[filterType] = np.array(currFiltsAll)

    return filters


def get_metrics(
        nSensors,
        filters,
        scalings,
        sigma_sr,
        sigma_nr,
        channelToNodeMap,
        filterType,
        computeForComparisonWithDANSE=True,
        exportDiffPerFilter=False
    ):
    
    # Compute FAS + SPF (baseline)
    if 'wola' in filterType:
        nBins = sigma_nr.shape[0]  # number of frequency bins
        if np.any(np.iscomplex(filters)):
            baseline = np.zeros((nSensors, nSensors, nBins), dtype=np.complex128)
        else:
            baseline = np.zeros((nSensors, nSensors, nBins))
        for m in range(nSensors):
            rtf = scalings / scalings[m]
            hs = (rtf.T.conj() @ rtf) * sigma_sr[:, m] ** 2
            spf = hs / (hs + sigma_nr[:, m] ** 2)  # spectral post-filter (real-valued)
            baseline[:, m, :] = np.outer(rtf / (rtf.T.conj() @ rtf), spf)  # FAS BF + spectral post-filter
    else:
        if np.any(np.iscomplex(filters)):
            baseline = np.zeros((nSensors, nSensors), dtype=np.complex128)
        else:
            baseline = np.zeros((nSensors, nSensors))
        for m in range(nSensors):
            rtf = scalings / scalings[m]
            hs = (rtf.T.conj() @ rtf) * sigma_sr[m] ** 2
            spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter (real-valued)
            baseline[:, m] = rtf / (rtf.T.conj() @ rtf) * spf  # FAS BF + spectral post-filter

    # Compute difference between normalized estimated filters
    # and normalized expected filters
    nFilters = filters.shape[-1]
    diffsPerCoefficient = [None for _ in range(nFilters)]
    currNode = 0
    for k in range(nFilters):
        if 'danse' in filterType:  # DANSE case
            # Determine reference sensor index
            idxRef = np.where(channelToNodeMap == k)[0][0]
            if 'online' in filterType:
                # Keep all iterations (all p.durations)
                currFilt = filters[:, :, k]
            elif 'wola' in filterType:
                currFilt = filters[:, :, :, k]
            else:
                # Only keep last iteration (converged filter coefficients)
                currFilt = filters[:, -1, k]
        else:  # MWF case
            if computeForComparisonWithDANSE:
                if channelToNodeMap[k] == currNode:
                    idxRef = np.where(channelToNodeMap == channelToNodeMap[k])[0][0]
                    if 'online' in filterType:
                        currFilt = filters[:, :, idxRef]
                    elif 'wola' in filterType:
                        currFilt = filters[:, :, :, idxRef]
                    else:
                        currFilt = filters[:, idxRef]
                    currNode += 1
                else:
                    diffsPerCoefficient[k] = np.nan
                    continue
            else:
                idxRef = copy.deepcopy(k)
                currFilt = filters[:, k]
        
        if 'wola' in filterType:
            currBaseline = baseline[:, idxRef, :]
            # Average over nodes and frequency bins, keeping iteration index
            diffsPerCoefficient[k] = np.mean(np.abs(
                currFilt - currBaseline[:, np.newaxis, :]
            ), axis=(0, -1))
        else:
            currBaseline = baseline[:, idxRef]
            if len(currFilt.shape) == 2:
                # Average over nodes, keeping iteration index
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline[:, np.newaxis]
                ), axis=0)
            else:
                # Average over nodes
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline
                ))
    diffsPerCoefficient = np.array(diffsPerCoefficient, dtype=object)
    
    # Average absolute difference over filters
    diffsPerCoefficient_noNan = []
    for d in diffsPerCoefficient:
        if isinstance(d, np.ndarray):
            diffsPerCoefficient_noNan.append(d)
        elif not np.isnan(d):
            diffsPerCoefficient_noNan.append(d)

    if exportDiffPerFilter:
        diffs = np.array(diffsPerCoefficient_noNan).T
    else:
        diffs = np.nanmean(diffsPerCoefficient_noNan, axis=0)

    return diffs


def get_sigma_wola(x, params: WOLAparameters):
    """Compute WOLA-based signal power estimate."""
    # Check for multichannel input
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    
    win = get_window(params.winType, params.nfft)
    nIter = int((x.shape[0] - params.nfft) / params.hop) + 1
    sigmasSquared = np.zeros((nIter, params.nfft, x.shape[1]))
    for i in range(nIter):
        idxStart = i * params.hop
        idxEnd = idxStart + params.nfft
        windowedChunk = x[idxStart:idxEnd, :] * win[:, np.newaxis]
        Chunk = np.fft.fft(windowedChunk, axis=0)
        sigmasSquared[i, :, :] = np.abs(Chunk) ** 2
    
    # Only keep positive frequencies
    sigmasSquared = sigmasSquared[:, :int(params.nfft / 2) + 1, :]

    # Average over iterations and get actual sigma (not squared)
    sigmaWOLA = np.sqrt(np.mean(sigmasSquared, axis=0))
    # Adapt array dimensions
    if params.singleFreqBinIndex is not None:
        sigmaWOLA = sigmaWOLA[params.singleFreqBinIndex, :]
        sigmaWOLA = sigmaWOLA[np.newaxis, :]
    else:
        sigmaWOLA = np.squeeze(sigmaWOLA)
    
    return sigmaWOLA


def get_vad(x, tw, eFactdB, fs, loadIfPossible=False, vadFilesFolder=''):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Signal.
    tw : int
        Window length in samples.
    eFactdB : float
        Energy factor [dB].
    fs : float
        Sampling frequency in Hz.
    loadIfPossible : bool
        If True, try to load VAD from file.
    vadFilesFolder : str
        Folder where VAD files are stored.

    Returns
    -------
    vad : [N x 1] np.ndarray (bool or int [0 or 1])
        VAD.
    """

    # Compute VAD filename
    vadFilename = f'{vadFilesFolder}/vad_{array_id(x)}_eFactdB_{dot_to_p(np.round(eFactdB, 1))}_Nw_{dot_to_p(np.round(tw, 3))}_fs_{dot_to_p(fs)}.npy'
    # Check if VAD can be loaded from file
    if loadIfPossible and os.path.isfile(vadFilename):
        # Load VAD from file
        print(f'Loading VAD from file {vadFilename}')
        vad = np.load(vadFilename)
    else:
        # Compute VAD
        vad, _ = oracleVAD(
            x,
            tw=tw,
            eFactdB=eFactdB,
            fs=fs
        )
        np.save(vadFilename, vad)

    return vad

    
def oracleVAD(x, tw=0.02, eFactdB=20, fs=16000):

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = int(tw * fs)
    else:
        Nw = 1

    # Check for multiple channels
    if len(x.shape) > 1:
        nCh = x.shape[1]
    else:
        nCh = 1
        x = x[:, np.newaxis]
    
    # Compute VAD
    oVAD = np.zeros((n, nCh))
    for ch in range(nCh):
        print(f'Computing VAD for channel {ch+1}/{nCh}')
        # Compute VAD threshold
        eFact = 10 ** (eFactdB / 10)
        thrs = np.amax(x[:, ch] ** 2) / eFact
        for ii in range(n):
            # Extract chunk
            idxBeg = int(np.amax([ii - Nw // 2, 0]))
            idxEnd = int(np.amin([ii + Nw // 2, n]))
            # Compute VAD frame
            oVAD[ii, ch] = compute_VAD(x[idxBeg:idxEnd, ch], thrs)

    # Time vector
    t = np.arange(n) / fs

    return oVAD, t


@njit
def compute_VAD(chunk_x,thrs):
    """
    JIT-ed time-domain VAD computation
    
    (c) Paul Didier - 6-Oct-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """
    # Compute short-term signal energy
    energy = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if energy > thrs:
        VADout = 1
    else:
        VADout = 0
    return VADout




def array_id(
        a: np.ndarray, *,
        include_dtype=False,
        include_shape=False,
        algo = 'xxhash'
    ):
    """
    Computes a unique ID for a numpy array.
    From: https://stackoverflow.com/a/64756069

    Parameters
    ----------
    a : np.ndarray
        The array to compute the ID for.
    include_dtype : bool, optional
        Whether to include the dtype in the ID.
    include_shape : bool, optional
    """
    data = bytes()
    if include_dtype:
        data += str(a.dtype).encode('ascii')
    data += b','
    if include_shape:
        data += str(a.shape).encode('ascii')
    data += b','
    data += a.tobytes()
    if algo == 'sha256':
        import hashlib
        return hashlib.sha256(data).hexdigest().upper()
    elif algo == 'xxhash':
        import xxhash
        return xxhash.xxh3_64(data).hexdigest().upper()
    else:
        assert False, algo


def dot_to_p(x):
    """
    Converts a float to a string with a 'p' instead of a '.'.
    """
    return str(x).replace('.', 'p')

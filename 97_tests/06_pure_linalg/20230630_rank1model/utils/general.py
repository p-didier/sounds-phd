import copy
import resampy
import numpy as np
import soundfile as sf
from numba import njit
import scipy.linalg as la
from utils.online_mwf import *
from utils.online_danse import *


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

    nSensors = cleanSigs.shape[1]
    Ryy = noisySignals.T.conj() @ noisySignals
    Rnn = noiseOnlySigs.T.conj() @ noiseOnlySigs
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
        }
        if 'sim' in type:
            kwargsBatch['nodeUpdatingStrategy'] = 'simultaneous'
        else:
            kwargsBatch['nodeUpdatingStrategy'] = 'sequential'

        # Keyword-arguments for online-mode DANSE function
        kwargsOnline = copy.deepcopy(kwargsBatch)
        kwargsOnline['p'] = wolaParams
        kwargsOnline['ignoreFusionForSSNodes'] = noSSfusion
        kwargsOnline['vad'] = vad
    
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
            yTilde = np.concatenate((
                y[:, channelToNodeMap == k],
                fusedSignals[:, idxNodes != k]
            ), axis=1)
            nTilde = np.concatenate((
                n[:, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, idxNodes != k]
            ), axis=1)

            # Compute covariance matrices
            Ryy = yTilde.T @ yTilde.conj()
            Rnn = nTilde.T @ nTilde.conj()
            
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
        targetSignalSpeechFile,
        nSensors,
        dur,
        scalings,
        fsTarget,
        sigType='speech',
        randomDelays=False,
        maxDelay=0.1
    ):
    # Load target signal
    if sigType == 'speech':
        latentSignal, fs = sf.read(targetSignalSpeechFile)
        if fs != fsTarget:
            # Resample
            latentSignal = resampy.resample(
                latentSignal,
                fs,
                fsTarget
            )
        # Truncate
        latentSignal = latentSignal[:int(dur * fsTarget)]
    elif sigType == 'noise_real':
        # Generate noise signal (real-valued)
        latentSignal = np.random.normal(size=int(dur * fsTarget))
    elif sigType == 'noise_complex':
        # Generate noise signal (complex-valued)
        latentSignal = np.random.normal(size=int(dur * fsTarget)) +\
            1j * np.random.normal(size=int(dur * fsTarget))
    # Normalize power
    latentSignal /= np.sqrt(np.mean(np.abs(latentSignal) ** 2))
    
    # Generate clean signals
    mat = np.tile(latentSignal, (nSensors, 1)).T
    # Random scalings
    cleanSigs = mat @ np.diag(scalings)  # rank-1 matrix (scalings only)
    # Random delays
    if randomDelays:
        for n in range(nSensors):
            delay = np.random.randint(0, int(maxDelay * fsTarget))
            cleanSigs[:, n] = np.roll(cleanSigs[:, n], delay)

    return cleanSigs, latentSignal


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
        'vad': vad
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
        baseline = np.zeros((nSensors, nSensors, nBins))
        for m in range(nSensors):
            rtf = scalings / scalings[m]
            hs = np.sum(rtf ** 2) * sigma_sr[:, m] ** 2
            spf = hs / (hs + sigma_nr[:, m] ** 2)  # spectral post-filter
            baseline[:, m, :] = np.outer(rtf / (rtf.T @ rtf), spf)  # FAS BF + spectral post-filter
    else:
        baseline = np.zeros((nSensors, nSensors))
        for m in range(nSensors):
            rtf = scalings / scalings[m]
            hs = np.sum(rtf ** 2) * sigma_sr[m] ** 2
            spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter
            baseline[:, m] = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

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
    if params.singleFreqBinIndex is not None:
        sigmaWOLA = sigmaWOLA[params.singleFreqBinIndex, :]
        sigmaWOLA = sigmaWOLA[np.newaxis, :]
    else:
        sigmaWOLA = np.squeeze(sigmaWOLA)
    
    return sigmaWOLA


def get_vad(x, tw, eFactdB, Fs):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    -x [N*1 float vector, -] - Time-domain signal.
    -tw [float, s] - VAD window length.
    -eFactdB [float] - Energy factor for threshold, in dB.
    -Fs [int, samples/s] - Sampling frequency.
    
    Returns
    -------
    -oVAD [N*1 binary vector] - Oracle VAD corresponding to <x>.

    (c) Paul Didier - 13-Sept-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = int(tw * Fs)
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
    t = np.arange(n) / Fs

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
# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import resampy
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.linalg as la
from utils.plots import *
from utils.online_mwf import *
from utils.online_danse import *
from dataclasses import dataclass, field
sys.path.append('..')

@dataclass
class ScriptParameters:
    randomDelays: bool = False
    targetSignalSpeechFile: str = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
    nSensors: int = 15
    nNodes: int = 3
    selfNoisePower: float = 1
    durations: np.ndarray = np.logspace(np.log10(1), np.log10(30), 30)
    fs: float = 16e3
    nMC: int = 1
    exportFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/figs/20230807_tests'
    taus: list[float] = field(default_factory=lambda: [2., 4., 8.])
    b: float = 0.1  # factor for determining beta from tau (online processing)
    signalType: str = 'noise_complex'
    toCompute: list[str] = field(default_factory=lambda: [
        # 'mwf_batch',
        'gevdmwf_batch',
        # 'danse_batch',
        # 'gevddanse_batch',
        # 'danse_sim_batch',
        # 'gevddanse_sim_batch',
        # 'mwf_online',
        'gevdmwf_online',
        # 'danse_online',
        # 'gevddanse_online',
        # 'danse_sim_online',
        'gevddanse_sim_online'
        # 'danse_wola',
        # 'gevddanse_wola',
        # 'danse_sim_wola',
        # 'gevddanse_sim_wola'
    ])
    seed: int = 0
    wolaParams: WOLAparameters = WOLAparameters(
        fs=fs,
        B=0,  # frames
        alpha=1,  # if ==1, no fusion vector relaxation
        betaExt=0.0,  # if ==0, no extra fusion vector relaxation
        # betaExt=0.9,  # if ==0, no extra fusion vector relaxation
        # betaExt=np.concatenate((np.linspace(0, 0.8, 10), np.linspace(0.9, 0.99, 10))),  # if ==0, no extra fusion vector relaxation
        startExpAvgAfter=2,  # frames
        startFusionExpAvgAfter=2,  # frames
    )
    showDeltaPerNode: bool = False
    useBatchModeFusionVectorsInOnlineDanse: bool = False
    ignoreFusionForSSNodes: bool = True  # in DANSE, ignore fusion vector for single-sensor nodes
    Mk: list[int] = field(default_factory=lambda: None)  # if None, randomly assign sensors to nodes
    exportFigures: bool = True
    verbose: bool = True


def main(p: ScriptParameters=ScriptParameters()):
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(p.seed)
    rngState = np.random.get_state()

    if p.Mk is None:
        # For DANSE: randomly assign sensors to nodes, ensuring that each node
        # has at least one sensor
        channelToNodeMap = np.zeros(p.nSensors, dtype=int)
        for k in range(p.nNodes):
            channelToNodeMap[k] = k
        for n in range(p.nNodes, p.nSensors):
            channelToNodeMap[n] = np.random.randint(0, p.nNodes)
        # Sort
        channelToNodeMap = np.sort(channelToNodeMap)
    else:
        # Assign sensors to nodes according to Mk
        channelToNodeMap = np.zeros(p.nSensors, dtype=int)
        for k in range(p.nNodes):
            idxStart = int(np.sum(p.Mk[:k]))
            idxEnd = idxStart + p.Mk[k]
            channelToNodeMap[idxStart:idxEnd] = k

    # Set rng state back to original after the random assignment of sensors
    np.random.set_state(rngState)

    if isinstance(p.wolaParams.betaExt, (float, int)):
        p.wolaParams.betaExt = np.array([p.wolaParams.betaExt])

    for betaExtCurr in p.wolaParams.betaExt:
        if p.verbose:
            print(f'>>>>>>>> Running with betaExt = {betaExtCurr}')

        # Set RNG state back to original for each betaExt loop iteration
        np.random.set_state(rngState)

        # Set external beta
        wolaParamsCurr = copy.deepcopy(p.wolaParams)
        wolaParamsCurr.betaExt = betaExtCurr
        # Initialize dictionary where results are stored for plotting
        toDict = []
        for filterType in p.toCompute:
            if 'online' in filterType:
                nIter = int(np.amax(p.durations) *\
                    wolaParamsCurr.fs / wolaParamsCurr.nfft)
                if p.showDeltaPerNode:
                    toDict.append((filterType, np.zeros((p.nMC, nIter, len(p.taus), p.nNodes))))
                else:
                    toDict.append((filterType, np.zeros((p.nMC, nIter, len(p.taus)))))
            else:
                if p.showDeltaPerNode:
                    toDict.append((filterType, np.zeros((p.nMC, len(p.durations), p.nNodes))))
                else:
                    toDict.append((filterType, np.zeros((p.nMC, len(p.durations)))))
        metricsData = dict(toDict)

        for idxMC in range(p.nMC):
            if p.verbose:
                print(f'Running Monte-Carlo iteration {idxMC+1}/{p.nMC}')

            # Get scalings
            scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors)
            # Get clean signals
            nSamplesMax = int(np.amax(p.durations) * wolaParamsCurr.fs)
            cleanSigs, _ = get_clean_signals(
                p.targetSignalSpeechFile,
                p.nSensors,
                np.amax(p.durations),
                scalings,
                wolaParamsCurr.fs,
                sigType=p.signalType,
                randomDelays=p.randomDelays,
                maxDelay=0.1
            )
            sigma_sr = np.sqrt(np.mean(np.abs(cleanSigs) ** 2, axis=0))

            # Generate noise signals
            if np.iscomplex(cleanSigs).any():
                noiseSignals = np.zeros((nSamplesMax, p.nSensors), dtype=np.complex128)
            else:
                noiseSignals = np.zeros((nSamplesMax, p.nSensors))
            sigma_nr = np.zeros(p.nSensors)
            for n in range(p.nSensors):
                # Generate random sequence with unit power
                if np.iscomplex(cleanSigs).any():
                    randSequence = np.random.normal(size=nSamplesMax) +\
                        1j * np.random.normal(size=nSamplesMax)
                    
                else:
                    randSequence = np.random.normal(size=nSamplesMax)
                # Make unit power
                randSequence /= np.sqrt(np.mean(np.abs(randSequence) ** 2))
                # Scale to desired power
                noiseSignals[:, n] = randSequence * np.sqrt(p.selfNoisePower)
                # Check power
                sigma_nr[n] = np.sqrt(np.mean(np.abs(noiseSignals[:, n]) ** 2))
                if np.abs(sigma_nr[n] ** 2 - p.selfNoisePower) > 1e-6:
                    raise ValueError(f'Noise signal power is {sigma_nr[n] ** 2} instead of {p.selfNoisePower}')
                
            # Compute desired filters
            allFilters = get_filters(
                cleanSigs,
                noiseSignals,
                channelToNodeMap,
                gevdRank=1,
                toCompute=p.toCompute,
                wolaParams=wolaParamsCurr,
                taus=p.taus,
                durations=p.durations,
                b=p.b,
                batchFusionInOnline=p.useBatchModeFusionVectorsInOnlineDanse,
                noSSfusion=p.ignoreFusionForSSNodes,
                verbose=p.verbose
            )

            # Compute metrics
            for filterType in p.toCompute:
                currFilters = allFilters[filterType]
                if 'online' in filterType:
                    for idxTau in range(len(p.taus)):
                        currFiltCurrTau = currFilters[idxTau, :, :, :] 
                        currMetrics = get_metrics(
                            p.nSensors,
                            currFiltCurrTau,
                            scalings,
                            sigma_sr,
                            sigma_nr,
                            channelToNodeMap,
                            filterType=filterType,
                            exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                        )
                        if p.showDeltaPerNode:
                            metricsData[filterType][idxMC, :, idxTau, :] = currMetrics
                        else:
                            metricsData[filterType][idxMC, :, idxTau] = currMetrics
                else:
                    for idxDur in range(len(p.durations)):
                        currFiltCurrDur = currFilters[idxDur, :, :]
                        currMetrics = get_metrics(
                            p.nSensors,
                            currFiltCurrDur,
                            scalings,
                            sigma_sr,
                            sigma_nr,
                            channelToNodeMap,
                            filterType=filterType,
                            exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                        )
                        if p.showDeltaPerNode:
                            metricsData[filterType][idxMC, idxDur, :] = currMetrics
                        else:
                            metricsData[filterType][idxMC, idxDur] = currMetrics
        
        if p.exportFigures:
            # Plot results
            fig = plot_final(
                p.durations,
                p.taus,
                metricsData,
                fs=wolaParamsCurr.fs,
                L=wolaParamsCurr.nfft,
                avgAcrossNodesFlag=not p.showDeltaPerNode,
                figTitleSuffix=f'$\\beta_{{\\mathrm{{EXT}}}} = {np.round(betaExtCurr, 4)}$'
            )

            if p.exportFolder is not None:
                if len(p.durations) > 1:
                    fname = f'{p.exportFolder}/diff'
                else:
                    fname = f'{p.exportFolder}/betas'
                fname += f'_betaExt_0p{int(betaExtCurr * 1000)}'
                for t in p.toCompute:
                    fname += f'_{t}'
                if not Path(p.exportFolder).is_dir():
                    Path(p.exportFolder).mkdir(parents=True, exist_ok=True)
                fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
        
            plt.close(fig)

    return metricsData


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
        verbose=True
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
            L=wolaParams.nfft,
            beta=wolaParams.betaMwf,
            verbose=verbose
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
            L=wolaParams.nfft,
            beta=wolaParams.betaMwf,
            verbose=verbose
        )
    elif 'danse' in type:
        kwargs = {
            'x': cleanSigs,
            'n': noiseOnlySigs,
            'channelToNodeMap': channelToNodeMap,
            'filterType': 'gevd' if 'gevd' in type else 'regular',
            'rank': rank,
            'verbose': verbose
        }
        if 'sim' in type:
            kwargs['nodeUpdatingStrategy'] = 'simultaneous'
        else:
            kwargs['nodeUpdatingStrategy'] = 'sequential'

        if 'wola' in type:
            raise NotImplementedError('To be implemented properly [PD~2023.08.02]')
            kwargs['referenceSensorIdx'] = 0
            kwargs['nfft'] = wolaParams.nfft
            kwargs['beta'] = wolaParams.betaDanse
            kwargs['hop'] = wolaParams.hop
            kwargs['windowType'] = wolaParams.winType
            kwargs['fs'] = wolaParams.fs
            kwargs['upExtFiltEvery'] = wolaParams.upExtTargetFiltEvery
            w = run_wola_danse(**kwargs)
        elif 'online' in type:
            if batchFusionInOnline:
                wBatchNetWide = run_danse(**kwargs)
                kwargs['batchModeNetWideFilters'] = wBatchNetWide
            kwargs['referenceSensorIdx'] = 0
            kwargs['L'] = wolaParams.nfft
            kwargs['beta'] = wolaParams.betaDanse
            kwargs['B'] = wolaParams.B
            kwargs['alpha'] = wolaParams.alpha
            kwargs['betaExt'] = wolaParams.betaExt
            kwargs['startExpAvgAfter'] = wolaParams.startExpAvgAfter
            kwargs['startFusionExpAvgAfter'] = wolaParams.startFusionExpAvgAfter
            kwargs['ignoreFusionForSSNodes'] = noSSfusion
            w = run_online_danse(**kwargs)
        else:
            w = run_danse(**kwargs)
    return w


def run_danse(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        refSensorIdx=0,
        verbose=True
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
        wCurr[0, :] = 1
        w.append(wCurr)
    idxNodes = np.arange(nNodes)
    idxUpdatingNode = 0
    wNet = np.zeros((x.shape[1], maxIter, nNodes), dtype=myDtype)

    for k in range(nNodes):
        # Determine reference sensor index
        idxRef = np.where(channelToNodeMap == k)[0][0]
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
                    e[refSensorIdx] = 1  # selection vector
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
                    e[refSensorIdx] = 1
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
        verbose=True
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

        if 'online' in filterType:
            # Only need to compute for longest duration
            # + computing for several tau's
            nSamples = int(np.amax(durations) * wolaParams.fs)
            noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]
            kwargs['noisySignals'] = noisySignals
            kwargs['cleanSigs'] = cleanSigs[:nSamples, :]
            kwargs['noiseOnlySigs'] = noiseSignals[:nSamples, :]
            currFiltsAll = []
            # ^^^ shape: (nTaus, nSensors, nIters, nNodes)
            for idxTau in range(len(taus)):
                # Set beta based on tau
                kwargs['wolaParams'].betaMwf = np.exp(
                    np.log(b) / (taus[idxTau] *\
                        kwargs['wolaParams'].fs / kwargs['wolaParams'].nfft)
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
    
    # Compute FAS + SPF
    fasAndSPF = np.zeros((nSensors, nSensors))
    for m in range(nSensors):
        rtf = scalings / scalings[m]
        hs = np.sum(rtf ** 2) * sigma_sr[m] ** 2
        spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter
        fasAndSPF[:, m] = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

    # Compute difference between normalized estimated filters
    # and normalized expected filters
    nFilters = filters.shape[-1]
    diffsPerCoefficient = [None for _ in range(nFilters)]
    currNode = 0
    for m in range(nFilters):
        if 'danse' in filterType:  # DANSE case
            # Determine reference sensor index
            idxRef = np.where(channelToNodeMap == m)[0][0]
            if 'online' in filterType:
                # Keep all iterations (all p.durations)
                currFilt = filters[:, :, m]
            else:
                # Only keep last iteration (converged filter coefficients)
                currFilt = filters[:, -1, m]
        else:  # MWF case
            if computeForComparisonWithDANSE:
                if channelToNodeMap[m] == currNode:
                    idxRef = np.where(channelToNodeMap == channelToNodeMap[m])[0][0]
                    if 'online' in filterType:
                        currFilt = filters[:, :, idxRef]
                    else:
                        currFilt = filters[:, idxRef]
                    currNode += 1
                else:
                    diffsPerCoefficient[m] = np.nan
                    continue
            else:
                idxRef = copy.deepcopy(m)
                currFilt = filters[:, m]
        
        currFas = fasAndSPF[:, idxRef]
        if len(currFilt.shape) == 2:
            diffsPerCoefficient[m] = np.mean(np.abs(
                currFilt - currFas[:, np.newaxis]
            ), axis=0)
        else:
            diffsPerCoefficient[m] = np.mean(np.abs(
                currFilt - currFas
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


if __name__ == '__main__':
    sys.exit(main())
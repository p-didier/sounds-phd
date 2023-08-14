# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
sys.path.append('..')

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

@dataclass
class ScriptParameters:
    signalType: str = 'noise_real'  # 'speech', 'noise_real', 'noise_complex'
    # signalType: str = 'noise_complex'  # 'speech', 'noise_real', 'noise_complex'
    targetSignalSpeechFile: str = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
    nSensors: int = 5
    nNodes: int = 3
    Mk: list[int] = field(default_factory=lambda: None)  # if None, randomly assign sensors to nodes
    # Mk: list[int] = field(default_factory=lambda: [1, 4])  # if None, randomly assign sensors to nodes
    selfNoisePower: float = 1
    durations: np.ndarray = np.linspace(1, 20, 30)
    fs: float = 8e3
    nMC: int = 10
    exportFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/figs/for20230823marcUpdate'
    taus: list[float] = field(default_factory=lambda: [2.])
    # taus: list[float] = field(default_factory=lambda: [2., 4., 8.])
    b: float = 0.1  # factor for determining beta from tau (online processing)
    toCompute: list[str] = field(default_factory=lambda: [
        'mwf_batch',
        'gevdmwf_batch',
        # 'danse_batch',
        # 'gevddanse_batch',
        'danse_sim_batch',
        'gevddanse_sim_batch',
        # 'mwf_online',     # --------- vvvv ONLINE vvvv
        # 'gevdmwf_online',
        # 'danse_online',
        # 'gevddanse_online',
        # 'danse_sim_online',
        # 'gevddanse_sim_online'
        # 'mwf_wola',       # --------- vvvv WOLA vvvv
        # 'gevdmwf_wola',
        # 'danse_wola',
        # 'gevddanse_wola',
        # 'danse_sim_wola',
        # 'gevddanse_sim_wola'
    ])
    seed: int = 0
    wolaParams: WOLAparameters = WOLAparameters(
        nfft=1024,
        hop=512,
        fs=fs,
        # B=10,
        # alpha=0.5,
        # winType='rect',
        # betaExt=.7,  # if ==0, no extra fusion vector relaxation
        startExpAvgAfter=2,  # frames
        startFusionExpAvgAfter=2,  # frames
        # singleFreqBinIndex=None,  # if not None, only consider the freq. bin at this index in WOLA-DANSE
        singleFreqBinIndex=99,  # if not None, only consider the freq. bin at this index in WOLA-DANSE
    )
    # Booleans vvvv
    randomDelays: bool = False
    showDeltaPerNode: bool = False
    useBatchModeFusionVectorsInOnlineDanse: bool = False
    ignoreFusionForSSNodes: bool = True  # in DANSE, ignore fusion vector for single-sensor nodes
    exportFigures: bool = True
    verbose: bool = True

    def __post_init__(self):
        if any(['wola' in t for t in self.toCompute]) and\
            'complex' in self.signalType:
            raise ValueError('WOLA not implemented for complex-valued signals')


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
            if 'online' in filterType or 'wola' in filterType:
                # Compute number of iterations
                if 'online' in filterType:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.nfft) # divide by frame size
                elif 'wola' in filterType:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.hop) - 1  # divide by hop size
                
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
            sigma_sr_wola = get_sigma_wola(cleanSigs, wolaParamsCurr)

            # Generate noise signals
            sigma_nr = np.zeros(p.nSensors)
            if wolaParamsCurr.singleFreqBinIndex is not None:
                sigma_nr_wola = np.zeros((1, p.nSensors))
            else:
                sigma_nr_wola = np.zeros((wolaParamsCurr.nPosFreqs, p.nSensors))
            if np.iscomplex(cleanSigs).any():
                noiseSignals = np.zeros((nSamplesMax, p.nSensors), dtype=np.complex128)
            else:
                noiseSignals = np.zeros((nSamplesMax, p.nSensors))
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
                sigma_nr_wola[:, n] = get_sigma_wola(
                    noiseSignals[:, n],
                    wolaParamsCurr
                )

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
                if 'online' in filterType or 'wola' in filterType:
                    for idxTau in range(len(p.taus)):
                        if 'online' in filterType:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :],
                                scalings,
                                sigma_sr,
                                sigma_nr,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                            )
                        elif 'wola' in filterType:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :, :],
                                scalings,
                                sigma_sr_wola,
                                sigma_nr_wola,
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
            figTitleSuffix = f'$\\beta_{{\\mathrm{{EXT}}}} = {np.round(betaExtCurr, 4)}$'
            if wolaParamsCurr.singleFreqBinIndex is not None and\
                any('wola' in t for t in p.toCompute):
                figTitleSuffix += f", WOLA's {wolaParamsCurr.singleFreqBinIndex + 1}-th freq. bin"
            fig = plot_final(
                p.durations,
                p.taus,
                metricsData,
                fs=wolaParamsCurr.fs,
                L=wolaParamsCurr.nfft,
                R=wolaParamsCurr.hop,
                avgAcrossNodesFlag=not p.showDeltaPerNode,
                figTitleSuffix=figTitleSuffix,
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
        
            # Wait for button press to close the figures
            plt.waitforbuttonpress()
            plt.close('all')

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
            p=wolaParams,
            verbose=verbose
        )
    elif type == 'mwf_wola':
        w = run_wola_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='regular',
            p=wolaParams,
            verbose=verbose,
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
            verbose=verbose
        )
    elif type == 'gevdmwf_wola':
        w = run_wola_mwf(
            x=cleanSigs,
            n=noiseOnlySigs,
            filterType='gevd',
            rank=rank,
            p=wolaParams,
            verbose=verbose,
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

            # fig, axes = plt.subplots(3,1, sharex=True, sharey=True)
            # fig.set_size_inches(8.5, 4.5)
            # axes[0].loglog(np.squeeze(np.real(currFilt)).T)
            # for ii in range(len(currBaseline)):
            #     axes[0].hlines(
            #         np.squeeze(np.real(currBaseline[ii])),
            #         0, 1000, linestyle='--', color=f'C{ii}'
            #     )
            # axes[0].grid(True)
            # axes[0].set_title('Real part')
            # axes[1].loglog(np.squeeze(np.imag(currFilt)).T)
            # for ii in range(len(currBaseline)):
            #     axes[1].hlines(
            #         np.squeeze(np.real(currBaseline[ii])),
            #         0, 1000, linestyle='--', color=f'C{ii}'
            #     )
            # axes[1].grid(True)
            # axes[1].set_title('Imaginary part')
            # axes[2].loglog(np.squeeze(np.abs(currFilt)).T)
            # for ii in range(len(currBaseline)):
            #     axes[2].hlines(
            #         np.squeeze(np.abs(currBaseline[ii])),
            #         0, 1000, linestyle='--', color=f'C{ii}'
            #     )
            # axes[2].grid(True)
            # axes[2].set_title('Magnitude')

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


if __name__ == '__main__':
    sys.exit(main())
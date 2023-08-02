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
sys.path.append('..')

RANDOM_DELAYS = False
TARGET_SIGNAL_SPEECHFILE = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
N_SENSORS = 5
N_NODES = 5
SELFNOISE_POWER = 1
DURATIONS = np.logspace(np.log10(1), np.log10(30), 20)
# DURATIONS = np.logspace(np.log10(0.5), np.log10(3), 20)
# DURATIONS = [30]
FS = 16e3
N_MC = 5
EXPORT_FOLDER = '97_tests/06_pure_linalg/20230630_rank1model/figs/20230802_tests'
# EXPORT_FOLDER = None
# TAUS = [2., 4., 8.]
# TAUS = list(np.linspace(1, 10, 10))
TAUS = [2.]
B = 0.1  # factor for determining beta from tau (online processing)

# Type of signal
# SIGNAL_TYPE = 'speech'
# SIGNAL_TYPE = 'noise_real'
SIGNAL_TYPE = 'noise_complex'

TO_COMPUTE = [
    # 'mwf',
    'gevdmwf',
    # 'danse',
    # 'gevddanse',
    # 'danse_sim',
    # 'gevddanse_sim',
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
]

SEED = 0

WOLA_PARAMS = WOLAparameters(
    fs=FS,
    betaDanse=0.75,
    # nfft=4096,
    B=0,  # frames
    alpha=1,  # if ==1, no fusion vector relaxation
    betaExt=0.98  # if ==0, no extra fusion vector relaxation
)

# Debug parameters
SHOW_DELTA_PER_NODE = False
USE_BATCH_MODE_FUSION_VECTORS_IN_ONLINE_DANSE = False

def main(
        M=N_SENSORS,
        K=N_NODES,
        toCompute=TO_COMPUTE,
        durations=DURATIONS,
        fs=FS,
        nMC=N_MC,
        selfNoisePower=SELFNOISE_POWER,
        seed=SEED,
        wolaParams=WOLA_PARAMS,
        taus=TAUS,
        b=B
    ):
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(seed)
    rngState = np.random.get_state()

    # For DANSE: randomly assign sensors to nodes, ensuring that each node
    # has at least one sensor
    channelToNodeMap = np.zeros(M, dtype=int)
    for k in range(K):
        channelToNodeMap[k] = k
    for n in range(K, M):
        channelToNodeMap[n] = np.random.randint(0, K)
    # Sort
    channelToNodeMap = np.sort(channelToNodeMap)

    # Set rng state back to original after the random assignment of sensors
    np.random.set_state(rngState)

    # Get scalings
    scalings = np.random.uniform(low=0.5, high=1, size=M)
    # Get clean signals
    nSamplesMax = int(np.amax(durations) * fs)
    cleanSigs, _ = get_clean_signals(
        M,
        np.amax(durations),
        scalings,
        fs,
        sigType=SIGNAL_TYPE,
        randomDelays=RANDOM_DELAYS,
        maxDelay=0.1
    )
    sigma_sr = np.sqrt(np.mean(np.abs(cleanSigs) ** 2, axis=0))

    # Initialize arrays for storing results
    shape = (nMC, len(durations), len(taus))
    if SHOW_DELTA_PER_NODE:
        shape += (K,)  # add dimension for nodes
    toPlot = dict([(key, np.zeros(shape)) for key in TO_COMPUTE])

    for idxMC in range(nMC):
        print(f'Running Monte-Carlo iteration {idxMC+1}/{nMC}')

        # Generate noise signals
        if np.iscomplex(cleanSigs).any():
            noiseSignals = np.zeros((nSamplesMax, M), dtype=np.complex128)
        else:
            noiseSignals = np.zeros((nSamplesMax, M))
        sigma_nr = np.zeros(M)
        for n in range(M):
            # Generate random sequence with unit power
            if np.iscomplex(cleanSigs).any():
                randSequence = np.random.normal(size=nSamplesMax) +\
                    1j * np.random.normal(size=nSamplesMax)
                
            else:
                randSequence = np.random.normal(size=nSamplesMax)
            # Make unit power
            randSequence /= np.sqrt(np.mean(np.abs(randSequence) ** 2))
            # Scale to desired power
            noiseSignals[:, n] = randSequence * np.sqrt(selfNoisePower)
            # Check power
            sigma_nr[n] = np.sqrt(np.mean(np.abs(noiseSignals[:, n]) ** 2))
            if np.abs(sigma_nr[n] ** 2 - selfNoisePower) > 1e-6:
                raise ValueError(f'Noise signal power is {sigma_nr[n] ** 2} instead of {selfNoisePower}')
            
        # Loop over durations
        for idxDur in range(len(durations)):
            # Generate noisy signals
            nSamples = int(durations[idxDur] * fs)
            noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]

            # Compute desired filters
            filtersAllTaus = get_filters(
                noisySignals,
                cleanSigs[:nSamples, :],
                noiseSignals[:nSamples, :],
                channelToNodeMap,
                gevdRank=1,
                toCompute=toCompute,
                wolaParams=wolaParams,
                taus=taus,
                b=b
            )

            # Compute metrics
            for idxFilter in range(len(toCompute)):

                filterType = toCompute[idxFilter]
                currFilt = filtersAllTaus[filterType]

                for idxTau in range(currFilt.shape[0]):

                    if len(currFilt.shape) == 4:
                        currFiltCurrTau = currFilt[idxTau, :, :, :]
                    else:
                        currFiltCurrTau = currFilt[idxTau, :, :]

                    metrics = get_metrics(
                        M,
                        currFiltCurrTau,
                        scalings,
                        sigma_sr,
                        sigma_nr,
                        channelToNodeMap,
                        filterType=filterType,
                        exportDiffPerFilter=SHOW_DELTA_PER_NODE  # export all differences individually
                    )

                    # Store metrics
                    if SHOW_DELTA_PER_NODE:
                        toPlot[filterType][idxMC, idxDur, idxTau, :] = metrics
                    else:
                        toPlot[filterType][idxMC, idxDur, idxTau] = metrics
    
    # Plot results
    fig = plot_final(durations, taus, toPlot, b=b, fs=fs, L=wolaParams.nfft)

    if EXPORT_FOLDER is not None:
        if len(durations) > 1:
            fname = f'{EXPORT_FOLDER}/diff'
        else:
            fname = f'{EXPORT_FOLDER}/betas'
        for t in toCompute:
            fname += f'_{t}'
        if not Path(EXPORT_FOLDER).is_dir():
            Path(EXPORT_FOLDER).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')

    stop = 1


def compute_filter(
        noisySignals: np.ndarray,
        cleanSigs: np.ndarray,
        noiseOnlySigs: np.ndarray,
        type='mwf',
        rank=1,
        channelToNodeMap=None,  # only used for DANSE
        wolaParams: WOLAparameters=WOLAparameters()
    ):
    """
    [1] Santiago Ruiz, Toon van Waterschoot and Marc Moonen, "Distributed
    combined acoustic echo cancellation and noise reduction in wireless
    acoustic sensor and actuator networks" - 2022
    """

    nSensors = cleanSigs.shape[1]
    Ryy = noisySignals.T.conj() @ noisySignals
    Rnn = noiseOnlySigs.T.conj() @ noiseOnlySigs
    if type == 'mwf':
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
            beta=wolaParams.betaMwf
        )
    elif type == 'gevdmwf':
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
            beta=wolaParams.betaMwf
        )
    elif 'danse' in type:
        kwargs = {
            'x': cleanSigs,
            'n': noiseOnlySigs,
            'channelToNodeMap': channelToNodeMap,
            'filterType': 'gevd' if 'gevd' in type else 'regular',
            'rank': rank
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
            if USE_BATCH_MODE_FUSION_VECTORS_IN_ONLINE_DANSE:
                wBatchNetWide = run_danse(**kwargs)
                kwargs['batchModeNetWideFilters'] = wBatchNetWide
            kwargs['referenceSensorIdx'] = 0
            kwargs['L'] = wolaParams.nfft
            kwargs['beta'] = wolaParams.betaDanse
            kwargs['B'] = wolaParams.B
            kwargs['alpha'] = wolaParams.alpha
            kwargs['betaExt'] = wolaParams.betaExt
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
        refSensorIdx=0
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
        M,
        dur,
        scalings,
        fsTarget,
        sigType='speech',
        randomDelays=False,
        maxDelay=0.1
    ):
    # Load target signal
    if sigType == 'speech':
        latentSignal, fs = sf.read(TARGET_SIGNAL_SPEECHFILE)
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
    mat = np.tile(latentSignal, (M, 1)).T
    # Random scalings
    cleanSigs = mat @ np.diag(scalings)  # rank-1 matrix (scalings only)
    # Random delays
    if randomDelays:
        for n in range(M):
            delay = np.random.randint(0, int(maxDelay * fsTarget))
            cleanSigs[:, n] = np.roll(cleanSigs[:, n], delay)

    return cleanSigs, latentSignal


def get_filters(
        noisySignals,
        cleanSigs,
        noiseSignals,
        channelToNodeMap,
        gevdRank=1,
        toCompute=[''],
        wolaParams: WOLAparameters=WOLAparameters(),
        taus=[2.],
        b=0.1
    ):
    """Compute filters."""
    kwargs = {
        'noisySignals': noisySignals,
        'cleanSigs': cleanSigs,
        'noiseOnlySigs': noiseSignals,
        'channelToNodeMap': channelToNodeMap,
        'rank': gevdRank,
        'wolaParams': wolaParams
    }
    filters = {}
    for filterType in toCompute:
        # Differentiate tau-dependent filters from other filters
        if 'danse' in filterType or 'online' in filterType:
            tausCurr = copy.deepcopy(taus)
        else:
            tausCurr = [0.]  # dummy value (no need to compute for several tau's)
        
        currFiltersAllTaus = []
        for idxTau in range(len(tausCurr)):
            # Set beta based on tau
            kwargs['wolaParams'].betaDanse = np.exp(
                np.log(b) / (tausCurr[idxTau] *\
                    kwargs['wolaParams'].fs / kwargs['wolaParams'].nfft)
            )
            # kwargs['wolaParams'].betaMwf = np.exp(
            #     np.log(b) / (tausCurr[idxTau] *\
            #         kwargs['wolaParams'].fs / kwargs['wolaParams'].nfft - 1)
            # )
            kwargs['wolaParams'].betaMwf = kwargs['wolaParams'].betaDanse

            currFiltersAllTaus.append(
                compute_filter(type=filterType, **kwargs)
            )
        
        filters[filterType] = np.array(currFiltersAllTaus)

    return filters


def get_metrics(
        M,
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
    fasAndSPF = np.zeros((M, M))
    for m in range(M):
        rtf = scalings / scalings[m]
        hs = np.sum(rtf ** 2) * sigma_sr[m] ** 2
        spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter
        fasAndSPF[:, m] = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

    # Compute difference between normalized estimated filters
    # and normalized expected filters
    nFilters = filters.shape[-1]
    diffsPerCoefficient = np.zeros(nFilters)
    currNode = 0
    for m in range(nFilters):
        if 'danse' in filterType:  # DANSE case
            # Determine reference sensor index
            if 'danse' in filterType:
                idxRef = np.where(channelToNodeMap == m)[0][0]
            else:
                idxRef = copy.deepcopy(m)
            currFilt = filters[:, -1, m]
        else:  # MWF case
            if computeForComparisonWithDANSE:
                if channelToNodeMap[m] == currNode:
                    idxRef = np.where(channelToNodeMap == channelToNodeMap[m])[0][0]
                    if 'online' in filterType:
                        currFilt = filters[:, -1, idxRef]
                    else:
                        currFilt = filters[:, idxRef]
                    currNode += 1
                else:
                    diffsPerCoefficient[m] = np.nan
                    continue
            else:
                idxRef = copy.deepcopy(m)
                currFilt = filters[:, m]
        
        diffsPerCoefficient[m] = np.mean(np.abs(
            currFilt - fasAndSPF[:, idxRef]
        ))
    
    if exportDiffPerFilter:
        # All differences individually
        diffs = np.array([d for d in diffsPerCoefficient if not np.isnan(d)])
    else:
        # Average absolute difference over filters
        diffs = np.nanmean(diffsPerCoefficient)

    return diffs


if __name__ == '__main__':
    sys.exit(main())
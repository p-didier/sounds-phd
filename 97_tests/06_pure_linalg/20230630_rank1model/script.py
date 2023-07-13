# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import resampy
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.linalg as la
from scipy.signal import stft
import matplotlib.pyplot as plt
from utils.online_danse import *
sys.path.append('..')

RANDOM_DELAYS = False
TARGET_SIGNAL_SPEECHFILE = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
N_SENSORS = 5
N_NODES = 3
SELFNOISE_POWER = 1
DURATIONS = np.logspace(np.log10(1), np.log10(30), 20)
# DURATIONS = [30]
FS = 16e3
N_MC = 1
EXPORT_FOLDER = '97_tests/06_pure_linalg/20230630_rank1model/figs/forPhDSU_20230707'
EXPORT_FOLDER = None

# Type of signal
# SIGNAL_TYPE = 'speech'
# SIGNAL_TYPE = 'noise_real'
SIGNAL_TYPE = 'noise_complex'

TO_COMPUTE = [
    'mwf',
    'gevdmwf',
    # 'danse',
    # 'gevddanse',
    # 'danse_sim',
    # 'gevddanse_sim',
    # 'danse_online',
    # 'gevddanse_online',
    # 'danse_sim_online',
    # 'gevddanse_sim_online'
]

SEED = 0

def main(
        M=N_SENSORS,
        K=N_NODES,
        durations=DURATIONS,
        fs=FS,
        nMC=N_MC,
        selfNoisePower=SELFNOISE_POWER,
        seed=SEED
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

    # scalings = np.random.uniform(low=50, high=100, size=M)
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
    toPlot = dict([
        (key, np.zeros((nMC, len(durations)))) for key in TO_COMPUTE
    ])
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
        for ii in range(len(durations)):
            # Generate noisy signals
            nSamples = int(durations[ii] * fs)
            noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]

            # Compute desired filters
            filters = get_filters(
                noisySignals,
                cleanSigs[:nSamples, :],
                noiseSignals[:nSamples, :],
                channelToNodeMap,
                # gevdRank=1,
                gevdRank=int(cleanSigs.shape[1]),
            )

            if 0:
                fig = plot_individual_run(
                    filters,
                    scalings,
                    sigma_sr,
                    sigma_nr,
                    nSamples
                )
                fname = f'{EXPORT_FOLDER}/indiv/diff_1stSensor_{nSamples}samples'
                if not Path( f'{EXPORT_FOLDER}/indiv/').is_dir():
                    Path(f'{EXPORT_FOLDER}/indiv/').mkdir(parents=True, exist_ok=True)
                fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')

            # Plot DANSE evolution
            if 0:
                plot_danse_evol(
                    K,
                    channelToNodeMap,
                    durations[ii],
                    filters[TO_COMPUTE[0]],
                    scalings,
                    sigma_sr,
                    sigma_nr,
                    savefigs=True,
                    figLabelRef=TO_COMPUTE[0]
                )
                stop = 1
            
            metrics = get_metrics(
                M,
                filters,
                scalings,
                sigma_sr,
                sigma_nr,
                channelToNodeMap
            )

            # Store metrics
            for key in metrics.keys():
                toPlot[key][idxMC, ii] = metrics[key]

    
    # Plot results
    fig = plot_final(durations, toPlot)

    if EXPORT_FOLDER is not None:
        fname = f'{EXPORT_FOLDER}/diff'
        for t in TO_COMPUTE:
            fname += f'_{t}'
        if not Path(EXPORT_FOLDER).is_dir():
            Path(EXPORT_FOLDER).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')

    stop = 1


def plot_individual_run(filters: dict, scalings, sigma_sr, sigma_nr, nSamples):
    """Plot a visual comparison of the estimate filters with the FAS-SPF,
    for a single MC run, using the absolulte values."""
    
    # Compute FAS + SPF
    rtf = scalings / scalings[0]
    hs = np.sum(rtf ** 2) * sigma_sr[0] ** 2
    spf = hs / (hs + sigma_nr[0] ** 2)  # spectral post-filter
    fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

    # Plot comparison in absolute value, for each filter and the first sensor
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 2.5)
    for key in filters.keys():
        axs.plot(np.abs(filters[key][:, 0]), '.-', label=key)
    axs.plot(np.abs(fasAndSPF), '.-',label='FAS + SPF')
    axs.legend()
    axs.grid(which='both')
    axs.set_title(f'{nSamples} samples signals')
    axs.set_xlabel('Filter coefficients (absolute value)')
    fig.tight_layout()
    plt.show(block=False)

    return fig



def plot_final(durations, toPlot: dict):

    nMC = toPlot[list(toPlot.keys())[0]].shape[0]
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    for ii, key in enumerate(toPlot.keys()):
        baseColor = f'C{ii}'
        lineStyle = np.random.choice(['-', '--', '-.', ':'])
        # Add a patch of color to show the range of values across MC runs
        axes.fill_between(
            durations,
            np.amin(toPlot[key], axis=0),
            np.amax(toPlot[key], axis=0),
            color=f'{baseColor}',
            alpha=0.2
        )
        # axes.loglog(durations, toPlot[key].T, f'{baseColor}.{lineStyle}', alpha=0.5)
        axes.loglog(durations, np.mean(toPlot[key], axis=0), f'{baseColor}.{lineStyle}', label=key)
    plt.grid(which='both')
    axes.legend(loc='lower left')
    plt.xlabel('Signal duration (s)')
    plt.ylabel('Abs. diff. $\\Delta$ bw. filter and FAS + SPF')
    axes.set_title(f'{nMC} MC runs')
    fig.tight_layout()
    plt.show(block=False)

    return fig


def plot_danse_evol(
        K,
        channelToNodeMap,
        dur,
        filterDANSE,
        scalings,
        sigma_sr,
        sigma_nr,
        savefigs=False,
        figLabelRef=''
    ):
    """ Plot DANSE evolution. """
    nSensors = filterDANSE.shape[0]
    for k in range(K):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8.5, 2.5)
        # Determine reference sensor index
        nodeChannels = np.where(channelToNodeMap == k)[0]
        idxRef = nodeChannels[0]  # first sensor of node k
        # Compute FAS + SPF
        rtf = scalings / scalings[idxRef]
        hs = np.sum(rtf ** 2) * sigma_sr[idxRef] ** 2
        spf = hs / (hs + sigma_nr[idxRef] ** 2)  # spectral post-filter
        fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter
        # Plot DANSE evolution
        for m in range(nSensors):
            lab = f'$[\\mathbf{{w}}_k^i]_{m+1}$'
            if m in nodeChannels:
                lab += f' (local)'
            if m == idxRef:
                lab += ' (ref.)'
            ax.plot(
                np.abs(filterDANSE[m, :, k].T),
                f'C{m}.-',
                label=lab
            )
            if m == 0:
                ax.hlines(
                    np.abs(fasAndSPF[m]),
                    0,
                    filterDANSE.shape[1] - 1,
                    color=f'C{m}',
                    linestyle='--',
                    label=f'$[\\mathbf{{w}}_{{\\mathrm{{MF}},k}}]_{m+1}\\cdot\\mathrm{{SPS}}_{m+1}$'
                )
            else:
                ax.hlines(
                    np.abs(fasAndSPF[m]),
                    0,
                    filterDANSE.shape[1] - 1,
                    color=f'C{m}',
                    linestyle='--'
                )
        ax.set_title(f'Node $k=${k + 1}, channels {nodeChannels + 1}')
        ax.legend(loc='upper right')
        ax.grid(which='both')
        plt.xlabel('Iteration index $i$')
        # fig.tight_layout()
        if savefigs:
            fig.savefig(f'{EXPORT_FOLDER}/danse_evol_n{k+1}_dur{int(dur * 1e3)}ms_{figLabelRef}.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)


def plot_filter(filter, scalings, sigma_sr, sigma_nr):
    fig, axs = plt.subplots(1, 1)
    for n in range(filter.shape[1]):
        # axs.plot(filter[:, n] * np.sum(scalings ** 2), '.-', label=f'MWF weights sensor {n}')
        axs.plot(filter[:, n], f'C{n}.-', label=f'MWF weights sensor {n}')
        # axs.plot(filter[:, n] / np.amax(np.abs(filter[:, n])), '.-', label=f'MWF weights sensor {n}')
        h = scalings / scalings[n]
        hs = np.sum(h ** 2) * sigma_sr[n] ** 2
        spectralPostFilter = hs / (hs + sigma_nr[n] ** 2)
        if n == 0:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--', label='Matched BF and spectral post-filter')
        else:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--')
    # axs.plot(scalings / np.amax(np.abs(scalings)), '.--', label='Signal scalings')
    # axs.hlines(1, 0, filter.shape[0] - 1, color='k', linestyle='--', linewidth=0.5)
    fig.legend()
    plt.show(block=False)


def plot_results(cleanSigs, noisySignals, filteredSignals, filteredSignal_RTFs):
    # Plot
    nSensors = cleanSigs.shape[1]
    nRows = int(np.floor(np.sqrt(nSensors)))
    nCols = int(np.ceil(nSensors / nRows))
    fig, axs = plt.subplots(nRows, nCols, sharex=True, sharey=True)
    delta = 2 * np.amax(np.abs(noisySignals))
    for n in range(nSensors):
        if nRows == 1:
            currAx = axs[n % nCols]
        else:
            currAx = axs[int(np.floor(n / nCols)), n % nCols]
        # overlay 1
        currAx.plot(noisySignals[:, n], 'k', label=f'Noisy signal')
        currAx.plot(filteredSignals[:, n], 'r', label=f'MWF-filtering')
        currAx.plot(cleanSigs[:, n], 'y', label=f'Clean signal')
        # overlay 2
        currAx.plot(noisySignals[:, n] + delta, 'k', label=f'Noisy signal')
        currAx.plot(filteredSignal_RTFs[:, n] + delta, 'r', label=f'RTF-filtering')
        currAx.plot(cleanSigs[:, n] + delta, 'y', label=f'Clean signal')
        if n == 0:
            currAx.legend()
        currAx.set_title(f'Sensor {n}')
    plt.show(block=False)


def compute_filter(
        noisySignals: np.ndarray,
        cleanSigs: np.ndarray,
        noiseOnlySigs: np.ndarray,
        type='mwf',
        rank=1,
        channelToNodeMap=None  # only used for DANSE
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
            # ryd = noisySignals.T.conj() @ cleanSigs[:, n]
            # Selection vector
            e = np.zeros(nSensors)
            e[n] = 1
            ryd = (Ryy - Rnn) @ e
            w[:, n] = RyyInv @ ryd
    elif type == 'gevdmwf':
        sigma, Xmat = la.eigh(Ryy, Rnn)
        idx = np.flip(np.argsort(sigma))
        sigma = sigma[idx]
        Xmat = Xmat[:, idx]
        Qmat = np.linalg.inv(Xmat.T.conj())
        Dmat = np.zeros((nSensors, nSensors))
        Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
        w = Xmat @ Dmat @ Qmat.T.conj()   # see eq. (24) in [1]
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

        if 'online' in type:
            kwargs['nfft'] = 1024
            kwargs['hop'] = 512
            kwargs['referenceSensorIdx'] = 0
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
            d = x[:, np.where(channelToNodeMap == k)[0][0]]
            Ryd = yTilde.T @ d.conj()
            
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False

            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    w[k][:, iter + 1] = np.linalg.inv(Ryy) @ Ryd
                elif filterType == 'gevd':
                    sigma, Xmat = la.eigh(Ryy, Rnn)
                    idx = np.flip(np.argsort(sigma))
                    sigma = sigma[idx]
                    Xmat = Xmat[:, idx]
                    Qmat = np.linalg.inv(Xmat.T.conj())
                    Dmat = np.zeros((Ryy.shape[0], Ryy.shape[0]))
                    Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
                    e = np.zeros(Ryy.shape[0])
                    e[:rank] = 1
                    w[k][:, iter + 1] = Xmat @ Dmat @ Qmat.T.conj() @ e
            else:
                w[k][: , iter + 1] = w[k][: , iter]  # keep old filter
            
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
                    wNet[m, iter + 1, k] = w[k][c, iter + 1]  # use local filter coefficient
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


def run_online_danse(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        nfft=1024,
        hop=512,
        windowType='sqrt-hann',
        referenceSensorIdx=0,
        fs=16000,
    ):

    # Check that signals are real-valued
    if np.iscomplex(x).any() or np.iscomplex(n).any():
        raise ValueError('Online DANSE only implemented for real-valued signals')

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1

    # Get window
    win = get_window(windowType, nfft)

    # Convert to STFT domain using SciPy
    kwargs = {
        'fs': fs,
        'window': win,
        'nperseg': nfft,
        'nfft': nfft,
        'noverlap': nfft - hop,
        'return_onesided': True,
        'axis': 0
    }
    x_stft = stft(x, **kwargs)[2]
    y_stft = stft(y, **kwargs)[2]
    n_stft = stft(n, **kwargs)[2]
    # Reshape
    x_stft = x_stft.reshape((x_stft.shape[0], -1, nNodes))
    y_stft = y_stft.reshape((y_stft.shape[0], -1, nNodes))
    n_stft = n_stft.reshape((n_stft.shape[0], -1, nNodes))
    nIter = x_stft.shape[1]

    # Select one frequency bin to work with
    freqIdx = 100  # HARDCODED
    x_in = x_stft[freqIdx, :, :]
    y_in = y_stft[freqIdx, :, :]
    n_in = n_stft[freqIdx, :, :]
    
    # Initialize
    w = []
    dimYtilde = np.zeros(nNodes, dtype=int)
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        dimYtilde[k] = nSensorsPerNode + nNodes - 1
        wCurr = np.zeros((dimYtilde[k], nIter), dtype=np.complex128)
        wCurr[referenceSensorIdx, :] = 1
        w.append(wCurr)
    Ryy = []
    Rnn = []
    ryd = []
    for k in range(nNodes):
        Ryy.append(np.zeros((dimYtilde[k], dimYtilde[k]), dtype=np.complex128))
        Rnn.append(np.zeros((dimYtilde[k], dimYtilde[k]), dtype=np.complex128))
        ryd.append(np.zeros(dimYtilde[k], dtype=np.complex128))
    
    wNet = np.zeros((x_in.shape[1], nIter, nNodes), dtype=np.complex128)
    idxUpdatingNode = 0
    if nodeUpdatingStrategy == 'sequential':
        label = 'Online DANSE [seq NU]'
    else:
        label = 'Online DANSE [sim NU]'
    if filterType == 'gevd':
        label += ' [GEVD]'
    # Loop over frames
    for i in range(nIter - 1):
        print(f'{label} iteration {i+1}/{nIter}')
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(nNodes, dtype=np.complex128)
        fusedSignalsNoiseOnly = np.zeros(nNodes, dtype=np.complex128)
        for q in range(nNodes):
            yq = y_in[i, channelToNodeMap == q]
            fusedSignals[q] = yq @ w[q][:len(yq), i].conj()
            nq = n_in[i, channelToNodeMap == q]
            fusedSignalsNoiseOnly[q] = nq @ w[q][:len(nq), i].conj()
        
        # Loop over nodes
        for k in range(nNodes):
            # Get y tilde
            yTilde = np.concatenate((
                y_in[i, channelToNodeMap == k],
                fusedSignals[channelToNodeMap != k]
            ), axis=0)
            nTilde = np.concatenate((
                n_in[i, channelToNodeMap == k],
                fusedSignalsNoiseOnly[channelToNodeMap != k]
            ), axis=0)

            # Compute covariance matrices
            RyyCurr = np.outer(yTilde, yTilde.conj())
            RnnCurr = np.outer(nTilde, nTilde.conj())
            d = x_in[i, np.where(channelToNodeMap == k)[0][0]]
            rydCurr = yTilde.T * d.conj()
            # Update covariance matrices
            Ryy[k] = (1 - 1 / (i + 1)) * Ryy[k] + 1 / (i + 1) * RyyCurr
            Rnn[k] = (1 - 1 / (i + 1)) * Rnn[k] + 1 / (i + 1) * RnnCurr
            ryd[k] = (1 - 1 / (i + 1)) * ryd[k] + 1 / (i + 1) * rydCurr

            # Update filter
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False

            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    w[k][:, i + 1] = np.linalg.inv(Ryy[k]) @ ryd[k]
                elif filterType == 'gevd':
                    sigma, Xmat = la.eigh(Ryy[k], Rnn[k])
                    idx = np.flip(np.argsort(sigma))
                    sigma = sigma[idx]
                    Xmat = Xmat[:, idx]
                    Qmat = np.linalg.inv(Xmat.T.conj())
                    Dmat = np.zeros((Ryy[k].shape[0], Ryy[k].shape[0]))
                    Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
                    e = np.zeros(Ryy[k].shape[0])
                    e[:rank] = 1
                    w[k][:, i + 1] = Xmat @ Dmat @ Qmat.T.conj() @ e
            else:
                w[k][: , i + 1] = w[k][: , i]

        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes
        
        # Compute network-wide filters
        for k in range(nNodes):
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(x_in.shape[1]):
                # Node index corresponding to channel `m`
                currNode = channelToNodeMap[m]
                # Count channel index within node
                c = channelCount[currNode]
                if currNode == k:
                    wNet[m, i + 1, k] = w[k][c, i + 1]
                else:
                    nChannels_k = np.sum(channelToNodeMap == k)
                    gkq = w[k][nChannels_k + neighborCount, i + 1]
                    wNet[m, i + 1, k] = w[currNode][c, i] * gkq
                channelCount[currNode] += 1
                
                if currNode != k and c == np.sum(channelToNodeMap == currNode) - 1:
                    neighborCount += 1

    # Plot
    if 0:
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8.5, 3.5)
        for k in range(nNodes):
            axs.plot(np.abs(w[k][referenceSensorIdx, :]), label=f'Node {k+1}')
        axs.set_title('Online DANSE - Ref. sensor |w|')
        axs.legend()
        plt.show(block=False)

    return wNet


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
        gevdRank=1
    ):
    """Compute filters."""
    kwargs = {
        'noisySignals': noisySignals,
        'cleanSigs': cleanSigs,
        'noiseOnlySigs': noiseSignals,
        'channelToNodeMap': channelToNodeMap,
        'rank': gevdRank
    }
    filters = {}
    for filterType in TO_COMPUTE:
        filters[filterType] = compute_filter(type=filterType, **kwargs)

    return filters


def get_metrics(M, filters, scalings, sigma_sr, sigma_nr, channelToNodeMap):
    
    # Compute FAS + SPF
    fasAndSPF = np.zeros((M, M))
    for m in range(M):
        rtf = scalings / scalings[m]
        hs = np.sum(rtf ** 2) * sigma_sr[m] ** 2
        spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter
        fasAndSPF[:, m] = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

    # Compute difference between normalized estimated filters
    # and normalized expected filters
    diffs = {}
    for filterType in TO_COMPUTE:
        nFilters = filters[filterType].shape[-1]
        diffsPerCoefficient = np.zeros(nFilters)
        for m in range(nFilters):
            if 'danse' in filterType:  # DANSE case
                # Determine reference sensor index
                idxRef = np.where(channelToNodeMap == m)[0][0]
                currFilt = filters[filterType][:, -1, m]
            else:
                idxRef = m
                currFilt = filters[filterType][:, m]
            
            diffsPerCoefficient[m] = np.mean(np.abs(
                currFilt - fasAndSPF[:, idxRef]
            ))
        diffs[filterType] = np.mean(diffsPerCoefficient)

    return diffs


if __name__ == '__main__':
    sys.exit(main())
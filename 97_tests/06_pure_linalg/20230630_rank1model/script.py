# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import resampy
import scipy.linalg as la
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

TARGET_SIGNAL = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
N_SENSORS = 3
N_NODES = 3
SELFNOISE_POWER = 1
DURATIONS = np.logspace(np.log10(1), np.log10(30), 20)
# DURATIONS = [5]
FS = 16e3
N_MC = 1
EXPORT_FOLDER = '97_tests/06_pure_linalg/20230630_rank1model/figs'
RANDOM_DELAYS = False

# Type of signal
# SIGNAL_TYPE = 'speech'
# SIGNAL_TYPE = 'noise_real'
SIGNAL_TYPE = 'noise_complex'

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

    # For DANSE: randomly assign sensors to nodes, ensuring that each node
    # has at least one sensor
    channelToNodeMap = np.zeros(M, dtype=int)
    for k in range(K):
        channelToNodeMap[k] = k
    for n in range(K, M):
        channelToNodeMap[n] = np.random.randint(0, K)
    # Sort
    channelToNodeMap = np.sort(channelToNodeMap)

    diff = np.zeros((nMC, len(durations)))
    diffGEVD = np.zeros((nMC, len(durations)))
    diffDANSE = np.zeros((nMC, len(durations)))
    diffGEVDDANSE = np.zeros((nMC, len(durations)))
    diffDANSEsim = np.zeros((nMC, len(durations)))
    diffGEVDDANSEsim = np.zeros((nMC, len(durations)))
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

            kwargs = {
                'noisySignals': noisySignals,
                'cleanSigs': cleanSigs[:nSamples, :],
                'noiseOnlySigs': noiseSignals[:nSamples, :],
                'channelToNodeMap': channelToNodeMap
            }
            # MWF
            filterMWF = compute_filter(type='mwf', **kwargs)
            # GEVD-MWF
            filterGEVDMWF = compute_filter(type='gevdmwf', **kwargs)
            # DANSE (sequential node updating)
            filterDANSE = compute_filter(type='danse', **kwargs)
            # GEVD-DANSE (sequential node updating)
            filterGEVDDANSE = compute_filter(type='gevddanse', **kwargs)
            # DANSE (simultaneous node updating)
            filterDANSEsim = compute_filter(type='danse_sim', **kwargs)
            # GEVD-DANSE (simultaneous node updating)
            filterGEVDDANSEsim = compute_filter(type='gevddanse_sim', **kwargs)

            # Plot DANSE evolution
            if 0:
                plot_danse_evol(
                    K,
                    channelToNodeMap,
                    durations[ii],
                    filterDANSE,
                    scalings,
                    sigma_sr,
                    sigma_nr,
                    savefigs=True
                )
            
            # Compute difference between normalized estimated filters
            # and normalized expected filters
            diffsPerSensor = np.zeros(M)
            diffsPerSensorGEVD = np.zeros(M)
            for n in range(M):
                rtf = scalings / scalings[n]
                hs = np.sum(rtf ** 2) * sigma_sr[n] ** 2
                spf = hs / (hs + sigma_nr[n] ** 2)  # spectral post-filter
                fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter
                diffsPerSensor[n] = np.mean(np.abs(filterMWF[:, n] - fasAndSPF))
                diffsPerSensorGEVD[n] = np.mean(np.abs(
                    filterGEVDMWF[:, n] - fasAndSPF
                ))

            diffsPerNodeDANSE = np.zeros(K)
            diffsPerNodeGEVDDANSE = np.zeros(K)
            diffsPerNodeDANSEsim = np.zeros(K)
            diffsPerNodeGEVDDANSEsim = np.zeros(K)
            for k in range(K):
                # Determine reference sensor index
                idxRef = np.where(channelToNodeMap == k)[0][0]
                #
                rtf = scalings / scalings[idxRef]
                hs = np.sum(rtf ** 2) * sigma_sr[idxRef] ** 2
                spf = hs / (hs + sigma_nr[idxRef] ** 2)  # spectral post-filter
                fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter
                diffsPerNodeDANSE[k] = np.mean(np.abs(
                    filterDANSE[:, -1, k] - fasAndSPF
                ))
                diffsPerNodeGEVDDANSE[k] = np.mean(np.abs(
                    filterGEVDDANSE[:, -1, k] - fasAndSPF
                ))
                diffsPerNodeDANSEsim[k] = np.mean(np.abs(
                    filterDANSEsim[:, -1, k] - fasAndSPF
                ))
                diffsPerNodeGEVDDANSEsim[k] = np.mean(np.abs(
                    filterGEVDDANSEsim[:, -1, k] - fasAndSPF
                ))

            diff[idxMC, ii] = np.mean(diffsPerSensor)
            diffGEVD[idxMC, ii] = np.mean(diffsPerSensorGEVD)
            diffDANSE[idxMC, ii] = np.mean(diffsPerNodeDANSE)
            diffGEVDDANSE[idxMC, ii] = np.mean(diffsPerNodeGEVDDANSE)
            diffDANSEsim[idxMC, ii] = np.mean(diffsPerNodeDANSEsim)
            diffGEVDDANSEsim[idxMC, ii] = np.mean(diffsPerNodeGEVDDANSEsim)

        # Plots
        if 0:
            filteredSignals = np.zeros((nSamples, M))
            filteredSignal_RTFs = np.zeros((nSamples, M))
            for n in range(M):
                filteredSignals[:, n] = noisySignals @ filterMWF[:, n]
                # Compute filtered signal using RTFs
                h = scalings / scalings[n]
                h2 = np.sum(h ** 2)
                hs = h2 * sigma_sr[n] ** 2
                spf = hs / (hs + sigma_nr[n] ** 2)
                filteredSignal_RTFs[:, n] = noisySignals @ h / h2 * spf
            # plot_results(
            #     cleanSigs[:nSamples, :],
            #     noisySignals,
            #     filteredSignals,
            #     filteredSignal_RTFs
            # )
            plot_filter(filterMWF, scalings, sigma_sr, sigma_nr)
            stop = 1

    # Plot difference
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.loglog(durations, diffGEVD.T, '-', color='#FFCACA')
    axes.loglog(durations, diffGEVDDANSE.T, ':', color='#FACAFF')
    axes.loglog(durations, diffGEVDDANSEsim.T, '-.', color='#CEFFCA')
    axes.loglog(durations, diff.T, '--', color='0.75')
    axes.loglog(durations, diffDANSE.T, ':', color='#CECAFF')
    axes.loglog(durations, diffDANSEsim.T, '-.', color='#CAFFFC')
    axes.loglog(durations, np.mean(diffGEVD, axis=0), '.-', color='r', label=f'GEVD-MWF (mean over $M$={M} sensors)')
    axes.loglog(durations, np.mean(diffGEVDDANSE, axis=0), '.:', color='m', label=f'GEVD-DANSE (mean over $K$={K} nodes)')
    axes.loglog(durations, np.mean(diffGEVDDANSEsim, axis=0), '.-.', color='g', label=f'rS-GEVD-DANSE (mean over $K$={K} nodes)')
    axes.loglog(durations, np.mean(diff, axis=0), '.--', color='k', label=f'MWF (mean over $M$={M} sensors)')
    axes.loglog(durations, np.mean(diffDANSE, axis=0), '.:', color='b', label=f'DANSE (mean over $K$={K} nodes)')
    axes.loglog(durations, np.mean(diffDANSEsim, axis=0), '.-.', color='c', label=f'rS-DANSE (mean over $K$={K} nodes)')
    plt.grid(which='both')
    axes.legend(loc='lower left')
    plt.xlabel('Signal duration (s)')
    plt.ylabel('Abs. diff. bw. MWF and FAS + SPF')
    axes.set_title(f'{nMC} MC runs - target signal: "{SIGNAL_TYPE}"')
    fig.tight_layout()
    plt.show(block=False)

    if 0:
        fname = f'{EXPORT_FOLDER}/diff'
        if RANDOM_DELAYS:
            fname += '_randomDelays'
        fig.savefig(f'{fname}_{SIGNAL_TYPE}.png', dpi=300, bbox_inches='tight')

    stop = 1


def plot_danse_evol(
        K,
        channelToNodeMap,
        dur,
        filterDANSE,
        scalings,
        sigma_sr,
        sigma_nr,
        savefigs=False
    ):
    """ Plot DANSE evolution. """
    nSensors = filterDANSE.shape[0]
    for k in range(K):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8.5, .5)
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
                lab += ' (reference)'
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
                    label=f'$[\\mathbf{{w}}_{{\\mathrm{{FAS}},k}}]_{m+1}$'
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
        ax.legend()
        ax.grid(which='both')
        plt.xlabel('Iteration index $i$')
        fig.tight_layout()
        if savefigs:
            fig.savefig(f'{EXPORT_FOLDER}/danse_evol_n{k+1}_dur{int(dur * 1e3)}ms.png', dpi=300, bbox_inches='tight')
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
    if type == 'mwf':
        RyyInv = np.linalg.inv(Ryy)
        if np.iscomplex(cleanSigs).any():
            w = np.zeros((nSensors, nSensors), dtype=np.complex128)
        else:
            w = np.zeros((nSensors, nSensors))
        for n in range(nSensors):
            Ryd = noisySignals.T.conj() @ cleanSigs[:, n]
            w[:, n] = RyyInv @ Ryd
    elif type == 'gevdmwf':
        Rnn = noiseOnlySigs.T.conj() @ noiseOnlySigs
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
        label = 'DANSE [seq NU]'
    else:
        label = 'DANSE [sim NU]'
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
    raise NotImplementedError('Online DANSE not implemented yet')
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
    # Run DANSEz
    for iter in range(maxIter):
        print(f'DANSE iteration {iter+1}/{maxIter}')
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
        latentSignal, fs = sf.read(TARGET_SIGNAL)
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

if __name__ == '__main__':
    sys.exit(main())
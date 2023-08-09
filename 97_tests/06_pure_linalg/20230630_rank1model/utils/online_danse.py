# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import scipy.linalg as la
from scipy.signal import stft
from dataclasses import dataclass
from matplotlib import pyplot as plt

@dataclass
class WOLAparameters:
    nfft: int = 1024
    hop: int = 512
    winType: str = 'sqrt-hann'  # sqrt-hann | rect
    fs: int = 16000  # [Hz]
    betaDanse: float = 0.99  # exponential averaging constant
    betaMwf: float = 0.99  # exponential averaging constant for MWF
    B: int = 0  # number of frames bw. consecutive updates of fusion vectors
    alpha: float = 1  # exponential averaging constant for fusion vectors
    betaExt: float = 0.99  # exponential averaging constant for external target filters
    startExpAvgAfter: int = 0  # number of frames after which to start exponential averaging
    startFusionExpAvgAfter: int = 0  # same as above, for the fusion vectors

def get_window(winType, nfft):
    
    if winType == 'hann':
        win = np.hanning(nfft)
    elif winType == 'sqrt-hann':
        win = np.sqrt(np.hanning(nfft))
    elif winType == 'hamming':
        win = np.hamming(nfft)
    elif winType == 'rect':
        win = np.ones(nfft)
    else:
        raise ValueError('Window type not recognized')

    return win


def run_wola_danse(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        nfft=1024,
        hop=512,
        windowType='sqrt-hann',  # `sqrt-hann` or `rect`
        referenceSensorIdx=0,
        fs=16000,
        beta=0.99,  # exponential averaging constant
        upExtTargetFiltEvery=1,  # time [s] bw. consecutive updates of external target filters
        upFusionVectEvery=0  # number of frames bw. consecutive updates of fusion vectors
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1

    # Check method
    if np.iscomplex(x).any() or np.iscomplex(n).any():
        print('Complex-valued signal as input of online DANSE: no WOLA permitted.')
        if windowType != 'rect':
            print('Setting rectangular window.')
            windowType = 'rect'
        if hop != nfft:
            print('Setting no window overlap (`hop` = `nfft`)')
            hop = nfft
        # Section signal
        nIter = x.shape[0] // nfft
        x_in = np.zeros((nfft, nIter, x.shape[1]), dtype=np.complex128)
        n_in = np.zeros((nfft, nIter, n.shape[1]), dtype=np.complex128)
        y_in = np.zeros((nfft, nIter, y.shape[1]), dtype=np.complex128)
        for ii in range(nIter):
            idxBeg = ii * nfft
            idxEnd = (ii + 1) * nfft
            x_in[:, ii, :] = x[idxBeg:idxEnd, :]
            n_in[:, ii, :] = n[idxBeg:idxEnd, :]
            y_in[:, ii, :] = y[idxBeg:idxEnd, :]
        # Get window
        win = get_window(windowType, nfft)
    
    else:  # WOLA-processing, STFTs
        # Convert to STFT domain using SciPy
        # Get window
        win = get_window(windowType, nfft)
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
        x_in = x_stft.reshape((x_stft.shape[0], -1, x.shape[1]))
        y_in = y_stft.reshape((y_stft.shape[0], -1, y.shape[1]))
        n_in = n_stft.reshape((n_stft.shape[0], -1, n.shape[1]))
        nIter = x_in.shape[1]
        nPosFreqs = x_in.shape[0]
    
    # Initialize
    w = []
    dimYtilde = np.zeros(nNodes, dtype=int)
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        dimYtilde[k] = nSensorsPerNode + nNodes - 1
        wCurr = np.zeros((nPosFreqs, dimYtilde[k], nIter), dtype=np.complex128)
        wCurr[:, referenceSensorIdx, :] = 1
        w.append(wCurr)
    Ryy = []
    Rnn = []
    for k in range(nNodes):
        Ryy.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        Rnn.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
    
    wNet = np.zeros(
        (nPosFreqs, x_in.shape[-1], nIter, nNodes),
        dtype=np.complex128
    )
    idxUpdatingNode = 0
    if nodeUpdatingStrategy == 'sequential':
        label = 'Online DANSE [seq NU]'
    elif nodeUpdatingStrategy == 'simultaneous':
        label = 'Online DANSE [sim NU]'
        wExt = []
        wExtTarget = []
        for k in range(nNodes):
            nSensorsPerNode = np.sum(channelToNodeMap == k)
            wExt.append(np.zeros(
                (nPosFreqs, nSensorsPerNode, nIter),
                dtype=np.complex128
            ))
            wExtTarget.append(np.zeros(
                (nPosFreqs, nSensorsPerNode, nIter),
                dtype=np.complex128
            ))
        lastExtFiltUp = -1
    if filterType == 'gevd':
        label += ' [GEVD]'
    nodeIndices = np.arange(nNodes)
    nRyyUpdates = np.zeros(nNodes, dtype=int)
    nRnnUpdates = np.zeros(nNodes, dtype=int)
    lastFusionVectorUpdate = -1 * np.ones(nNodes)
    # Loop over frames
    for i in range(nIter - 1):
        print(f'{label} iteration {i+1}/{nIter}')
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(
            (nPosFreqs, nNodes),
            dtype=np.complex128
        )
        fusedSignalsNoiseOnly = np.zeros(
            (nPosFreqs, nNodes),
            dtype=np.complex128
        )

        for q in range(nNodes):
            # Define fusion vector
            if nodeUpdatingStrategy == 'sequential':
                fusionFilt = w[q][:, :yq.shape[1], i].conj()
            elif nodeUpdatingStrategy == 'simultaneous':
                # Deal with external target filters
                currTime = i * hop / fs
                if lastExtFiltUp == -1 or currTime - lastExtFiltUp >= upExtTargetFiltEvery:
                    print(f'Updating external target filters at time {np.round(currTime, 3)} s...')
                    # Update external target filters
                    wExtTarget[q][:, i, :] = .5 * (wExtTarget[q][:, i, :] +\
                        w[q][:, :yq.shape[1], i].conj())
                    lastExtFiltUp = currTime
                else:
                    # Do not update external target filters
                    wExtTarget[q][:, i, :] = wExtTarget[q][:, i - 1, :]
                
                # Update external filters (effectively used)
                wExt[q][:, i, :] = beta * wExt[:, i - 1, :] +\
                    (1 - beta) * wExtTarget[:, i, :]
                fusionFilt = wExt[q][:, i, :]
            # Perform fusion
            yq = y_in[:, i, channelToNodeMap == q]
            fusedSignals[:, q] = np.einsum('ij,ij->i', yq, fusionFilt)
            nq = n_in[:, i, channelToNodeMap == q]
            fusedSignalsNoiseOnly[:, q] = np.einsum('ij,ij->i', nq, fusionFilt)
        
        # Loop over nodes
        for k in range(nNodes):
            # Get y tilde
            yTilde = np.concatenate((
                y_in[:, i, channelToNodeMap == k],
                fusedSignals[:, nodeIndices != k]
            ), axis=1)
            nTilde = np.concatenate((
                n_in[:, i, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, nodeIndices != k]
            ), axis=1)

            # Compute covariance matrices
            RyyCurr = np.einsum('ij,ik->ijk', yTilde, yTilde.conj())
            RnnCurr = np.einsum('ij,ik->ijk', nTilde, nTilde.conj())
            # Update covariance matrices
            Ryy[k] = beta * Ryy[k] + (1 - beta) * RyyCurr
            Rnn[k] = beta * Rnn[k] + (1 - beta) * RnnCurr
            # Update counter
            nRyyUpdates[k] += 1
            nRnnUpdates[k] += 1

            # Check if filter ought to be updated
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False
            if updateFilter:
                # Check if Ryy is full rank
                if np.any(np.linalg.matrix_rank(Ryy[k]) < dimYtilde[k]):
                    print(f'Rank-deficient Ryy[{k}]')
                    updateFilter = False
                # Check if Rnn is full rank
                if np.any(np.linalg.matrix_rank(Rnn[k]) < dimYtilde[k]):
                    print(f'Rank-deficient Rnn[{k}]')
                    updateFilter = False
            
            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    e = np.zeros(w[k].shape[1])
                    e[referenceSensorIdx] = 1  # selection vector
                    for kappa in range(nPosFreqs):
                        ryd = (Ryy[k][kappa, :, :] - Rnn[k][kappa, :, :]) @ e
                        w[k][kappa, :, i + 1] =\
                            np.linalg.inv(Ryy[k][kappa, :, :]) @ ryd
                elif filterType == 'gevd':
                    raise NotImplementedError('not yet implemented for multiple frequency lines')
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
                w[k][:, :, i + 1] = w[k][:, :, i]

        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes
        
        # Compute network-wide filters
        for k in range(nNodes):
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(x_in.shape[-1]):
                # Node index corresponding to channel `m`
                currNode = channelToNodeMap[m]
                # Count channel index within node
                c = channelCount[currNode]
                if currNode == k:
                    wNet[:, m, i + 1, k] = w[k][:, c, i + 1]
                else:
                    nChannels_k = np.sum(channelToNodeMap == k)
                    gkq = w[k][:, nChannels_k + neighborCount, i + 1]
                    wNet[:, m, i + 1, k] = w[currNode][:, c, i] * gkq
                channelCount[currNode] += 1
                
                if currNode != k and c == np.sum(channelToNodeMap == currNode) - 1:
                    neighborCount += 1

    # Plot
    if 0:
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(8.5, 3.5)
        for k in range(nNodes):
            axs.plot(
                np.abs(w[k][:, referenceSensorIdx, :]),
                label=f'Node {k+1}'
            )
        axs.set_title('Online DANSE - Ref. sensor |w|')
        axs.legend()
        plt.show(block=False)

    return wNet


def run_online_danse(
        x: np.ndarray,
        n: np.ndarray,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        L=1024,
        referenceSensorIdx=0,
        beta=0.99,  # exponential averaging constant
        B=0,  # number of frames bw. consecutive updates of fusion vectors
        alpha=0.5,  # exponential averaging constant for fusion vectors
        betaExt=0.99,  # exponential averaging constant for external target filters
        batchModeNetWideFilters=None,
        startExpAvgAfter=0, # number of frames after which to start exponential averaging
        startFusionExpAvgAfter=0, # same as above, for the fusion vectors
        ignoreFusionForSSNodes=False,  # if True, fusion vectors are not updated for single-sensor nodes
        verbose=True,
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    nSensors = len(channelToNodeMap)

    # Number of frames
    nIter = y.shape[0] // L

    # Initialize
    w = []
    dimYtilde = np.zeros(nNodes, dtype=int)
    for k in range(nNodes):
        dimYtilde[k] = np.sum(channelToNodeMap == k) + nNodes - 1
        wCurr = np.zeros((dimYtilde[k], nIter), dtype=np.complex128)
        wCurr[referenceSensorIdx, :] = 1
        w.append(wCurr)
    Ryy = []
    Rnn = []
    for k in range(nNodes):
        Ryy.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        Rnn.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        # Ryy.append(
        #     np.random.uniform(0, 1000, (dimYtilde[k], dimYtilde[k])) +\
        #     1j * np.random.uniform(0, 1000, (dimYtilde[k], dimYtilde[k]))
        # )
        # Rnn.append(
        #     np.random.uniform(0, 1000, (dimYtilde[k], dimYtilde[k])) +\
        #     1j * np.random.uniform(0, 1000, (dimYtilde[k], dimYtilde[k]))
        # )

    wNet = np.zeros(
        (nSensors, nIter, nNodes),
        dtype=np.complex128
    )
    idxUpdatingNode = 0
    if nodeUpdatingStrategy == 'sequential':
        label = 'Online DANSE [seq NU]'
    else:
        label = 'Online DANSE [sim NU]'
        wExt = []
        wExtTarget = []
        for k in range(nNodes):
            nSensorsPerNode = np.sum(channelToNodeMap == k)
            wExt.append(np.zeros(
                (nSensorsPerNode, nIter),
                dtype=np.complex128
            ))
            wExtTarget.append(np.zeros(
                (nSensorsPerNode, nIter),
                dtype=np.complex128
            ))
    if filterType == 'gevd':
        label += ' [GEVD]'
    nodeIndices = np.arange(nNodes)

    # Initialize fusion vectors
    fusionVectTargets = [None for _ in range(nNodes)]
    fusionVects = [None for _ in range(nNodes)]
    for k in range(nNodes):
        baseVect = np.zeros(sum(channelToNodeMap == k))
        baseVect[0] = 1
        fusionVectTargets[k] = baseVect
        fusionVects[k] = baseVect
    lastFuVectUp = -1 * np.ones(nNodes)
    # Loop over frames
    for i in range(nIter - 1):
        if verbose:
            print(f'{label} iteration {i+1}/{nIter}')
        idxBegFrame = i * L
        idxEndFrame = (i + 1) * L
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(
            (L, nNodes),
            dtype=np.complex128
        )
        fusedSignalsNoiseOnly = np.zeros(
            (L, nNodes),
            dtype=np.complex128
        )
        
        # Update fusion vectors and perform fusion
        for k in range(nNodes):
            yk = y[idxBegFrame:idxEndFrame, channelToNodeMap == k]
            nk = n[idxBegFrame:idxEndFrame, channelToNodeMap == k]

            if ignoreFusionForSSNodes and yk.shape[1] == 1:
                pass  # do not update the fusion vector for single-sensor nodes
            else:
                if batchModeNetWideFilters is None:

                    # Update target fusion vector
                    if i > startFusionExpAvgAfter:
                        # Update using `beta_EXT`
                        fusionVectTargets[k] = betaExt * fusionVectTargets[k] +\
                            (1 - betaExt) * w[k][:yk.shape[1], i]
                    else:
                        # Update without `beta_EXT`
                        fusionVectTargets[k] = w[k][:yk.shape[1], i]

                    # Check if effective fusion vector is to be updated
                    if lastFuVectUp[k] == -1 or i - lastFuVectUp[k] >= B:
                        fusionVects[k] = (1 - alpha) * fusionVects[k] +\
                            alpha * fusionVectTargets[k]
                        lastFuVectUp[k] = i
                else:
                    # Compute index where the fusion filters are stored in the
                    # batch-mode network-wide DANSE filters
                    fusionVects[k] = batchModeNetWideFilters[channelToNodeMap == k, -1, k]
            
            # Perform fusion
            fusedSignals[:, k] = yk @ fusionVects[k].conj()
            fusedSignalsNoiseOnly[:, k] = nk @ fusionVects[k].conj()
        
        # Filter update loop
        for k in range(nNodes):
            # Get y tilde
            yTilde = np.concatenate((
                y[idxBegFrame:idxEndFrame, channelToNodeMap == k],
                fusedSignals[:, nodeIndices != k]
            ), axis=1)
            nTilde = np.concatenate((
                n[idxBegFrame:idxEndFrame, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, nodeIndices != k]
            ), axis=1)

            # Compute covariance matrices
            RyyCurr = np.einsum('ij,ik->jk', yTilde, yTilde.conj())
            RnnCurr = np.einsum('ij,ik->jk', nTilde, nTilde.conj())
            # Update covariance matrices
            if i > startExpAvgAfter:
                # Using `beta`
                Ryy[k] = beta * Ryy[k] + (1 - beta) * RyyCurr
                Rnn[k] = beta * Rnn[k] + (1 - beta) * RnnCurr
            else:
                # Without `beta`
                Ryy[k] = RyyCurr
                Rnn[k] = RnnCurr

            # Check if filter ought to be updated
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False
            if updateFilter:
                # Check if Ryy is full rank
                if np.linalg.matrix_rank(Ryy[k]) < dimYtilde[k]:
                    print(f'Rank-deficient Ryy[{k}]')
                    updateFilter = False
                # Check if Rnn is full rank
                if np.linalg.matrix_rank(Rnn[k]) < dimYtilde[k]:
                    print(f'Rank-deficient Rnn[{k}]')
                    updateFilter = False

            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    e = np.zeros(dimYtilde[k])
                    e[referenceSensorIdx] = 1  # selection vector
                    ryd = (Ryy[k] - Rnn[k]) @ e
                    w[k][:, i + 1] = np.linalg.inv(Ryy[k]) @ ryd
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
                w[k][:, i + 1] = w[k][:, i]

        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes
        
        # Compute network-wide filters
        for k in range(nNodes):
            Mk = np.sum(channelToNodeMap == k)
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(nSensors):
                # Node index corresponding to channel `m`
                currNode = channelToNodeMap[m]
                # Channel index within node
                cIdx = channelCount[currNode]
                if currNode == k:  # current node channel
                    wNet[m, i + 1, k] = w[currNode][cIdx, i + 1]
                else:  # neighbor node channel
                    gkq = w[k][Mk + neighborCount, i + 1]
                    # wNet[m, i + 1, k] = w[currNode][cIdx, i] * gkq
                    wNet[m, i + 1, k] = fusionVects[currNode][cIdx] * gkq
                    # If we have reached the last channel of the current 
                    # neighbor node, increment neighbor count
                    if cIdx == np.sum(channelToNodeMap == currNode) - 1:
                        neighborCount += 1
                channelCount[currNode] += 1

    return wNet
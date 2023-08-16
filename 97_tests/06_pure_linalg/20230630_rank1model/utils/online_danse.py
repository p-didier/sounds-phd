# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass

@dataclass
class WOLAparameters:
    nfft: int = 1024
    hop: int = 512
    nPosFreqs: int = int(nfft / 2 + 1)
    winType: str = 'sqrt-hann'  # sqrt-hann | rect
    fs: int = 16000  # [Hz]
    betaDanse: float = 0.99  # exponential averaging constant
    betaMwf: float = 0.99  # exponential averaging constant for MWF
    B: int = 0  # number of frames bw. consecutive updates of fusion vectors
    alpha: float = 1  # exponential averaging constant for fusion vectors
    betaExt: float = 0  # exponential averaging constant for external target filters
    startExpAvgAfter: int = 0  # number of frames after which to start exponential averaging
    startFusionExpAvgAfter: int = 0  # same as above, for the fusion vectors
    #
    singleFreqBinIndex: int = None  # if not None, consider only the freq. bin at this index for WOLA-DANSE
    
    def __post_init__(self):
        if self.singleFreqBinIndex is not None:
            if self.singleFreqBinIndex > self.nPosFreqs:
                raise ValueError('singleFreqBinIndex cannot be larger than nPosFreqs')

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
        x: np.ndarray,
        n: np.ndarray,
        p: WOLAparameters,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        referenceSensorIdx=0,
        ignoreFusionForSSNodes=False,  # if True, fusion vectors are not updated for single-sensor nodes
        verbose=False,
    ):

    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    # Get number of nodes
    nSensors = x.shape[1]
    # Get number of frames
    nIter = x.shape[0] // p.hop - 1
    # Get noisy signal (time-domain)
    y = x + n

    # Convert to WOLA domain
    win = get_window(p.winType, p.nfft)
    yWola = np.zeros((nIter, p.nfft, nSensors), dtype=np.complex128)
    nWola = np.zeros((nIter, p.nfft, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        for i in range(nIter):
            idxBegFrame = i * p.hop
            idxEndFrame = idxBegFrame + p.nfft
            yWola[i, :, m] = np.fft.fft(
                y[idxBegFrame:idxEndFrame, m] * win
            ) / np.sqrt(p.hop)
            nWola[i, :, m] = np.fft.fft(
                n[idxBegFrame:idxEndFrame, m] * win
            ) / np.sqrt(p.hop)
    # Get number of positive frequencies
    nPosFreqs = p.nfft // 2 + 1
    # Keep only positive frequencies (spare computations)
    yWola = yWola[:, :nPosFreqs, :]
    nWola = nWola[:, :nPosFreqs, :]
    
    # Special user request: single frequency bin study
    if p.singleFreqBinIndex is not None:
        if verbose:
            print(f'/!\ /!\ /!\ WOLA-DANSE: single frequency bin study (index {p.singleFreqBinIndex})')
        yWola = yWola[:, p.singleFreqBinIndex, :]
        nWola = nWola[:, p.singleFreqBinIndex, :]
        yWola = np.expand_dims(yWola, axis=1)  # add singleton dimension
        nWola = np.expand_dims(nWola, axis=1)
        nPosFreqs = 1
    
    # Initialize
    w = []
    dimYtilde = np.zeros(nNodes, dtype=int)
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        dimYtilde[k] = nSensorsPerNode + nNodes - 1
        wCurr: np.ndarray = np.zeros(
            (nPosFreqs, nIter, dimYtilde[k]),
            dtype=np.complex128
        )
        wCurr[:, :, referenceSensorIdx] = 1
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
        # Ryy.append(np.random.randn(
        #     nPosFreqs, dimYtilde[k], dimYtilde[k]
        # ))
        # Rnn.append(np.random.randn(
        #     nPosFreqs, dimYtilde[k], dimYtilde[k]
        # ))
    
    # Initialize network-wide DANSE filters
    wNet = np.zeros(
        (nSensors, nIter, nPosFreqs, nNodes),
        dtype=np.complex128
    )
    for k in range(nNodes):
        idxRef = np.where(channelToNodeMap == k)[0][referenceSensorIdx]
        wNet[idxRef, :, :, k] = 1  # initialize with identity matrix (selecting ref. sensor)
    # Prepare for DANSE filter updates
    idxUpdatingNode = 0
    if nodeUpdatingStrategy == 'sequential':
        algLabel = 'WOLA-DANSE [seq NU]'
    elif nodeUpdatingStrategy == 'simultaneous':
        algLabel = 'WOLA-DANSE [sim NU]'
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
    if filterType == 'gevd':
        algLabel += ' [GEVD]'
    nodeIndices = np.arange(nNodes)

    # Initialize fusion vectors
    fusionVectTargets = [None for _ in range(nNodes)]
    fusionVects = [None for _ in range(nNodes)]
    for k in range(nNodes):
        # baseVect = np.ones((nPosFreqs, sum(channelToNodeMap == k)))
        baseVect = np.zeros((nPosFreqs, sum(channelToNodeMap == k)))
        baseVect[:, referenceSensorIdx] = 1  # select reference sensor
        fusionVectTargets[k] = baseVect
        fusionVects[k] = baseVect
    lastFuVectUp = -1 * np.ones(nNodes)

    # Loop over frames
    for i in range(nIter - 1):
        if verbose:
            print(f'{algLabel} iteration {i+1}/{nIter}')
        
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(
            (nPosFreqs, nNodes),
            dtype=np.complex128
        )
        fusedSignalsNoiseOnly = np.zeros(
            (nPosFreqs, nNodes),
            dtype=np.complex128
        )

        # Get current frame [necessary step because of some unexplained NumPy
        # behaviour which make it so that:
        # shape(yWola[i, :, channelToNodeMap == k]) = (Mk, nPosFreqs)
        # instead of (nPosFreqs, Mk)]
        yCurr = yWola[i, :, :]
        nCurr = nWola[i, :, :]

        # Update fusion vectors and perform fusion
        for k in range(nNodes):
            yk: np.ndarray = yCurr[:, channelToNodeMap == k]
            nk: np.ndarray = nCurr[:, channelToNodeMap == k]

            if ignoreFusionForSSNodes and yk.shape[1] == 1:
                pass  # do not update the fusion vector for single-sensor nodes
            else:
                # Update target fusion vector
                if i > p.startFusionExpAvgAfter:
                    # Update using `beta_EXT`
                    fusionVectTargets[k] = p.betaExt * fusionVectTargets[k] +\
                        (1 - p.betaExt) * w[k][:, i, :yk.shape[1]]
                else:
                    # Update without `beta_EXT`
                    fusionVectTargets[k] = w[k][:, i, :yk.shape[1]]

                # Check if effective fusion vector is to be updated
                if i - lastFuVectUp[k] >= p.B:
                    fusionVects[k] = (1 - p.alpha) * fusionVects[k] +\
                        p.alpha * fusionVectTargets[k]
                    lastFuVectUp[k] = i

            # Perform fusion
            fusedSignals[:, k] = np.einsum(
                'ij,ij->i',
                yk,
                fusionVects[k].conj()
            )
            fusedSignalsNoiseOnly[:, k] = np.einsum(
                'ij,ij->i',
                nk,
                fusionVects[k].conj()
            )

        # Loop over nodes
        for k in range(nNodes):
            # Get y tilde
            yTilde: np.ndarray = np.concatenate((
                yCurr[:, channelToNodeMap == k],
                fusedSignals[:, nodeIndices != k]
            ), axis=1)
            nTilde: np.ndarray = np.concatenate((
                nCurr[:, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, nodeIndices != k]
            ), axis=1)

            # Compute covariance matrices
            RyyCurr = np.einsum('ij,ik->ijk', yTilde, yTilde.conj())
            RnnCurr = np.einsum('ij,ik->ijk', nTilde, nTilde.conj())
            # Update covariance matrices
            if i > p.startExpAvgAfter:
                Ryy[k] = p.betaDanse * Ryy[k] + (1 - p.betaDanse) * RyyCurr
                Rnn[k] = p.betaDanse * Rnn[k] + (1 - p.betaDanse) * RnnCurr
            else:
                Ryy[k] = copy.deepcopy(RyyCurr)
                Rnn[k] = copy.deepcopy(RnnCurr)

            # Check if filter should be updated
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False
            
            # Check if SCMs are full rank 
            if updateFilter:
                if np.any(np.linalg.matrix_rank(Ryy[k]) < dimYtilde[k]):
                    if verbose:
                        print(f'Rank-deficient Ryy[{k}]')
                    updateFilter = False
                if np.any(np.linalg.matrix_rank(Rnn[k]) < dimYtilde[k]):
                    if verbose:
                        print(f'Rank-deficient Rnn[{k}]')
                    updateFilter = False
            
            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    e = np.zeros(w[k].shape[-1])
                    e[referenceSensorIdx] = 1  # selection vector
                    w[k][:, i + 1, :] = np.linalg.inv(Ryy[k]) @\
                        (Ryy[k] - Rnn[k]) @ e
                elif filterType == 'gevd':
                    Xmat = np.zeros(
                        (nPosFreqs, dimYtilde[k], dimYtilde[k]),
                        dtype=np.complex128
                    )
                    sigma = np.zeros((nPosFreqs, dimYtilde[k]))
                    for kappa in range(nPosFreqs):
                        sigmaCurr, XmatCurr = la.eigh(
                            Ryy[k][kappa, :, :],
                            Rnn[k][kappa, :, :]
                        )
                        indices = np.flip(np.argsort(sigmaCurr))
                        sigma[kappa, :] = sigmaCurr[indices]
                        Xmat[kappa, :, :] = XmatCurr[:, indices]
                    Qmat = np.linalg.inv(
                        np.transpose(Xmat.conj(), axes=[0, 2, 1])
                    )
                    # GEVLs tensor
                    Dmat = np.zeros((nPosFreqs, dimYtilde[k], dimYtilde[k]))
                    for r in range(rank):
                        Dmat[:, r, r] = np.squeeze(1 - 1 / sigma[:, r])
                    e = np.zeros(dimYtilde[k])
                    e[:rank] = 1
                    Qhermitian = np.transpose(Qmat.conj(), axes=[0, 2, 1])
                    wCurr = np.matmul(np.matmul(
                        np.matmul(Xmat, Dmat),
                        Qhermitian
                    ), e)
                    w[k][:, i + 1, :] = wCurr
            else:
                w[k][:, i + 1, :] = w[k][:, i, :]

        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes
        
        # Compute network-wide filters
        for k in range(nNodes):
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(nSensors):
                # Node index corresponding to channel `m`
                currNetWideNodeIdx = channelToNodeMap[m]
                # Count channel index within node
                c = channelCount[currNetWideNodeIdx]
                if currNetWideNodeIdx == k:
                    wNet[m, i + 1, :, k] = w[k][:, i + 1, c]
                else:
                    nChannels_k = np.sum(channelToNodeMap == k)
                    gkq = w[k][:, i + 1, nChannels_k + neighborCount]
                    wNet[m, i + 1, :, k] =\
                        fusionVects[currNetWideNodeIdx][:, c] * gkq
                channelCount[currNetWideNodeIdx] += 1
                
                if currNetWideNodeIdx != k and\
                    c == np.sum(channelToNodeMap == currNetWideNodeIdx) - 1:
                    neighborCount += 1

    return wNet


def run_online_danse(
        x: np.ndarray,
        n: np.ndarray,
        p: WOLAparameters,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        referenceSensorIdx=0,
        batchModeNetWideFilters=None,
        ignoreFusionForSSNodes=False,  # if True, fusion vectors are not updated for single-sensor nodes
        verbose=True,
        vad=None
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    nSensors = len(channelToNodeMap)

    # Number of frames
    nIter = y.shape[0] // p.nfft

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
    RyyCurr = []
    RnnCurr = []
    for k in range(nNodes):
        Ryy.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        Rnn.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        RyyCurr.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        RnnCurr.append(np.zeros(
            (dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
    nUpdatesRyy = np.zeros(nNodes, dtype=int)
    nUpdatesRnn = np.zeros(nNodes, dtype=int)

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
        idxBegFrame = i * p.nfft
        idxEndFrame = (i + 1) * p.nfft
        # Get VAD for current frame
        if vad is not None:
            vadCurr = vad[idxBegFrame:idxEndFrame]
            # Convert to single boolean value
            vadCurr = np.any(vadCurr.astype(bool))
        
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(
            (p.nfft, nNodes),
            dtype=np.complex128
        )
        fusedSignalsNoiseOnly = np.zeros(
            (p.nfft, nNodes),
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
                    if i > p.startFusionExpAvgAfter:
                        # Update using `beta_EXT`
                        fusionVectTargets[k] = p.betaExt * fusionVectTargets[k] +\
                            (1 - p.betaExt) * w[k][:yk.shape[1], i]
                    else:
                        # Update without `beta_EXT`
                        fusionVectTargets[k] = w[k][:yk.shape[1], i]

                    # Check if effective fusion vector is to be updated
                    if lastFuVectUp[k] == -1 or i - lastFuVectUp[k] >= p.B:
                        fusionVects[k] = (1 - p.alpha) * fusionVects[k] +\
                            p.alpha * fusionVectTargets[k]
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
            
            if vad is not None:
                # Compute covariance matrices following VAD
                if vadCurr:
                    RyyCurr[k] = np.einsum('ij,ik->jk', yTilde, yTilde.conj())
                    nUpdatesRyy[k] += 1
                else:
                    RnnCurr[k] = np.einsum('ij,ik->jk', yTilde, yTilde.conj())
                    nUpdatesRnn[k] += 1
                # Condition to start exponential averaging
                startExpAvgCond = nUpdatesRyy[k] > 1 and nUpdatesRnn[k] > 1
            else:
                # Compute covariance matrices
                RyyCurr[k] = np.einsum('ij,ik->jk', yTilde, yTilde.conj())
                RnnCurr[k] = np.einsum('ij,ik->jk', nTilde, nTilde.conj())
                # Condition to start exponential averaging
                startExpAvgCond = i > p.startExpAvgAfter
            
            # Update covariance matrices
            if startExpAvgCond:
                # Using `beta`
                Ryy[k] = p.betaDanse * Ryy[k] +\
                    (1 - p.betaDanse) * RyyCurr[k]
                Rnn[k] = p.betaDanse * Rnn[k] +\
                    (1 - p.betaDanse) * RnnCurr[k]
            else:
                # Without `beta`
                Ryy[k] = copy.deepcopy(RyyCurr[k])
                Rnn[k] = copy.deepcopy(RnnCurr[k])

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
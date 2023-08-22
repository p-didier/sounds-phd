# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import scipy.linalg as la
from .online_common import *

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
        vad=None
    ):

    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    nodeIndices = np.arange(nNodes)
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
    vadFramewise = np.full((nIter, nSensors), fill_value=None)
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
            # Get VAD for current frame
            if vad is not None:
                vadCurr = vad[idxBegFrame:idxEndFrame]
                # Convert to single boolean value (True if at least 50% of the frame is active)
                vadFramewise[i, m] = np.sum(vadCurr.astype(bool)) > p.nfft // 2
    # Convert to single boolean value
    vadFramewise = np.any(vadFramewise, axis=1)
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
        
    # Initialize fusion vectors
    fusionVectTargets = [None for _ in range(nNodes)]
    fusionVects = [None for _ in range(nNodes)]
    for k in range(nNodes):
        baseVect = np.ones((nPosFreqs, sum(channelToNodeMap == k)))
        # baseVect = np.zeros((nPosFreqs, sum(channelToNodeMap == k)))
        # baseVect[:, referenceSensorIdx] = 1  # select reference sensor
        fusionVectTargets[k] = baseVect
        fusionVects[k] = baseVect
    lastFuVectUp = -1 * np.ones(nNodes)

    Ryy = []
    Rnn = []
    RyyCurr = []
    RnnCurr = []
    for k in range(nNodes):
        RyyTmp = np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        )
        RnnTmp = np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        )
        for i in range(nIter):
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
                yk = yWola[i, :, channelToNodeMap == q]
                nk = nWola[i, :, channelToNodeMap == q]
                if ignoreFusionForSSNodes and yk.shape[1] == 1:
                    pass
                else:
                    fusedSignals[:, q] = np.einsum(
                        'ij,ij->i',
                        fusionVects[q].conj(),
                        yk.T,
                    )
                    fusedSignalsNoiseOnly[:, q] = np.einsum(
                        'ij,ij->i',
                        fusionVects[q].conj(),
                        nk.T,
                    )
            # Get y tilde
            yTilde = np.concatenate((
                yWola[i, :, channelToNodeMap == k].T,
                fusedSignals[:, nodeIndices != k]
            ), axis=1)
            nTilde = np.concatenate((
                nWola[i, :, channelToNodeMap == k].T,
                fusedSignalsNoiseOnly[:, nodeIndices != k]
            ), axis=1)
            # Update covariance matrices
            # RyyTmp = p.betaDanse * RyyTmp +\
            #     (1 - p.betaDanse) * 1 / dimYtilde[k] * np.einsum(
            #         'ij,ik->jk',
            #         yTilde,
            #         yTilde.conj()
            #     )
            # RnnTmp = p.betaDanse * RnnTmp +\
            #     (1 - p.betaDanse) * 1 / dimYtilde[k] * np.einsum(
            #         'ij,ik->jk',
            #         nTilde,
            #         nTilde.conj()
            #     )
            if i == 0:
                fact = 1
            else:
                fact = 1 / i
            RyyTmp = fact * ((i - 1) * RyyTmp + np.einsum(
                'ij,ik->jk',
                yTilde,
                yTilde.conj()
            ))
            RnnTmp = fact * ((i - 1) * RnnTmp + np.einsum(
                'ij,ik->jk',
                nTilde,
                nTilde.conj()
            ))
        # Ryy.append(RyyTmp)
        # Rnn.append(RnnTmp)

        Ryy.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        Rnn.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        RyyCurr.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        RnnCurr.append(np.zeros(
            (nPosFreqs, dimYtilde[k], dimYtilde[k]),
            dtype=np.complex128
        ))
        # Ryy.append(np.random.randn(
        #     nPosFreqs, dimYtilde[k], dimYtilde[k]
        # ))
        # Rnn.append(np.random.randn(
        #     nPosFreqs, dimYtilde[k], dimYtilde[k]
        # ))
    nUpdatesRyy = np.zeros(nNodes, dtype=int)
    nUpdatesRnn = np.zeros(nNodes, dtype=int)
    
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
                    fusionVects[k] = p.betaExt * fusionVects[k] +\
                        (1 - p.betaExt) * fusionVectTargets[k]
                else:
                    # Update without `beta_EXT`
                    fusionVects[k] = fusionVectTargets[k]

                # Check if target fusion vector is to be updated
                if i - lastFuVectUp[k] >= p.B:
                    fusionVectTargets[k] = (1 - p.alpha) * fusionVectTargets[k] +\
                        p.alpha * w[k][:, i, :yk.shape[1]]
                    lastFuVectUp[k] = i

            # Perform fusion
            fusedSignals[:, k] = np.einsum(
                'ij,ij->i',
                fusionVects[k].conj(),
                yk,
            )
            fusedSignalsNoiseOnly[:, k] = np.einsum(
                'ij,ij->i',
                fusionVects[k].conj(),
                nk,
            )

        # Loop over nodes to update filters
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
            
            # Update covariance matrices
            Ryy[k], Rnn[k], RyyCurr[k], RnnCurr[k],\
                updateFilter, nUpdatesRyy[k], nUpdatesRnn[k] = update_cov_mats(
                p,
                vadFramewise[i],
                yTilde,
                nTilde,
                nUpdatesRyy[k],
                nUpdatesRnn[k],
                i,
                Ryy[k],
                Rnn[k],
                RyyCurr[k],
                RnnCurr[k],
                wolaMode=True,
                danseMode=True,
                Mk=np.sum(channelToNodeMap == k),
            )

            # Check if filter should be updated according to the 
            # node updating strategy
            if updateFilter:
                if nodeUpdatingStrategy == 'sequential' and\
                    k == idxUpdatingNode:
                    pass  # updateFilter = True
                elif nodeUpdatingStrategy == 'simultaneous':
                    pass  # updateFilter = True
                else:
                    updateFilter = False

            if updateFilter:
                w[k][:, i + 1, :] = update_filter(
                    Ryy[k],
                    Rnn[k],
                    filterType=filterType,
                    rank=rank,
                    referenceSensorIdx=referenceSensorIdx
                )
                # for kppp in range(nNodes):
                #     print(w[kppp][:, i + 1, :])
            else:
                w[k][:, i + 1, :] = w[k][:, i, :]

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
                currNwIdx = channelToNodeMap[m]
                # Count channel index within node
                cIdx = channelCount[currNwIdx]
                if currNwIdx == k:  # current node channel
                    wNet[m, i + 1, :, k] = w[currNwIdx][:, i + 1, cIdx]
                else:  # neighbor node channel
                    gkq = w[k][:, i + 1, Mk + neighborCount]
                    wNet[m, i + 1, :, k] =\
                        fusionVects[currNwIdx][:, cIdx] * gkq
                    # wNet[m, i + 1, :, k] = fusionVects[currNwIdx][:, cIdx]
                    # wNet[m, i + 1, :, k] = gkq
                    # If we have reached the last channel of the current 
                    # neighbor node, increment neighbor count
                    if cIdx == np.sum(channelToNodeMap == currNwIdx) - 1:
                        neighborCount += 1
                channelCount[currNwIdx] += 1

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
    for k in range(nNodes):
        idxRef = np.where(channelToNodeMap == k)[0][referenceSensorIdx]
        wNet[idxRef, :, k] = 1  # initialize with identity matrix (selecting ref. sensor)
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
        else:
            vadCurr = None
        
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
                        fusionVects[k] = p.betaExt * fusionVects[k] +\
                            (1 - p.betaExt) * fusionVectTargets[k]
                    else:
                        # Update without `beta_EXT`
                        fusionVects[k] = fusionVectTargets[k]

                    # Check if target fusion vector is to be updated
                    if i - lastFuVectUp[k] >= p.B:
                        fusionVectTargets[k] = (1 - p.alpha) * fusionVectTargets[k] +\
                            p.alpha * w[k][:yk.shape[1], i]
                        lastFuVectUp[k] = i

                    # # Update target fusion vector
                    # if i > p.startFusionExpAvgAfter:
                    #     # Update using `beta_EXT`
                    #     fusionVectTargets[k] = p.betaExt * fusionVectTargets[k] +\
                    #         (1 - p.betaExt) * w[k][:yk.shape[1], i]
                    # else:
                    #     # Update without `beta_EXT`
                    #     fusionVectTargets[k] = w[k][:yk.shape[1], i]

                    # # Check if effective fusion vector is to be updated
                    # if i - lastFuVectUp[k] >= p.B:
                    #     fusionVects[k] = (1 - p.alpha) * fusionVects[k] +\
                    #         p.alpha * fusionVectTargets[k]
                    #     lastFuVectUp[k] = i
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
            
            # Update covariance matrices
            Ryy[k], Rnn[k], RyyCurr[k], RnnCurr[k],\
                updateFilter, nUpdatesRyy[k], nUpdatesRnn[k] = update_cov_mats(
                p,
                vadCurr,
                yTilde,
                nTilde,
                nUpdatesRyy[k],
                nUpdatesRnn[k],
                i,
                Ryy[k],
                Rnn[k],
                RyyCurr[k],
                RnnCurr[k],
                danseMode=True
            )

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
                    if verbose:
                        print(f'Rank-deficient Ryy[{k}]')
                    updateFilter = False
                # Check if Rnn is full rank
                if np.linalg.matrix_rank(Rnn[k]) < dimYtilde[k]:
                    if verbose:
                        print(f'Rank-deficient Rnn[{k}]')
                    updateFilter = False

            if updateFilter:
                w[k][:, i + 1] = update_filter(
                    Ryy[k],
                    Rnn[k],
                    filterType=filterType,
                    rank=rank,
                    referenceSensorIdx=referenceSensorIdx
                )
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
                currNwIdx = channelToNodeMap[m]
                # Channel index within node
                cIdx = channelCount[currNwIdx]
                if currNwIdx == k:  # current node channel
                    wNet[m, i + 1, k] = w[currNwIdx][cIdx, i + 1]
                else:  # neighbor node channel
                    gkq = w[k][Mk + neighborCount, i + 1]
                    wNet[m, i + 1, k] = fusionVects[currNwIdx][cIdx] * gkq
                    # wNet[m, i + 1, k] = gkq
                    # wNet[m, i + 1, k] = fusionVects[currNwIdx][cIdx]
                    # If we have reached the last channel of the current 
                    # neighbor node, increment neighbor count
                    if cIdx == np.sum(channelToNodeMap == currNwIdx) - 1:
                        neighborCount += 1
                channelCount[currNwIdx] += 1

    return wNet
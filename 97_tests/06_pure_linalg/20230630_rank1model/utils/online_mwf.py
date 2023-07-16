# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np

def run_online_mwf(
        x: np.ndarray,
        n: np.ndarray,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        L=1024,
        beta=0.99  # exponential averaging constant
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nSensors = len(channelToNodeMap)

    # Number of frames
    nIter = y.shape[0] // L

    # Initialize
    w = np.zeros((nSensors, nSensors, nIter), dtype=np.complex128)
    for m in range(nSensors):
        w[m, m, 0] = 1  # initialize with identity matrix (no filtering)
    Ryy = np.zeros((nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((nSensors, nSensors), dtype=np.complex128)
    
    # Loop over frames
    for i in range(nIter - 1):
        print(f'Online MWF iteration {i+1}/{nIter}')
        idxBegFrame = i * L
        idxEndFrame = (i + 1) * L
        # Get current frame
        yCurr = y[idxBegFrame:idxEndFrame, :]
        nCurr = n[idxBegFrame:idxEndFrame, :]

        # Compute covariance matrices
        RyyCurr = np.einsum('ij,ik->jk', yCurr, yCurr.conj())
        RnnCurr = np.einsum('ij,ik->jk', nCurr, nCurr.conj())
        # Update covariance matrices
        Ryy = beta * Ryy + (1 - beta) * RyyCurr
        Rnn = beta * Rnn + (1 - beta) * RnnCurr
        
        # Check if Ryy is full rank
        updateFilter = True
        if np.linalg.matrix_rank(Ryy) < nSensors:
            print('Rank-deficient Ryy')
            updateFilter = False
        # Check if Rnn is full rank
        if np.linalg.matrix_rank(Rnn) < nSensors:
            print('Rank-deficient Rnn')
            updateFilter = False
        
        # Loop over sensors
        if updateFilter:
            # Compute filter
            if filterType == 'regular':
                w[:, :, i + 1] = np.linalg.inv(Ryy) @ (Ryy - Rnn)
            elif filterType == 'gevd':
                raise NotImplementedError('not yet implemented for multiple frequency lines')
                sigma, Xmat = la.eigh(Ryy[m], Rnn[m])
                idx = np.flip(np.argsort(sigma))
                sigma = sigma[idx]
                Xmat = Xmat[:, idx]
                Qmat = np.linalg.inv(Xmat.T.conj())
                Dmat = np.zeros((Ryy[m].shape[0], Ryy[m].shape[0]))
                Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
                e = np.zeros(Ryy[m].shape[0])
                e[:rank] = 1
                w[m][:, i + 1] = Xmat @ Dmat @ Qmat.T.conj() @ e
        else:
            w[:, :, i + 1] = w[:, :, i]
    
    return w
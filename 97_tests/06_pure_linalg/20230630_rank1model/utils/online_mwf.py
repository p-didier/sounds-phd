# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import scipy.linalg as la

def run_online_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        L=1024,
        beta=0.99  # exponential averaging constant
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nSensors = x.shape[1]

    # Number of frames
    nIter = y.shape[0] // L

    # Initialize
    w = np.zeros((nSensors, nIter, nSensors), dtype=np.complex128)
    Ryy = np.zeros((nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((nSensors, nSensors), dtype=np.complex128)
    
    if filterType == 'gevd':
        algLabel = 'GEVD-MWF'
    elif filterType == 'regular':
        algLabel = 'MWF'
    # Loop over frames
    for i in range(nIter - 1):
        print(f'Online {algLabel} iteration {i+1}/{nIter}')
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

        # Compute filter
        if filterType == 'regular':
            for m in range(nSensors):
                e = np.zeros(nSensors)
                e[m] = 1
                w[:, i + 1, m] = np.linalg.inv(Ryy) @ (Ryy - Rnn) @ e
        elif filterType == 'gevd':
            sigma, Xmat = la.eigh(Ryy, Rnn)
            idx = np.flip(np.argsort(sigma))
            sigma = sigma[idx]
            Xmat = Xmat[:, idx]
            Qmat = np.linalg.inv(Xmat.T.conj())
            Dmat = np.zeros((nSensors, nSensors))
            Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
            myMat = Xmat @ Dmat @ Qmat.T.conj()
            for m in range(nSensors):
                e = np.zeros(nSensors)
                e[m] = 1
                w[:, i + 1, m] = myMat @ e
    
    return w
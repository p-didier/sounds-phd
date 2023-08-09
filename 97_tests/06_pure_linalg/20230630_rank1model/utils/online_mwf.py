# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import scipy.linalg as la
from .online_danse import get_window

def run_online_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        L=1024,
        beta=0.99,  # exponential averaging constant
        verbose=True
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
        if verbose:
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
                e[m] = 1  # selection vector for the ref. sensor index
                w[:, i + 1, m] = myMat @ e
    
    return w


def run_wola_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        L=1024,     # frame length
        R=512,      # hop size
        windowType='hann',
        beta=0.99,  # exponential averaging constant
        verbose=True
    ):
    """Weighted OverLap-Add (WOLA) MWF."""

    # Get number of nodes
    nSensors = x.shape[1]
    # Get number of frames
    nIter = x.shape[0] // R - 1
    # Get noisy signal (time-domain)
    y = x + n

    # Compute WOLA domain signal
    win = get_window(windowType, L)
    yWola = np.zeros((nIter, L, nSensors), dtype=np.complex128)
    nWola = np.zeros((nIter, L, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        for i in range(nIter):
            idxBegFrame = i * R
            idxEndFrame = idxBegFrame + L
            yWola[i, :, m] = np.fft.fft(
                y[idxBegFrame:idxEndFrame, m] * win
            ) / np.sqrt(L)
            nWola[i, :, m] = np.fft.fft(
                n[idxBegFrame:idxEndFrame, m] * win
            ) / np.sqrt(L)

    # Initialize
    w = np.zeros((nSensors, nIter, L, nSensors), dtype=np.complex128)
    Ryy = np.zeros((L, nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((L, nSensors, nSensors), dtype=np.complex128)
    
    if filterType == 'gevd':
        algLabel = 'GEVD-MWF'
    elif filterType == 'regular':
        algLabel = 'MWF'

    # Loop over frames
    for i in range(nIter - 1):
        if verbose:
            print(f'WOLA-based {algLabel} iteration {i+1}/{nIter}')

        # Get current frame
        yCurr = yWola[i, :, :]
        nCurr = nWola[i, :, :]
        # Compute covariance matrices
        RyyCurr = np.einsum('ij,ik->ijk', yCurr, yCurr.conj())
        RnnCurr = np.einsum('ij,ik->ijk', nCurr, nCurr.conj())
        # Update covariance matrices
        Ryy = beta * Ryy + (1 - beta) * RyyCurr
        Rnn = beta * Rnn + (1 - beta) * RnnCurr
        # Compute filter
        if np.linalg.matrix_rank(Ryy[0, :, :]) >= nSensors and\
            np.linalg.matrix_rank(Rnn[0, :, :]) >= nSensors:

            if filterType == 'regular':
                currw = np.linalg.inv(Ryy) @ (Ryy - Rnn)
                w[:, i + 1, :, :] = np.transpose(currw, (2, 0, 1))

            elif filterType == 'gevd':
                Xmat = np.zeros((L, nSensors, nSensors), dtype=np.complex128)
                sigma = np.zeros((L, nSensors))
                # Looping over frequencies because of the GEVD
                for kappa in range(L):
                    sigmaCurr, XmatCurr = la.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :])
                    indices = np.flip(np.argsort(sigmaCurr))
                    sigma[kappa, :] = sigmaCurr[indices]
                    Xmat[kappa, :, :] = XmatCurr[:, indices]
                Qmat = np.linalg.inv(np.transpose(Xmat.conj(), axes=[0, 2, 1]))
                # GEVLs tensor
                Dmat = np.zeros((L, nSensors, nSensors))
                for r in range(rank):
                    Dmat[:, r, r] = np.squeeze(1 - 1 / sigma[:, r])
                # LMMSE weights
                Qhermitian = np.transpose(Qmat.conj(), axes=[0, 2, 1])
                wCurr = np.matmul(np.matmul(Xmat, Dmat), Qhermitian)
                w[:, i + 1, :, :] = np.transpose(wCurr, (2, 0, 1))
                
        else:
            print(f'i = {i}: rank deficient covariance matrix/matrices')
            w[:, i + 1, :, :] = w[:, i, :, :]
        
    return w
# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import scipy.linalg as la
from .online_danse import get_window, WOLAparameters

def run_online_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        p: WOLAparameters=WOLAparameters(),
        verbose=True
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nSensors = x.shape[1]

    # Number of frames
    nIter = y.shape[0] // p.nfft

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
        idxBegFrame = i * p.nfft
        idxEndFrame = (i + 1) * p.nfft
        # Get current frame
        yCurr: np.ndarray = y[idxBegFrame:idxEndFrame, :]
        nCurr: np.ndarray = n[idxBegFrame:idxEndFrame, :]

        # Compute covariance matrices
        RyyCurr = np.einsum('ij,ik->jk', yCurr, yCurr.conj())
        RnnCurr = np.einsum('ij,ik->jk', nCurr, nCurr.conj())
        # Update covariance matrices
        if i > p.startExpAvgAfter:
            Ryy = p.betaMwf * Ryy + (1 - p.betaMwf) * RyyCurr
            Rnn = p.betaMwf * Rnn + (1 - p.betaMwf) * RnnCurr
        else:
            Ryy = copy.deepcopy(RyyCurr)
            Rnn = copy.deepcopy(RnnCurr)

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
        p: WOLAparameters=WOLAparameters(),
        verbose=True
    ):
    """Weighted OverLap-Add (WOLA) MWF."""

    # Get number of nodes
    nSensors = x.shape[1]
    # Get number of frames
    nIter = x.shape[0] // p.hop - 1
    # Get noisy signal (time-domain)
    y = x + n

    # Compute WOLA domain signal
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
            print(f'/!\ /!\ /!\ WOLA-MWF: single frequency bin study (index {p.singleFreqBinIndex})')
        yWola = yWola[:, p.singleFreqBinIndex, :]
        nWola = nWola[:, p.singleFreqBinIndex, :]
        yWola = np.expand_dims(yWola, axis=1)  # add singleton dimension
        nWola = np.expand_dims(nWola, axis=1)
        nPosFreqs = 1
    
    # Initialize
    w = np.zeros((nSensors, nIter, nPosFreqs, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        w[m, :, :, m] = 1  # initialize with identity matrix (selecting ref. sensor)
    Ryy = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    
    if filterType == 'gevd':
        algLabel = 'GEVD-MWF'
    elif filterType == 'regular':
        algLabel = 'MWF'

    # Loop over frames
    for i in range(nIter - 1):
        if verbose:
            print(f'WOLA-based {algLabel} iteration {i+1}/{nIter}')

        # Get current frame
        yCurr: np.ndarray = yWola[i, :, :]
        nCurr: np.ndarray = nWola[i, :, :]
        # Compute covariance matrices
        RyyCurr = np.einsum('ij,ik->ijk', yCurr, yCurr.conj())
        RnnCurr = np.einsum('ij,ik->ijk', nCurr, nCurr.conj())
        # Update covariance matrices
        if i > p.startExpAvgAfter:
            Ryy = p.betaMwf * Ryy + (1 - p.betaMwf) * RyyCurr
            Rnn = p.betaMwf * Rnn + (1 - p.betaMwf) * RnnCurr
        else:
            Ryy = copy.deepcopy(RyyCurr)
            Rnn = copy.deepcopy(RnnCurr)
            
        # Check if SCMs are full rank
        updateFilter = True
        if np.any(np.linalg.matrix_rank(Ryy) < nSensors):
            if verbose:
                print('Rank-deficient Ryy')
            updateFilter = False
        if np.any(np.linalg.matrix_rank(Rnn) < nSensors):
            if verbose:
                print('Rank-deficient Rnn')
            updateFilter = False
        
        # Compute filter
        if updateFilter:
            if filterType == 'regular':
                currw = np.linalg.inv(Ryy) @ (Ryy - Rnn)
                for kk in range(nSensors):
                    w[:, i + 1, :, kk] = currw[:, :, kk].T

            elif filterType == 'gevd':
                Xmat = np.zeros(
                    (nPosFreqs, nSensors, nSensors),
                    dtype=np.complex128
                )
                sigma = np.zeros((nPosFreqs, nSensors))
                # Looping over frequencies because of the GEVD
                for kappa in range(nPosFreqs):
                    sigmaCurr, XmatCurr = la.eigh(
                        Ryy[kappa, :, :],
                        Rnn[kappa, :, :]
                    )
                    indices = np.flip(np.argsort(sigmaCurr))
                    sigma[kappa, :] = sigmaCurr[indices]
                    Xmat[kappa, :, :] = XmatCurr[:, indices]
                Qmat = np.linalg.inv(
                    np.transpose(Xmat.conj(), axes=[0, 2, 1])
                )
                # GEVLs tensor
                Dmat = np.zeros((nPosFreqs, nSensors, nSensors))
                for r in range(rank):
                    Dmat[:, r, r] = np.squeeze(1 - 1 / sigma[:, r])
                # LMMSE weights
                Qhermitian = np.transpose(Qmat.conj(), axes=[0, 2, 1])
                wCurr = np.matmul(np.matmul(Xmat, Dmat), Qhermitian)
                w[:, i + 1, :, :] = np.transpose(wCurr, (2, 0, 1))
        else:
            w[:, i + 1, :, :] = w[:, i, :, :]

    return w
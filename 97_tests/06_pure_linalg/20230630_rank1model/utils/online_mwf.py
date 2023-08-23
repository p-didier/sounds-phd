# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import numpy as np
import scipy.linalg as la
from .online_danse import get_window, WOLAparameters
from .online_common import *

def run_online_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        p: WOLAparameters=WOLAparameters(),
        verbose=True,
        vad=None
    ):

    # Get noisy signal (time-domain)
    y = x + n
    # Get number of nodes
    nSensors = x.shape[1]

    # Number of frames
    nIter = y.shape[0] // p.nfft

    # Initialize
    w = np.zeros((nSensors, nIter, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        w[m, :, m] = 1  # initialize with identity matrix (selecting ref. sensor)
    Ryy = np.zeros((nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((nSensors, nSensors), dtype=np.complex128)
    RyyCurr = np.zeros((nSensors, nSensors), dtype=np.complex128)
    RnnCurr = np.zeros((nSensors, nSensors), dtype=np.complex128)
    nUpdatesRyy = 0
    nUpdatesRnn = 0
    
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

        # Get VAD for current frame
        if vad is not None:
            vadCurr = vad[idxBegFrame:idxEndFrame]
            # Convert to single boolean value
            vadCurr = np.any(vadCurr.astype(bool))
        else:
            vadCurr = None

        # Update covariance matrices
        Ryy, Rnn, RyyCurr, RnnCurr,\
            updateFilter, nUpdatesRyy, nUpdatesRnn = update_cov_mats(
            p,
            vadCurr,
            yCurr,
            nCurr,
            nUpdatesRyy,
            nUpdatesRnn,
            i,
            Ryy,
            Rnn,
            RyyCurr,
            RnnCurr
        )

        if updateFilter:
            w[:, i + 1, :] = update_filter(
                Ryy,
                Rnn,
                filterType=filterType,
                rank=rank
            )
        else:
            w[:, i + 1, :] = w[:, i, :]
    
    return w


def run_wola_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        p: WOLAparameters=WOLAparameters(),
        verbose=True,
        vad=None
    ):
    """Weighted OverLap-Add (WOLA) MWF."""

    # Get number of nodes
    nSensors = x.shape[1]
    # Get number of frames
    nIter = x.shape[0] // p.hop - 1
    # Get noisy signal (time-domain)
    y = x + n

    # Compute WOLA domain signal
    yWola, nWola, vadFramewise = to_wola(p, y, n, vad, verbose)
    nPosFreqs = yWola.shape[1]
    
    # Initialize
    w = np.zeros((nSensors, nIter, nPosFreqs, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        w[m, :, :, m] = 1  # initialize with identity matrix (selecting ref. sensor)
    Ryy = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    Rnn = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    RyyCurr = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    RnnCurr = np.zeros((nPosFreqs, nSensors, nSensors), dtype=np.complex128)
    nUpdatesRyy = 0
    nUpdatesRnn = 0
    
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

        Ryy, Rnn, RyyCurr, RnnCurr,\
            updateFilter, nUpdatesRyy, nUpdatesRnn = update_cov_mats(
            p,
            vadFramewise[i],
            yCurr,
            nCurr,
            nUpdatesRyy,
            nUpdatesRnn,
            i,
            Ryy,
            Rnn,
            RyyCurr,
            RnnCurr,
            wolaMode=True
        )

        # Compute filter
        if updateFilter:
            w[:, i + 1, :, :] = update_filter(
                Ryy,
                Rnn,
                filterType=filterType,
                rank=rank
            )
        else:
            w[:, i + 1, :, :] = w[:, i, :, :]

    return w
import copy
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass


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
    startExpAvgAfter: int = -1  # idx of frame after which to start exponential averaging
    startFusionExpAvgAfter: int = -1  # same as above, for the fusion vectors
    #
    singleFreqBinIndex: int = None  # if not None, consider only the freq. bin at this index for WOLA-DANSE
    
    def __post_init__(self):
        if self.singleFreqBinIndex is not None:
            if self.singleFreqBinIndex > self.nPosFreqs:
                raise ValueError('singleFreqBinIndex cannot be larger than nPosFreqs')


def update_cov_mats(
        p: WOLAparameters,
        vadCurr,
        yCurr,
        nCurr,
        nUpdatesRyy,
        nUpdatesRnn,
        i,
        Ryy,
        Rnn,
        RyyCurr,
        RnnCurr,
        wolaMode=False,
        danseMode=False,
        verbose=False,
        Mk=1,
        skipExpAvg=False,
    ):

    # # Check rank of Ryy and Rnn
    # ryyFR = np.all(np.linalg.matrix_rank(Ryy) == Ryy.shape[-1])
    # rnnFR = np.all(np.linalg.matrix_rank(Rnn) == Rnn.shape[-1])
    # if not ryyFR and rnnFR:
    #     # If Rnn is full rank but Ryy is not, update Ryy only
    #     updateRyy, updateRnn = True, False
    # elif ryyFR and not rnnFR:
    #     # If Ryy is full rank but Rnn is not, update Rnn only
    #     updateRyy, updateRnn = False, True
    # else:
    #     # If both are full rank, update both
    #     updateRyy, updateRnn = True, True
    # updateRyy, updateRnn = True, True

    # Initialize
    einSumOp = 'ij,ik->jk'
    normFact = p.nPosFreqs
    if wolaMode:
        einSumOp = 'ij,ik->ijk'
        normFact = yCurr.shape[1]
    beta = p.betaMwf
    # In DANSE, update the top-left Mk x Mk block of Ryy with betaDanse,
    # the rest with betaExt
    if danseMode:  
        # beta = np.full((Ryy.shape[-1], Ryy.shape[-1]), fill_value=p.betaExt)
        # beta[:Mk, :Mk] = p.betaDanse
        beta = p.betaDanse
    
    # Update covariance matrices
    if vadCurr is not None:
        # Compute covariance matrices following VAD
        if vadCurr:
            RyyCurr = 1 / normFact * np.einsum(einSumOp, yCurr, yCurr.conj())
            nUpdatesRyy += 1
        else:
            RnnCurr = 1 / normFact * np.einsum(einSumOp, yCurr, yCurr.conj())
            nUpdatesRnn += 1
        updateFilter = nUpdatesRnn > 0 and nUpdatesRyy > 0
        # Condition to start exponential averaging
        startExpAvgCondRyy = nUpdatesRyy > 0
        startExpAvgCondRnn = nUpdatesRnn > 0
    else:
        # Compute covariance matrices using oracle noise knowledge
        RyyCurr = 1 / normFact * np.einsum(einSumOp, yCurr, yCurr.conj())
        RnnCurr = 1 / normFact * np.einsum(einSumOp, nCurr, nCurr.conj())
        nUpdatesRyy += 1
        nUpdatesRnn += 1
        updateFilter = True
        # Condition to start exponential averaging
        startExpAvgCondRyy = i > p.startExpAvgAfter
        startExpAvgCondRnn = i > p.startExpAvgAfter
    
    # if updateRyy:
    if startExpAvgCondRyy and not skipExpAvg:
        Ryy = beta * Ryy + (1 - beta) * RyyCurr
    else:
        if verbose:
            print(f'i={i}, not yet starting exponential averaging for Ryy')
        Ryy = copy.deepcopy(RyyCurr)
    # elif verbose:
    #     print(f'i={i}, not updating Ryy (already full rank while Rnn is not)')
    
    # if updateRnn:
    if startExpAvgCondRnn and not skipExpAvg:
        Rnn = beta * Rnn + (1 - beta) * RnnCurr
    else:
        if verbose:
            print(f'i={i}, not yet starting exponential averaging for Rnn')
        Rnn = copy.deepcopy(RnnCurr)
    # elif verbose:
    #     print(f'i={i}, not updating Rnn (already full rank while Ryy is not)')

    if updateFilter:  # Check rank of updated covariance matrices
        if np.any(np.linalg.matrix_rank(Ryy) < Ryy.shape[-1]) or\
            np.any(np.linalg.matrix_rank(Rnn) < Rnn.shape[-1]):
            updateFilter = False
            if verbose:
                print(f'i={i}, not updating filter (rank deficient covariance matrices)')

    return Ryy, Rnn, RyyCurr, RnnCurr, updateFilter, nUpdatesRyy, nUpdatesRnn


def update_filter(
        Ryy,
        Rnn,
        filterType='regular',
        rank=1,
        referenceSensorIdx=None,
    ):
    dim = Ryy.shape[-1]
    wolaMode = len(Ryy.shape) == 3  # True if WOLA mode on
    # Compute filter
    if filterType == 'regular':
        bigW = np.linalg.inv(Ryy) @ (Ryy - Rnn)
        if wolaMode:
            if referenceSensorIdx is None:
                # return np.transpose(bigW, (2, 0, 1))
                return np.swapaxes(bigW, 0, 1)
            else:
                return bigW[:, :, referenceSensorIdx]
                # return bigW[:, referenceSensorIdx, :]
        else:
            if referenceSensorIdx is None:
                return bigW
            else:
                return bigW[:, referenceSensorIdx]
                # return bigW[referenceSensorIdx , :]
    elif filterType == 'gevd':
        if wolaMode:
            nBins = Ryy.shape[0]
            Xmat = np.zeros((nBins, dim, dim), dtype=np.complex128)
            sigma = np.zeros((nBins, dim))
            # Looping over frequencies because of the GEVD
            for kappa in range(nBins):
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
            Dmat = np.zeros((nBins, dim, dim))
            for r in range(rank):
                Dmat[:, r, r] = np.squeeze(1 - 1 / sigma[:, r])
            # LMMSE weights
            Qhermitian = np.transpose(Qmat.conj(), axes=[0, 2, 1])
            bigW = np.matmul(np.matmul(Xmat, Dmat), Qhermitian)
            if referenceSensorIdx is None:
                # return np.transpose(bigW, (2, 0, 1))
                return np.swapaxes(bigW, 0, 1)
            else:
                return bigW[:, :, referenceSensorIdx]
                # return bigW[:, referenceSensorIdx, :]
        else:
            sigma, Xmat = la.eigh(Ryy, Rnn)
            idx = np.flip(np.argsort(sigma))
            sigma = sigma[idx]
            Xmat = Xmat[:, idx]
            Qmat = np.linalg.inv(Xmat.T.conj())
            Dmat = np.zeros((dim, dim))
            Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
            if referenceSensorIdx is None:
                return Xmat @ Dmat @ Qmat.T.conj()
            else:
                w = Xmat @ Dmat @ Qmat.T.conj()
                return w[:, referenceSensorIdx]
                # return w[referenceSensorIdx, :]


def to_wola(
        p: WOLAparameters,
        sig: np.ndarray,
        vad: np.ndarray=None,
        verbose=False,
    ):
    """Converts time-domain signal to WOLA domain signal."""
    # Get number of nodes
    nSensors = sig.shape[1]
    # Get number of frames
    nIter = sig.shape[0] // p.hop - 1
    # Compute WOLA domain signal
    win = get_window(p.winType, p.nfft)
    sigWola = np.zeros((nIter, p.nfft, nSensors), dtype=np.complex128)
    vadFramewise = np.full((nIter, nSensors), fill_value=None)
    for m in range(nSensors):
        for i in range(nIter):
            idxBegFrame = i * p.hop
            idxEndFrame = idxBegFrame + p.nfft
            sigWola[i, :, m] = np.fft.fft(
                sig[idxBegFrame:idxEndFrame, m] * win
            ) / np.sqrt(p.hop)
            # Get VAD for current frame
            if vad is not None:
                vadCurr = vad[idxBegFrame:idxEndFrame]
                # Convert to single boolean value
                # (True if at least 50% of the frame is active)
                vadFramewise[i, m] =\
                    np.sum(vadCurr.astype(bool)) > p.nfft // 2
    # Convert to single boolean value
    vadFramewise = np.any(vadFramewise, axis=1)
    # Get number of positive frequencies
    nPosFreqs = p.nfft // 2 + 1
    # Keep only positive frequencies (spare computations)
    sigWola = sigWola[:, :nPosFreqs, :]

    # Special user request: single frequency bin study
    if p.singleFreqBinIndex is not None:
        if verbose:
            print(f'/!\ /!\ /!\ WOLA-MWF: single frequency bin study (index {p.singleFreqBinIndex})')
        sigWola = sigWola[:, p.singleFreqBinIndex, :]
        sigWola = np.expand_dims(sigWola, axis=1)  # add singleton dimension
        nPosFreqs = 1

    return sigWola, vadFramewise


def from_wola(
        p: WOLAparameters,
        sigWola: np.ndarray,
    ):
    """Converts WOLA domain signal to time-domain signal."""

    if len(sigWola.shape) == 2:
        sigWola = sigWola[:, :, np.newaxis]  # single-sensor case

    # Get number of nodes
    nSensors = sigWola.shape[-1]
    # Get number of frames
    nIter = sigWola.shape[0]
    # Compute time-domain signal
    win = get_window(p.winType, p.nfft)
    sig = np.zeros((nIter * p.hop + p.nfft, nSensors), dtype=np.complex128)
    for m in range(nSensors):
        for i in range(nIter):
            idxBegFrame = i * p.hop
            idxEndFrame = idxBegFrame + p.nfft
            # Re-include negative frequencies
            currSigFrame = np.concatenate((
                sigWola[i, :, m],
                np.flip(sigWola[i, 1:-1, m].conj(), axis=0)
            ))

            sig[idxBegFrame:idxEndFrame, m] += np.fft.ifft(
                currSigFrame * np.sqrt(p.hop)
            ) * win
    # Remove zero-padding
    # sig = sig[p.nfft:-p.nfft, :]
    
    sig = np.real_if_close(sig, tol=1000)

    return sig

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
        danseMode=False
    ):
    # Initialize
    einSumOp = 'ij,ik->jk'
    if wolaMode:
        einSumOp = 'ij,ik->ijk'
    beta = p.betaMwf
    if danseMode:
        beta = p.betaDanse
    
    # Update covariance matrices
    if vadCurr is not None:
        # Compute covariance matrices following VAD
        if vadCurr:
            RyyCurr = 1 / p.nfft * np.einsum(einSumOp, yCurr, yCurr.conj())
            nUpdatesRyy += 1
        else:
            RnnCurr = 1 / p.nfft * np.einsum(einSumOp, yCurr, yCurr.conj())
            nUpdatesRnn += 1
        updateFilter = nUpdatesRnn > 0 and nUpdatesRyy > 0
        # Condition to start exponential averaging
        startExpAvgCondRyy = nUpdatesRyy > 0
        startExpAvgCondRnn = nUpdatesRnn > 0
    else:
        # Compute covariance matrices using oracle noise knowledge
        RyyCurr = 1 / p.nfft * np.einsum(einSumOp, yCurr, yCurr.conj())
        RnnCurr = 1 / p.nfft * np.einsum(einSumOp, nCurr, nCurr.conj())
        updateFilter = True
        # Condition to start exponential averaging
        startExpAvgCondRyy = i > p.startExpAvgAfter
        startExpAvgCondRnn = i > p.startExpAvgAfter
    
    if startExpAvgCondRyy:
        Ryy = beta * Ryy + (1 - beta) * RyyCurr
    else:
        print(f'i={i}, not yet starting exponential averaging for Ryy')
        Ryy = copy.deepcopy(RyyCurr)
    
    if startExpAvgCondRnn:
        Rnn = beta * Rnn + (1 - beta) * RnnCurr
    else:
        print(f'i={i}, not yet starting exponential averaging for Rnn')
        Rnn = copy.deepcopy(RnnCurr)

    return Ryy, Rnn, RyyCurr, RnnCurr, updateFilter, nUpdatesRyy, nUpdatesRnn


def update_filter(
        Ryy,
        Rnn,
        filterType='regular',
        rank=1,
        referenceSensorIdx=None,
    ):
    n = Ryy.shape[-1]
    wolaMode = len(Ryy.shape) == 3  # True if WOLA mode on
    # Compute filter
    if filterType == 'regular':
        bigW = np.linalg.inv(Ryy) @ (Ryy - Rnn)
        if wolaMode:
            if referenceSensorIdx is None:
                return np.swapaxes(bigW, 0, 1)
            else:
                return bigW[:, :, referenceSensorIdx]
        else:
            if referenceSensorIdx is None:
                return bigW
            else:
                return bigW[:, referenceSensorIdx]
    elif filterType == 'gevd':
        if wolaMode:
            nBins = Ryy.shape[0]
            Xmat = np.zeros((nBins, n, n), dtype=np.complex128)
            sigma = np.zeros((nBins, n))
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
            Dmat = np.zeros((nBins, n, n))
            for r in range(rank):
                Dmat[:, r, r] = np.squeeze(1 - 1 / sigma[:, r])
            # LMMSE weights
            Qhermitian = np.transpose(Qmat.conj(), axes=[0, 2, 1])
            bigW = np.matmul(np.matmul(Xmat, Dmat), Qhermitian)
            if referenceSensorIdx is None:
                return np.transpose(bigW, (2, 0, 1))
            else:
                return bigW[:, :, referenceSensorIdx]
        else:
            sigma, Xmat = la.eigh(Ryy, Rnn)
            idx = np.flip(np.argsort(sigma))
            sigma = sigma[idx]
            Xmat = Xmat[:, idx]
            Qmat = np.linalg.inv(Xmat.T.conj())
            Dmat = np.zeros((n, n))
            Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
            if referenceSensorIdx is None:
                return Xmat @ Dmat @ Qmat.T.conj()
            else:
                w = Xmat @ Dmat @ Qmat.T.conj()
                return w[:, referenceSensorIdx]

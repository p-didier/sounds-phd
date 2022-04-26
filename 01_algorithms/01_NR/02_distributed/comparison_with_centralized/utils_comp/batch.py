
import sys, time
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("01_algorithms/01_NR/02_distributed" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_utilities.classes import Signals, AcousticScenario, ProgramSettings
from danse_utilities import danse_subfcns as subs
from danse_utilities.setup import prep_for_ffts


def gevd_mwf_batch(signals: Signals, asc: AcousticScenario, settings: ProgramSettings, plotit=False):
    """Batch-mode GEVD-MWF.
    
    Parameters
    ----------
    signals : Signals object
        The microphone signals and their relevant attributes.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.


    Returns
    -------
    DfiltGEVD
    Dfilt
    """

    y = signals.sensorSignals

    # Get STFTs
    for k in range(y.shape[-1]):
        f, t, out = sig.stft(y[:, k],
                            fs=asc.samplingFreq,
                            window=settings.stftWin,
                            nperseg=settings.stftWinLength,
                            noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
                            return_onesided=True)
        if k == 0:
            ystft = np.zeros((out.shape[0], out.shape[1], y.shape[-1]), dtype=complex)
        ystft[:, :, k] = out

    # Get VAD per frame
    nFrames = len(t)
    oVADframes = np.zeros(nFrames)
    for l in range(nFrames):
        idxBegChunk = int(l * settings.stftWinLength * (1 - settings.stftFrameOvlp))
        idxEndChunk = int(idxBegChunk + settings.stftWinLength)
        VADinFrame = signals.VAD[idxBegChunk:idxEndChunk]
        oVADframes[l] = sum(VADinFrame == 0) <= settings.stftWinLength / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

    # Batch-mode spatial autocorrelation matrices
    w = np.zeros((len(f), y.shape[-1]), dtype=complex)
    Dfilt = np.zeros((len(f), len(t)), dtype=complex)
    DfiltGEVD = np.zeros((len(f), len(t)), dtype=complex)
    for kappa in range(len(f)):
        Ryy = np.mean(np.einsum('ij,ik->ijk', ystft[kappa, oVADframes==1, :], ystft[kappa, oVADframes==1, :].conj()), axis=0)
        Rnn = np.mean(np.einsum('ij,ik->ijk', ystft[kappa, oVADframes==0, :], ystft[kappa, oVADframes==0, :].conj()), axis=0)

        # NO GEVD
        ed = np.zeros(Ryy.shape[-1])
        ed[0] = 1
        w[kappa, :] = np.linalg.pinv(Ryy) @ (Ryy - Rnn) @ ed
        Dfilt[kappa,:] = ystft[kappa,:,:] @ w[kappa, :].conj()

        # GEVD
        w[kappa, :], _ = subs.perform_gevd(Ryy, Rnn, rank=settings.GEVDrank, refSensorIdx=0)

        # # GEVD
        # sigma, Xmat = scipy.linalg.eigh(Ryy, Rnn)
        # # Flip Xmat to sort eigenvalues in descending order
        # idx = np.flip(np.argsort(sigma))
        # sigma = sigma[idx]
        # Xmat = Xmat[:, idx]

        # # Reference sensor selection vector 
        # Evect = np.zeros((Ryy.shape[-1],))
        # Evect[settings.referenceSensor] = 1

        # Qmat = np.linalg.inv(Xmat.conj().T)
        # # Sort eigenvalues in descending order
        # idx = np.flip(np.argsort(sigma))
        # GEVLs_yy = np.flip(np.sort(sigma))
        # Qmat = Qmat[:, idx]
        # diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:settings.GEVDrank]])   # rank <GEVDrank> approximation
        # diagveig = np.append(diagveig, np.zeros(Ryy.shape[0] - settings.GEVDrank))
        # # LMMSE weights
        # w[kappa, :] = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect

        # Apply filter
        DfiltGEVD[kappa,:] = ystft[kappa,:,:] @ w[kappa, :].conj()

    if plotit:
        # Get colorbar limits
        climHigh = np.amax(np.concatenate((20*np.log10(np.abs(ystft[:,:,0])), 20*np.log10(np.abs(Dfilt)), 20*np.log10(np.abs(DfiltGEVD))), axis=0))
        climLow = np.amin(np.concatenate((20*np.log10(np.abs(ystft[:,:,0])), 20*np.log10(np.abs(Dfilt)), 20*np.log10(np.abs(DfiltGEVD))), axis=0))

        # Plot
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(311)
        mapp = plt.imshow(20*np.log10(np.abs(ystft[:,:,0])), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title('Original reference mic signal')
        ax = fig.add_subplot(312)
        mapp = plt.imshow(20*np.log10(np.abs(Dfilt)), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title('After batch-MWF')
        ax = fig.add_subplot(313)
        mapp = plt.imshow(20*np.log10(np.abs(DfiltGEVD)), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'After batch-GEVD-MWF (rank {settings.GEVDrank})')
        plt.tight_layout()	
        plt.show()

        stop = 1

    return DfiltGEVD, Dfilt

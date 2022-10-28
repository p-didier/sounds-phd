import sys
import soundfile as sf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("danse_utilities" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed/danse_utilities')
from sro_subfcns import *


@dataclass
class Settings():
    filePath : str
    sro : float  # [ppm]
    dftSize : int = 1024
    stftWin : np.ndarray = np.hanning(dftSize)
    stftOvlp : float = 0.5
    stftWinShift : int = int(np.floor(dftSize * (1 - stftOvlp)))
    ld : int = 10
    alpha : float = 0.95


def main():

    s = Settings(
        filePath='02_data/00_raw_signals/01_speech/speech1.wav',
        sro = 500, # [ppm]
    )

    # Fetch raw signal
    x, fs = sf.read(s.filePath)
    x = x[:, np.newaxis]
    # Apply SRO
    x_sro, _, _ = apply_sro_sto(x, fs, [1], [s.sro], STOdelays=[0])

    numChunks = int(len(x) // s.stftWinShift) - 1
    Xcurr = np.zeros((s.dftSize, numChunks), dtype=complex)
    X_srocurr = np.zeros((s.dftSize, numChunks), dtype=complex)
    apr = np.zeros(s.dftSize, dtype=complex)
    sro_est = np.zeros(numChunks)
    flagFirstEstimate = True
    
    for ii in range(numChunks):

        idxBeg = ii * s.stftWinShift
        idxEnd = idxBeg + s.dftSize
        xcurr = x[idxBeg:idxEnd]
        Xcurr[:, ii] = np.squeeze(np.fft.fft(xcurr * s.stftWin[:, np.newaxis], s.dftSize, axis=0))
        x_srocurr = x_sro[idxBeg:idxEnd]
        X_srocurr[:, ii] = np.squeeze(np.fft.fft(x_srocurr * s.stftWin[:, np.newaxis], s.dftSize, axis=0))

        if ii >= s.ld:

            cohCurr = (Xcurr[:, ii] * np.conj(X_srocurr[:, ii])) \
                / np.sqrt(np.abs(Xcurr[:, ii])**2 * np.abs(X_srocurr[:, ii])**2)
                
            cohPast = (Xcurr[:, ii - s.ld] * np.conj(X_srocurr[:, ii - s.ld])) \
                / np.sqrt(np.abs(Xcurr[:, ii - s.ld])**2 * np.abs(X_srocurr[:, ii - s.ld])**2)

            sro_est[ii], apr = cohdrift_sro_estimation(
                                wPos=cohCurr,
                                wPri=cohPast,
                                avgResProd=apr,
                                Ns=s.stftWinShift,
                                ld=s.ld,
                                alpha=s.alpha,
                                method='gs',
                                flagFirstSROEstimate=flagFirstEstimate
                            )

            if flagFirstEstimate:
                flagFirstEstimate = False

    # Plot
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(sro_est * 1e6)
    axes.hlines(y=s.sro, xmin=0, xmax=len(sro_est), colors='k')
    axes.grid()
    plt.tight_layout()	
    plt.show()

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
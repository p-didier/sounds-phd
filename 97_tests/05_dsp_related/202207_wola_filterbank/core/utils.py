import numpy as np
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass

@dataclass
class Params:
    T : int
    Nh : int
    R : int
    seed : int = 12345

def wola_fd_broadcast(x, Nh, R, w):
    """
    Parameters
    ----------
    x : np.ndarray
        Input signal (1-dimensional).
    Nh : int
        Window length. 
    R : int
        Window shift.
    w : np.ndarray
        Filter coefficients. 
    """

    h = np.sqrt(np.hanning(Nh))
    f = copy.copy(h)    # <-- synthesis window is the same as the analysis window

    nframes = len(x) // R 
    
    z = np.zeros((Nh, nframes), dtype=complex)
    ii = 0
    while True:
        idxBeg = ii*R
        idxEnd = idxBeg + Nh
        if idxEnd > len(x):
            break
        xcurr = copy.copy(x[idxBeg:idxEnd])

        xcurr *= h  # synthesis window

        Xcurr = np.fft.fft(xcurr, Nh, axis=0)

        # spectral modifications
        # Ycurr = copy.copy(Xcurr)   # <-- no modifications
        z[:, ii] = Xcurr * w

        # ycurr = np.fft.ifft(Ycurr, Nh, axis=0)
        # ycurr = np.real_if_close(ycurr)
        # ycurr *= f  # analysis window

        # # y[idxBeg:idxEnd] += ycurr
        # z[:, ii] = ycurr

        ii += 1

    return z


def plotit(x, y, w):

    fig, axes = plt.subplots(3,1)
    fig.set_size_inches(12.5, 3.5)
    axes[0].plot(x, label='x')
    axes[0].plot(y, label='y')
    axes[0].legend()
    axes[0].grid()
    # Filter coefficients
    axes[1].plot(np.real(w), 'k', label='Re{w}')
    axes[1].plot(np.imag(w), 'r', label='Im{w}')
    axes[1].legend()
    axes[1].grid()
    # Spectra
    axes[2].plot(20*np.log10(np.abs(np.fft.fft(x))), label='X')
    axes[2].plot(20*np.log10(np.abs(np.fft.fft(y))), label='Y')
    axes[2].legend()
    axes[2].grid()
    plt.tight_layout()	
    plt.show()
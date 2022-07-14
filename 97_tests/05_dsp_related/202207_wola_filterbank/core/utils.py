import numpy as np
import scipy.signal as sig
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
    # f = copy.copy(h)    # <-- synthesis window is the same as the analysis window

    nframes = len(x) // R 
    
    z = np.zeros((Nh, nframes), dtype=complex)
    ii = 0
    chunks = np.zeros((nframes, Nh))
    while True:
        idxBeg = ii*R
        idxEnd = idxBeg + Nh
        if idxEnd > len(x):
            break
        xcurr = copy.copy(x[idxBeg:idxEnd])

        chunks[ii, :] = xcurr

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

    return z, chunks


def wola_td_broadcast(x, Nh, w, updateFilterEveryXsamples=1):
    """
    Parameters
    ----------
    x : np.ndarray
        Input signal (1-dimensional).
    Nh : int
        Window length. 
    w : np.ndarray
        Filter coefficients in the frequency domain.
    updateFilterEveryXsamples : int
        Number of samples in-between consecutive time-domain filter updates.
    """

    h = np.sqrt(np.hanning(Nh))
    f = copy.copy(h)    # <-- synthesis window is the same as the analysis window

    # Exponential complex factor (typical 'FFT' notation `W`)
    W = np.exp(1j * 2 * np.pi / Nh) 

    lastUpdate = -1
    ztd = np.zeros_like(x)
    t = np.zeros(Nh)
    t[0] = 1    # initialize impulse responde as Dirac
    for n in range(len(x)):     # loop over samples
        print(f'Sample {n+1}/{len(x)}')

        # Update filter
        if (n - lastUpdate) >= updateFilterEveryXsamples:
            # Build sum
            T = 0
            terms = np.zeros((Nh, len(h)), dtype=complex)
            for kappa in range(Nh):

                # w[kappa] = 1

                term = np.fft.fft(W ** (kappa * np.arange(len(h))) * h, Nh, axis=0)\
                    * w[kappa]\
                    * np.fft.fft(W ** (kappa * np.arange(len(h))) * f, Nh, axis=0)

                terms[kappa, :] = term

                # if kappa == Nh - 1:
                #     fig, axes = plt.subplots(1,1)
                #     fig.set_size_inches(6.5, 2.5)
                #     axes.plot(20*np.log10(np.abs(terms.T)), label='individual channel filters')
                #     # axes.plot(np.sum(20*np.log10(np.abs(terms)), axis=0), 'k')
                #     axes.plot(20*np.log10(np.abs(T)), 'k', label='fullband filter')
                #     axes.plot(20*np.log10(np.abs(w)), 'r--', label='spectral modifications')
                #     axes.grid()
                #     axes.set_ylim([-50, 100])
                #     axes.set_title(f'Sample #{n+1}')
                #     plt.tight_layout()	
                #     plt.show()

                #     stop = 1

                T += term
            T /= Nh
            # Time-domain version of distortion filter
            t = np.fft.ifft(T, Nh, axis=0)
            t = np.real_if_close(t)
            lastUpdate = n

        # Perform convolution
        if n < Nh:
            currChunk = np.concatenate((np.zeros(Nh - n), x[:n]))
        else:
            currChunk = x[(n - Nh):n]

        # # Convolve
        # tmp = sig.convolve(currChunk, t, mode='valid')
        # ztd[n] = tmp[-1]
        # Convolve (manually, just the last sample)
        ztd[n] = sum(currChunk * np.flip(t))

    return ztd


def plotit(z, z_fromtd, idx, unwrapPhase=False):

    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(12.5, 3.5)
    axes[0].plot(20*np.log10(np.abs(z[:, idx])), label='From FD')
    axes[0].plot(20*np.log10(np.abs(z_fromtd[:, idx])), label='From TD')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title('Magnitude [dB]')
    # Filter coefficients
    if not unwrapPhase:
        axes[1].plot(np.angle(z[:, idx]), label='From FD')
        axes[1].plot(np.angle(z_fromtd[:, idx]), label='From TD')
        axes[1].set_title('Phase angle [rad]')
    else:
        axes[1].plot(np.unwrap(np.angle(z[:, idx])), label='From FD')
        axes[1].plot(np.unwrap(np.angle(z_fromtd[:, idx])), label='From TD')
        axes[1].set_title('Phase angle [rad] (unwrapped)')
    axes[1].legend()
    axes[1].grid()
    plt.tight_layout()	
    plt.show()
# def plotit(x, y, w):

#     fig, axes = plt.subplots(3,1)
#     fig.set_size_inches(12.5, 3.5)
#     axes[0].plot(x, label='x')
#     axes[0].plot(y, label='y')
#     axes[0].legend()
#     axes[0].grid()
#     # Filter coefficients
#     axes[1].plot(np.real(w), 'k', label='Re{w}')
#     axes[1].plot(np.imag(w), 'r', label='Im{w}')
#     axes[1].legend()
#     axes[1].grid()
#     # Spectra
#     axes[2].plot(20*np.log10(np.abs(np.fft.fft(x))), label='X')
#     axes[2].plot(20*np.log10(np.abs(np.fft.fft(y))), label='Y')
#     axes[2].legend()
#     axes[2].grid()
#     plt.tight_layout()	
#     plt.show()
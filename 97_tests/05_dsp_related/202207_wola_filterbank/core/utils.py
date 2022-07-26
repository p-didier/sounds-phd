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

def wola_fd_broadcast(x, Nh, R, w, h):
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

    # f = copy.copy(h)    # <-- synthesis window is the same as the analysis window

    nframes = len(x) // R 
    
    z = np.zeros((Nh, nframes), dtype=complex)
    ii = 0
    xchunks = np.zeros((nframes, Nh))
    zchunks = np.zeros((nframes, Nh))
    while True:
        idxBeg = ii * R
        idxEnd = idxBeg + Nh
        if idxEnd > len(x):
            break
        xcurr = copy.copy(x[idxBeg:idxEnd])

        xchunks[ii, :] = xcurr

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

        tmp = np.fft.ifft(z[:, ii])
        zchunks[ii, :] = np.real_if_close(tmp)

        ii += 1

    return z, xchunks, zchunks


def wola_td_broadcast(x, Nh, w, h, updateFilterEveryXsamples=1):
    """
    Parameters
    ----------
    x : np.ndarray
        Input signal (1-dimensional).
    Nh : int
        Window length. 
    w : np.ndarray
        Filter coefficients in the frequency domain.
    h : np.ndarray
        Analysis window.
    updateFilterEveryXsamples : int
        Number of samples in-between consecutive time-domain filter updates.
        If `== None`, the filters are computed once and never updated afterwards.
    """

    f = copy.copy(h)    # <-- WOLA with root-Hann and 50% overlap: synthesis window = analysis window

    # Exponential complex factor (typical 'FFT' notation `W`)
    W = np.exp(1j * 2 * np.pi / Nh) 

    updateFilterDynamically = updateFilterEveryXsamples is not None
    
    if not updateFilterDynamically:
        # Build sum
        T = 0
        terms = np.zeros((Nh, len(h)), dtype=complex)
        for kappa in range(Nh):
            term = np.fft.fft(W ** (kappa * np.arange(len(h))) * h, Nh, axis=0)\
                * w[kappa]\
                * np.fft.fft(W ** (kappa * np.arange(len(f))) * f, Nh, axis=0)
            # term = np.fft.fft(W ** (kappa * np.arange(len(h))) * h, Nh, axis=0)\
            #     * w\
            #     * np.fft.fft(W ** (kappa * np.arange(len(f))) * f, Nh, axis=0)
            term /= (h**2).sum()        # normalize for analysis / synthesis window
            terms[kappa, :] = term      # used for debugging / investigating
            T += term
            
            # if kappa == Nh - 1:
                # fig, axes = plt.subplots(2,1)
                # fig.set_size_inches(6.5, 2.5)
                # axes[0].plot(20*np.log10(np.abs(terms.T)), label='individual channel filters')
                # # axes.plot(np.sum(20*np.log10(np.abs(terms)), axis=0), 'k')
                # axes[0].plot(20*np.log10(np.abs(T)), 'k', label='fullband filter')
                # axes[0].plot(20*np.log10(np.abs(w)), 'r--', label='spectral modifications $w$')
                # axes[0].grid()
                # # axes[0].legend()
                # # axes[0].set_ylim([-50, 100])
                # axes[0].set_title(f'Magnitude of $T(z)$ [dB]')
                # axes[1].plot(np.angle(terms.T), label='individual channel filters')
                # axes[1].plot(np.angle(T), 'k', label='fullband filter')
                # axes[1].plot(np.angle(w), 'r--', label='spectral modifications $w$')
                # axes[1].set_title('Phase angle of $T(z)$ [rad]')
                # axes[1].grid()
                # # axes[1].legend()
                # plt.tight_layout()	
                # plt.show()

                # stop = 1
        T /= Nh
        # Time-domain version of distortion filter
        t = np.fft.ifft(T, Nh, axis=0)
        t = np.real_if_close(t)
        if isinstance(t[0], complex):
            t = np.real(t)
            stop = 1

    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(np.real(t))
    # axes.plot(np.imag(t))
    # axes.grid()
    # plt.tight_layout()	
    # plt.show()

    lastUpdate = -1
    ztd = np.zeros_like(x)
    if updateFilterDynamically:
        t = np.zeros(Nh)
        t[0] = 1    # initialize impulse responde as Dirac
    for n in range(len(x)):     # loop over samples

        # Update filter
        if updateFilterDynamically and ((n - lastUpdate) >= updateFilterEveryXsamples or n == 0):
            # Build sum
            T = 0
            terms = np.zeros((Nh, len(h)), dtype=complex)
            for kappa in range(Nh):
                term = np.fft.fft(W ** (kappa * np.arange(len(h))) * h, Nh, axis=0)\
                    * w[kappa]\
                    * np.fft.fft(W ** (kappa * np.arange(len(h))) * f, Nh, axis=0)
                # term = np.fft.fft(W ** (kappa * np.arange(len(h))) * h, Nh, axis=0)\
                #     * w\
                #     * np.fft.fft(W ** (kappa * np.arange(len(h))) * f, Nh, axis=0)
                term /= h.sum()     # normalize for analysis window
                terms[kappa, :] = term
                T += term

            T /= Nh
            # Time-domain version of distortion filter
            t = np.fft.ifft(T, Nh, axis=0)
            t = np.real_if_close(t)
            lastUpdate = n

        # Perform convolution
        if n < Nh:
            currChunk = np.concatenate((np.zeros(Nh - n - 1), x[:n + 1]))
        else:
            currChunk = x[(n + 1 - Nh):n + 1]

        # # Convolve
        # tmp = sig.convolve(currChunk, t, mode='valid')
        # ztd[n] = tmp[-1]
        # Convolve (manually, just the last sample)
        ztd[n] = sum(currChunk * np.flip(t))

    return ztd


def wola_td_broadcast_naive(x, w):
    """
    Parameters
    ----------
    x : np.ndarray
        Input signal (1-dimensional).
    w : np.ndarray
        Filter coefficients in the frequency domain.
    """

    # Get time-domain version of `w`
    wtd = np.fft.ifft(w)
    wtd = np.real_if_close(wtd)

    # Convolve
    ztd = sig.convolve(x, wtd)

    return ztd


def plotit(x, z, z_fromtd, idx, unwrapPhase=False, plotFullSpectrum=False):

    labelx = 'Original $\\bar{{x}}(l)$'
    labelzFD = '$\\bar{{z}}_\\mathrm{{FD}}(l)$'
    labelzTD = '$\\bar{{z}}_\\mathrm{{TD}}(l)$'

    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(12.5, 3.5)
    if plotFullSpectrum:
        axes[0].plot(20*np.log10(np.abs(x[:, idx])), ':', label=labelx)
        axes[0].plot(20*np.log10(np.abs(z[:, idx])), label=labelzFD)
        axes[0].plot(20*np.log10(np.abs(z_fromtd[:, idx])), label=labelzTD)
    else:
        axes[0].plot(20*np.log10(np.abs(x[:, idx]))[:int(len(x) // 2 + 1)], ':',label=labelx)
        axes[0].plot(20*np.log10(np.abs(z[:, idx]))[:int(len(z) // 2 + 1)], label=labelzFD)
        axes[0].plot(20*np.log10(np.abs(z_fromtd[:, idx]))[:int(len(z_fromtd) // 2 + 1)], label=labelzTD)
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title(f'Frame $l$={idx+1} -- Magnitude [dB]')
    # axes[0].set_ylim([-130, 30])
    # Filter coefficients
    if not unwrapPhase:
        if plotFullSpectrum:
            axes[1].plot(np.angle(x[:, idx]),':', label=labelx)
            axes[1].plot(np.angle(z[:, idx]), label=labelzFD)
            axes[1].plot(np.angle(z_fromtd[:, idx]), label=labelzTD)
        else:
            axes[1].plot(np.angle(x[:, idx])[:int(len(x) // 2 + 1)],':', label=labelx)
            axes[1].plot(np.angle(z[:, idx])[:int(len(z) // 2 + 1)], label=labelzFD)
            axes[1].plot(np.angle(z_fromtd[:, idx])[:int(len(z_fromtd) // 2 + 1)], label=labelzTD)
        axes[1].set_title('Phase angle [rad]')
    else:
        if plotFullSpectrum:
            axes[1].plot(np.unwrap(np.angle(x[:, idx])),':', label=labelx)
            axes[1].plot(np.unwrap(np.angle(z[:, idx])), label=labelzFD)
            axes[1].plot(np.unwrap(np.angle(z_fromtd[:, idx])), label=labelzTD)
        else:
            axes[1].plot(np.unwrap(np.angle(x[:, idx]))[:int(len(x) // 2 + 1)], ':',label=labelx)
            axes[1].plot(np.unwrap(np.angle(z[:, idx]))[:int(len(z) // 2 + 1)], label=labelzFD)
            axes[1].plot(np.unwrap(np.angle(z_fromtd[:, idx]))[:int(len(z_fromtd) // 2 + 1)], label=labelzTD)
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
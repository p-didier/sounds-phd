import time
import scipy.linalg
import soundfile as sf
import simpleaudio as sa
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
    signalType : str
    filterType : str
    windowType : str
    seed : int = 12345


def load_signal(signalType, T, fs, plotit=False, listen=False):
    """
    Loads the input signal.

    Parameters
    ----------
    signalType : str
        Type of input signal ("speech" or "tone").
    T : float
        Signal duration [s]. 
    fs : float or int
        Sampling frequency (used iff `signalType == 'tone'`) [Hz].
    plotit : bool
        If true, plot signal. 
    listen : bool
        If true, plays signal back. 

    Returns
    -------
    x : np.ndarray (float)
        Signal.
    fs : float or int
        Sampling frequency [Hz].
    """

    if signalType == 'speech':
        signalPath = 'U:/py/sounds-phd/02_data/00_raw_signals/01_speech/speech1.wav'
        x, fs = sf.read(signalPath)
        x = x[20000:20000+int(T * fs)]  # cut out initial silence
    elif signalType == 'tone':
        freq = 1000  # frequency [Hz]
        x = np.sin(2 * np.pi * freq * np.arange(int(T * fs)) / fs)
    else:
        raise ValueError(f'Invalid value for `signalType`: "{signalType}"')

    if listen:
        audio_array = x * 32767 / max(abs(audio_array))
        audio_array = audio_array.astype(np.int16)
        sa.play_buffer(audio_array,1,2,fs)

    if plotit:
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        axes.plot(np.arange(len(x)) / fs, x)
        axes.grid()
        axes.set_title(f'Input signal (selected type: "{signalType}")')
        axes.set_xlabel('$t$ [s]')
        plt.tight_layout()
        plt.show()

    return x, fs


def get_filter(filterType, N, seed):
    """
    Generates the frequency-domain filter to be used.

    Parameters
    ----------
    filterType : str
        Type of filter ("none" or "random" or "rir").
    N : int
        Number of taps. 
    seed : int
        Random generator seed (used iff `filterType == 'random'`).

    Returns
    -------
    w : np.ndarray (complex)
        Frequency-domain filter coefficients.
    """

    if filterType == 'none':
        w = np.ones(N)
    elif filterType == 'random':
        # Create random generator
        rng = np.random.default_rng(seed)
        # Generate random impulse response (time-domain filter)
        w = 2 * rng.random(N) - 1
        w = np.fft.fft(w, N, axis=0)
    elif filterType == 'rir':
        wTD, _ = sf.read('U:/py/sounds-phd/97_tests/05_dsp_related/00_signals/rir1.wav')
        wTD = wTD[:N]                    # truncate
        w = np.fft.fft(wTD, N, axis=0)   # freq-domain coefficients
    else:
        raise ValueError(f'Invalid value for `filterType`: "{filterType}"')

    return w


def get_window(windowType, N):
    """
    Generates the analysis window to be used.

    Parameters
    ----------
    windowType : str
        Type of analysis window ("roothann" or "rect").
    N : int
        Window length.

    Returns
    -------
    h : np.ndarray (float)
        Analysis window (time-domain).
    """

    if windowType == 'rect':
        h = np.ones(N)               # rectangular
    elif windowType == 'roothann':
        h = np.sqrt(np.hanning(N))   # root-Hann
    else:
        raise ValueError(f'Invalid value for `windowType`: "{windowType}"')

    return h


def wola_fd_broadcast(x, Nh, R, w, h):
    """
    Runs WOLA with direct broadcasting of frequency-domain
    compressed signals. 

    Parameters
    ----------
    x : np.ndarray (float)
        Input signal (1-dimensional).
    Nh : int
        Window length. 
    R : int
        Window shift.
    w : np.ndarray (complex)
        Filter coefficients.
    h : np.ndarray (float)
        Analysis window.

    Returns
    -------
    z : [Nh x nframes] np.ndarray (complex)
        Compressed signal frames (in WOLA-domain).
    xchunks : [nframes x Nh] np.ndarray (real)
        Time-domain chunks of input signal.
    zchunks : [nframes x Nh] np.ndarray (real)
        Time-domain chunks of compressed signal.
    """

    # Number of signal frames
    nframes = len(x) // R 
    
    # Init output arrays
    z = np.zeros((Nh, nframes), dtype=complex)
    xchunks = np.zeros((nframes, Nh))
    zchunks = np.zeros((nframes, Nh))

    # Start while-loop
    ii = 0
    while True:

        # Chunk input signal
        idxBeg = ii * R
        idxEnd = idxBeg + Nh
        if idxEnd > len(x):     # condition to end while-loop
            break
        xcurr = copy.copy(x[idxBeg:idxEnd])
        xchunks[ii, :] = xcurr  # save for export

        # Multiply by analysis window
        xcurr *= h

        # Apply DFT
        Xcurr = np.fft.fft(xcurr, Nh, axis=0)
        
        # Apply spectral modificationsc(freq.-domain filter `w`)
        z[:, ii] = Xcurr * w

        # Save time-domain version of filtered chunks for export
        tmp = np.fft.ifft(z[:, ii])
        zchunks[ii, :] = np.real_if_close(tmp)

        # Next chunk...
        ii += 1

    return z, xchunks, zchunks


def plot_dist_fct(om, T, w, wtd):
    """
    Plot the distortion function (freq.- and time-domain).

    Parameters
    ----------
    om : np.ndarray (float)
        Angular frequency bins.
    T : np.ndarray (complex)
        Distortion function (freq.-domain). 
    w : np.ndarray (complex)
        Filter coefficients in the frequency domain.
    wtd : np.ndarray (float)
        Filter coefficients in the time domain.
    """

    fig, axes = plt.subplots(3,1)
    fig.set_size_inches(8.5, 3.5)
    plt.tight_layout()
    axes[0].plot(np.real(np.fft.fftshift(w)), label='Real part')
    axes[0].plot(np.imag(np.fft.fftshift(w)), label='Imaginary part')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title('Freq.-domain filter coefficients $\\bar{{w}}$')
    axes[1].plot(om, 20 * np.log10(np.abs(T)), label='Magnitude')
    axes[1].plot(om, np.angle(T), label='Phase')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[1].set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    axes[1].set_title('Distortion function $T(\\mathrm{{e}}^{{\\mathrm{{j}}\\omega}})$')
    axes[1].set_xlabel('$\\omega$')
    myIR = np.fft.ifft(np.fft.ifftshift(T), len(T), axis=0)
    myIR = np.real_if_close(myIR)
    axes[2].plot(myIR, label='$t=\\mathcal{{F}}^{{-1}}(T) / N_h$')
    axes[2].plot(wtd, label='$\\mathcal{{F}}^{{-1}}(\\bar{{w}})$')
    axes[2].grid()
    axes[2].legend()
    axes[2].set_title('Time-domain filter coefficients $t=\\mathcal{{F}}^{{-1}}(T)$')
    plt.show()


def wola_td_broadcast(x, Nh, w, h, plotDistFct=False):
    """
    Approximating WOLA via time-domain version of distortion
    function (fitlerbank interpretation of WOLA).

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
    plotDistFct : bool
        If true, plot the resulting distortion function (freq.- and time-domain).

    Returns
    -------
    ztd : np.ndarray (float)
        Time-domain filtered signal. 
    """

    # WOLA with root-Hann and 50% overlap: synthesis window = analysis window
    f = copy.copy(h)

    # Build distortion function `T(z)` and its time-domain version `t(n)`
    w_td = np.fft.ifft(w, Nh, axis=0)  # compute IDFT of `w`
    w_td = np.real_if_close(w_td)
    T_fct = lambda z : np.flip(z ** (- np.arange(Nh))) @\
                    np.diag(f) @\
                    scipy.linalg.circulant(w_td) @\
                    np.diag(h) @\
                    (z ** (- np.arange(Nh))).T / (Nh // 2)  # <-- normalizing by decimation factor: Nh/2
            
    m = Nh
    om = np.linspace(-np.pi, np.pi, num=m+1)
    om = np.flip(om)
    allz = np.exp(1j * om)

    T_evaluated = np.zeros(len(allz), dtype=complex)
    for ii in range(len(T_evaluated)):
        T_evaluated[ii] = T_fct(allz[ii])
        
    # Time-domain version of distortion filter
    t = np.fft.ifft(np.fft.ifftshift(T_evaluated), len(T_evaluated), axis=0)
    t = np.real_if_close(t)
    t = t[1:]

    # Plot if asked
    if plotDistFct:
        plot_dist_fct(om, T_evaluated, w, w_td)

    ztd = np.zeros_like(x)
    for n in range(len(x)):     # loop over samples
        # Perform convolution
        if n < Nh:
            currChunk = np.concatenate((np.zeros(Nh - n - 1), x[:n + 1]))
        else:
            currChunk = x[(n + 1 - Nh):n + 1]
        # Convolve (manually, just the last sample)
        ztd[n] = sum(currChunk * np.flip(t))

    ztd_wola = np.zeros_like(x)
    ii = 0
    while True:
        # Chunk input signal
        idxBeg = ii * (Nh // 2)
        idxEnd = idxBeg + Nh
        if idxEnd > len(x):     # condition to end while-loop
            break
        xcurr = copy.copy(x[idxBeg:idxEnd])
        # Multiply by analysis window
        xcurr *= h
        # Apply DFT
        Xcurr = np.fft.fft(xcurr, Nh, axis=0)
        # Apply spectral modificationsc(freq.-domain filter `w`)
        Zcurr = Xcurr * w
        # Apply IDFT
        zcurr = np.fft.ifft(Zcurr, Nh, axis=0)
        zcurr = np.real_if_close(zcurr)
        zcurr *= f
        # Combine chunks
        ztd_wola[idxBeg:idxEnd] += zcurr
        # Next chunk...
        ii += 1

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    # axes.plot(ztd)
    axes.plot(ztd_wola)
    axes.plot(sig.convolve(x, t, mode='valid'))
    # axes.plot(x)
    axes.plot(sig.convolve(x, w_td, mode='valid'))
    axes.grid()
    plt.tight_layout()	
    plt.show()

    stop = 1

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
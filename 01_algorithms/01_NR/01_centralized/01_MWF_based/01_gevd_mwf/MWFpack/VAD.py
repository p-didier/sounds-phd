import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
# import time

# @jit(nopython=True)
def oracleVAD(x,tw,thrs,Fs,plotVAD=False):
    # oracleVAD -- Oracle Voice Activity Detection (VAD) function. Returns the
    # oracle VAD for a given speech (+ background noise) signal <x>.
    # Based on the computation of the short-time signal energy.
    #
    # >>> Inputs:
    # -x [N*1 float vector, -] - Time-domain signal.
    # -tw [float, s] - VAD window length.
    # -thrs [float, [<x>]^2] - Energy threshold.
    # -Fs [int, samples/s] - Sampling frequency.
    # -plotVAD [bool] - If true, plots <oVAD> (function output) on a figure.
    # >>> Outputs:
    # -oVAD [N*1 binary vector] - Oracle VAD corresponding to <x>.

    # (c) Paul Didier - 13-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Check input format
    x = np.array(x)     # Ensure it is an array
    if len(x.shape) > 1:
        print('<oracleVAD>: input signal is multidimensional: using 1st row as reference')
        dimsidx = range(len(x.shape))
        x = np.transpose(x, tuple(np.take(dimsidx,np.argsort(x.shape))))   # rearrange x dimensions in increasing order of size
        for ii in range(x.ndim-1):
            x = x[0]    # extract 1 "row" along the largest dimension

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = tw*Fs
    else:
        Nw = 1

    # Compute VAD
    oVAD = np.zeros(n)
    for ii in range(n):
        if Nw == 1:
            chunk_x = x[ii]
        else:
            chunk_x = x[np.arange(ii,int(min(ii+Nw, n)))]

        # Compute short-term signal energy
        E = np.mean(np.abs(chunk_x)**2)

        # Assign VAD value
        if E > thrs:
            oVAD[ii] = 1

    # Time vector
    t = np.arange(n)/Fs

    if plotVAD:
        fig, ax = plt.subplots()
        ax.plot(t, x)
        ax.plot(t, oVAD * np.amax(x)/2)

        plt.legend(['Signal', 'VAD'])

        ax.set(xlabel='t [s]', ylabel='',
            title='Voice Activity Detector')
        ax.grid()

        plt.show()

    return oVAD,t


# @jit(nopython=True)
def getSNR(Y,VAD,silent=False):
    # getSNR -- Sub-function for SNRest().

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Only start computing VAD from the first frame where there has been VAD =
    # 0 and VAD = 1 at least once (condition added on 2021/08/25).
    idx_start = 0
    for ii in range(1,len(VAD)):
        if VAD[ii] != VAD[0]:
            idx_start = ii
            break

    # Check input lengths
    if len(Y) < len(VAD):
        if not silent:
            print('WARNING: VAD is longer than provided signal (possibly due to non-integer Tmax*Fs/(L-R)) --> truncating VAD')
        VAD = VAD[:len(Y)]
    
    # Truncate signals and VAD accordingly
    VAD = VAD[idx_start:]
    Y = Y[idx_start:]

    # Number of time frames where VAD is active/inactive
    Ls = np.count_nonzero(VAD)
    Ln = len(VAD) - Ls

    if Ls > 0 and Ln > 0:
        sigma_n_hat = 1/Ln * np.sum(np.abs(Y)**2 * (1 - VAD))
        sigma_x = 1/Ls * np.sum(np.abs(Y)**2 * VAD)
        sigma_s_hat = sigma_x - sigma_n_hat
        if sigma_s_hat < 0:
            SNR = -1 * float('inf')
        else:
            SNR = 20*np.log10(sigma_s_hat/sigma_n_hat)
    elif Ls == 0:
        SNR = -1 * float('inf')
    elif Ln == 0:
        SNR = float('inf')

    return SNR


# @jit(nopython=True)
def SNRest(Y,VAD):
    # SNRest -- Estimate SNR from time-domain or STFT-domain signal + VAD.
    #
    # >>> Inputs:
    # -Y [Nt*1 float vector /or/ Nf*Nt (complex) float matrix, - ] -
    # Time-domain or STFT of signal.
    # -VAD [Nt*1 binary vector /or/ Nf*Nt binary matrix] - Voice activity
    # detector output (1 = speech present; 0 = speech absent).
    # >>> Outputs:
    # -SNR [float, dB] - Signal-to-Noise Ratio.

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    if len(Y.shape) > 1:   # if STFT format
        SNR = np.zeros(Y.shape[0])
        for k in range(Y.shape[0]):
            SNR[k] = getSNR(Y[k,:], VAD[k,:])
        SNR = np.mean(SNR[np.abs(SNR) != float('inf')])
    else:                   # if time-domain format
        SNR = getSNR(Y, VAD)

    return SNR
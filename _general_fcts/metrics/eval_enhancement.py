from numba.core.types.scalars import EnumMember
import numpy as np
import sys,os
import scipy.signal as sig
sys.path.append(os.path.join(os.path.expanduser('~'), 'py/sounds-phd/_third_parties'))
import pysepm


def get_metrics(cleanSignal, enhancedOrNoisySignal, fs, VAD, gammafwSNRseg=0.2, frameLen=0.03):
    """Compute evaluation metrics for signal enhancement given a single-channel signal.
    Parameters
    ----------
    cleanSignal : [N x 1] np.ndarray (real)
        The clean, noise-free signal used as reference.
    enhancedOrNoisySignal : [N x 1] np.ndarray (real)
        The signal to evaluate (pre- or post-signal enhancement).
    fs : int
        Sampling frequency [samples/s].
    VAD : [N x 1] np.ndarray (real)
        Voice Activity Detector (1: voice + noise; 0: noise only).
    gammafwSNRseg : float
        Gamma exponent for fwSNRseg computation.
    frameLen : float
        Time window duration for fwSNRseg computation [s].

    Returns
    -------
    snr : [3 x 1] np.ndarray (real)
        Unweighted signal-to-noise ratio (SNR).
    fwSNRseg : float    
        Frequency-weighted segmental SNR.
    sisnr : [3 x 1] np.ndarray (real)
        Speech-Intelligibility-weighted SNR [before, after, difference].
    stoi : float
        Short-Time Objective Intelligibility.
    """
    # Check input dimensions
    if len(cleanSignal) != len(enhancedOrNoisySignal):
        if len(cleanSignal) > len(enhancedOrNoisySignal):
            print(f'Enhanc. metrics:: The signal durations do not match: shortening clean signal ({len(cleanSignal)} to {len(enhancedOrNoisySignal)} samples).')
            cleanSignal = cleanSignal[:len(enhancedOrNoisySignal)]
        elif len(cleanSignal) < len(enhancedOrNoisySignal):
            print(f'Enhanc. metrics:: The signal durations do not match: shortening enhanced/noisy signal ({len(enhancedOrNoisySignal)} to {len(cleanSignal)} samples).')
            enhancedOrNoisySignal = enhancedOrNoisySignal[:len(cleanSignal)]

    # Init output arrays
    snr = np.zeros(3)
    sisnr = np.zeros(3)

    # Unweighted SNR
    snr[0] = SNRest(cleanSignal, VAD)
    snr[1] = SNRest(enhancedOrNoisySignal, VAD)
    snr[2] = snr[1] - snr[0]
    # Frequency-weight segmental SNR
    fwSNRseg = pysepm.fwSNRseg(cleanSignal, enhancedOrNoisySignal,
                                    fs, frameLen=frameLen, gamma=gammafwSNRseg)
    # Speech-Intelligibility-weighted SNR (SI-SNR)
    sisnr[0] = get_SISNR(cleanSignal, fs, VAD)
    sisnr[1] = get_SISNR(enhancedOrNoisySignal, fs, VAD)
    sisnr[2] = sisnr[1] - sisnr[0]
    # Short-Time Objective Intelligibility (STOI)
    stoi = pysepm.stoi(cleanSignal, enhancedOrNoisySignal, fs)
    
    return snr, fwSNRseg, sisnr, stoi


def eval(clean_speech, enhanced_or_noisy_speech, Fs, VAD, gamma_fwSNRseg=0.2, frameLen=0.03, onlySTOI=False):

    # Check dimensionalities
    if clean_speech.shape != enhanced_or_noisy_speech.shape:
        raise ValueError('The input array shapes do not match')
    if len(clean_speech.shape) == 1:
        clean_speech = clean_speech[:, np.newaxis()]
    if len(enhanced_or_noisy_speech.shape) == 1:
        enhanced_or_noisy_speech = enhanced_or_noisy_speech[:, np.newaxis()]
    if isinstance(gamma_fwSNRseg, float):
        gamma_fwSNRseg = [gamma_fwSNRseg]
    if isinstance(frameLen, float):
        frameLen = [frameLen]

    # Number of channels to evaluate
    nChannels = clean_speech.shape[1]

    fwSNRseg = np.zeros((nChannels, len(gamma_fwSNRseg), len(frameLen)))
    stoi = np.zeros(nChannels)
    sisnr = np.zeros(nChannels)
    for ii in range(nChannels):
        if not onlySTOI:
            # Frequency-weight segmental SNR
            for jj, gamma in enumerate(gamma_fwSNRseg):
                for kk, lenf in enumerate(frameLen):
                    print('Estimating fwSNRseg for channel %i, for gamma=%.2f and a frame length of %.2f s.' % (ii+1, gamma, lenf))
                    fwSNRseg[ii,jj,kk] = pysepm.fwSNRseg(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs, frameLen=lenf, gamma=gamma)
            # Speech-Intelligibility-weighted SNR (SI-SNR)
            print('Estimating SI-SNR for channel %i.' % (ii+1))
            sisnr[ii] = get_SISNR(enhanced_or_noisy_speech[:,ii], Fs, VAD)
        # Short-Time Objective Intelligibility (STOI)
        print('Estimating STOI for channel %i.' % (ii+1))
        stoi[ii] = pysepm.stoi(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
    
    return fwSNRseg,sisnr,stoi


def get_SISNR(enhanced_or_noisy_speech, Fs, VAD):

    # Speech intelligibility indices (ANSI-S3.5-1997)
    Indices = 1e-4 * np.array([83.0, 95.0, 150.0, 289.0, 440.0, 578.0, 653.0, 711.0,\
         818.0, 844.0, 882.0, 898.0, 868.0, 844.0, 771.0, 527.0, 364.0, 185.0])   
    fc = np.array([160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,\
         1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0])  # corresp. 1/3-octave centre freqs

    SISNR_enhanced = 0
    for ii, fc_curr in enumerate(fc):

        # Filter in 1/3-octave bands
        Wn = 1/Fs * np.array([fc_curr*2**(-1/6), fc_curr*2**(1/6)])
        sos = sig.butter(10, Wn, btype='bandpass', analog=False, output='sos', fs=2*np.pi)
        enhanced_filtered = sig.sosfilt(sos, enhanced_or_noisy_speech)

        # Build the SI-SNR sum
        SISNR_enhanced += Indices[ii] * SNRest(enhanced_filtered,VAD)

    return SISNR_enhanced


def getSNR(Y,VAD,silent=False):
    # getSNR -- Sub-function for SNRest().
    #
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


def SNRest(Y,VAD):
    # SNRest -- Estimate SNR from time-domain VAD.
    #
    # >>> Inputs:
    # -Y [Nt*1 float vector /or/ Nf*J float matrix, - ] - Time-domain signal(s) | frames x channels.
    # -VAD [Nt*1 binary vector] - Voice activity detector output (1 = speech present; 0 = speech absent).
    # >>> Outputs:
    # -SNR [float, dB] - Signal-to-Noise Ratio.

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Input format check
    if len(Y.shape) == 2:
        if Y.shape[1] > Y.shape[0]:
            Y = Y.T
    elif len(Y.shape) == 1:
        Y = Y[:, np.newaxis] 

    nChannels = Y.shape[1]
    SNRy = np.zeros(nChannels)
    for ii in range(nChannels):
        SNRy[ii] = getSNR(Y[:,ii], VAD)

    return SNRy
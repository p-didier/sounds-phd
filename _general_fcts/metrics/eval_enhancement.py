import numpy as np
import sys
import scipy.signal as sig
from scipy.signal import stft
from pathlib import Path, PurePath
from pystoi import stoi as stoi_fcn
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if f'{pathToRoot}/_third_parties' not in sys.path:
    sys.path.append(f'{pathToRoot}/_third_parties')


def get_metrics(cleanSignal, noisySignal, enhancedSignal, fs, VAD, gammafwSNRseg=0.2, frameLen=0.03):
    """Compute evaluation metrics for signal enhancement given a single-channel signal.
    Parameters
    ----------
    cleanSignal : [N x 1] np.ndarray (real)
        The clean, noise-free signal used as reference.
    noisySignal : [N x 1] np.ndarray (real)
        The noisy signal (pre-signal enhancement).
    enhancedSignal : [N x 1] np.ndarray (real)
        The enhanced signal (post-signal enhancement).
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
    stoi : float
        Short-Time Objective Intelligibility.
    """
    # Init output arrays
    snr = np.zeros(3)
    sisnr = np.zeros(3)
    # Unweighted SNR
    snr[0] = get_snr(noisySignal, VAD)
    snr[1] = get_snr(enhancedSignal, VAD)
    snr[2] = snr[1] - snr[0]
    # Frequency-weight segmental SNR
    fwSNRseg = get_fwsnrseg(cleanSignal, enhancedSignal, fs, frameLen, gammafwSNRseg)
    # Short-Time Objective Intelligibility (STOI)
    stoi = stoi_fcn(cleanSignal, enhancedSignal, int(fs))
    return snr, fwSNRseg, stoi


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
                    fwSNRseg[ii,jj,kk] = get_fwsnrseg(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs, frameLen=lenf, gamma=gamma)
            # Speech-Intelligibility-weighted SNR (SI-SNR)
            print('Estimating SI-SNR for channel %i.' % (ii+1))
            sisnr[ii] = get_sisnr(enhanced_or_noisy_speech[:,ii], Fs, VAD)
        # Short-Time Objective Intelligibility (STOI)
        print('Estimating STOI for channel %i.' % (ii+1))
        stoi[ii] = stoi_fcn(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
    
    return fwSNRseg,sisnr,stoi


def get_sisnr(x, Fs, VAD):

    # Speech intelligibility indices (ANSI-S3.5-1997)
    Indices = 1e-4 * np.array([83.0, 95.0, 150.0, 289.0, 440.0, 578.0, 653.0, 711.0,\
         818.0, 844.0, 882.0, 898.0, 868.0, 844.0, 771.0, 527.0, 364.0, 185.0])   
    fc = np.array([160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,\
         1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0])  # corresp. 1/3-octave centre freqs

    sisnr = 0
    for ii, fc_curr in enumerate(fc):

        # Filter in 1/3-octave bands
        Wn = 1/Fs * np.array([fc_curr*2**(-1/6), fc_curr*2**(1/6)])
        sos = sig.butter(10, Wn, btype='bandpass', analog=False, output='sos', fs=2*np.pi)
        x_filtered = sig.sosfilt(sos, x)

        # Build the SI-SNR sum
        sisnr += Indices[ii] * get_snr(x_filtered,VAD)

    return sisnr


def getSNR(timeDomainSignal, VAD, silent=False):
    # getSNR -- Sub-function for SNRest().
    #
    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Ensure correct input formats
    VAD = np.array(VAD)

    # Only start computing VAD from the first frame where there has been VAD =
    # 0 and VAD = 1 at least once (condition added on 25/08/2021).
    idx_start = 0
    for ii in range(1,len(VAD)):
        if VAD[ii] != VAD[0]:
            idx_start = ii
            break

    # Check input lengths
    if len(timeDomainSignal) < len(VAD):
        if not silent:
            print('WARNING: VAD is longer than provided signal (possibly due to non-integer Tmax*Fs/(L-R)) --> truncating VAD')
        VAD = VAD[:len(timeDomainSignal)]
    
    # Truncate signals and VAD accordingly
    VAD = VAD[idx_start:]
    timeDomainSignal = timeDomainSignal[idx_start:]

    # Number of time frames where VAD is active/inactive
    Ls = np.count_nonzero(VAD)
    Ln = len(VAD) - Ls

    if Ls > 0 and Ln > 0:

        noisePower = np.mean(np.power(np.abs(timeDomainSignal[VAD == 0]), 2))
        signalPower = np.mean(np.power(np.abs(timeDomainSignal[VAD == 1]), 2))
        speechPower = signalPower - noisePower
        if speechPower < 0:
            SNRout = -1 * float('inf')
        else:
            SNRout = 20 * np.log10(speechPower / noisePower)
    elif Ls == 0:
        SNRout = -1 * float('inf')
    elif Ln == 0:
        SNRout = float('inf')

    return SNRout


def get_snr(Y,VAD):
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
        SNRy[ii] = getSNR(Y[:, ii], VAD)

    return SNRy


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# vvvv FROM PYSEPM PACKAGE vvvv  https://github.com/schmiph2/pysepm
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
def get_fwsnrseg(cleanSig, enhancedSig, fs, frameLen=0.03, overlap=0.75, gamma=0.2):
    """
    Extracted (and slightly adapted) from pysepm.qualityMeasures package
    See https://github.com/schmiph2/pysepm
    """
    if cleanSig.shape!=enhancedSig.shape:
        raise ValueError('The two signals do not match!')
    eps=np.finfo(np.float64).eps
    cleanSig=cleanSig.astype(np.float64)+eps
    enhancedSig=enhancedSig.astype(np.float64)+eps
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 25# number of critical bands
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    cent_freq[0]  = 50.0000;   bandwidth[0]  = 70.0000;
    cent_freq[1]  = 120.000;   bandwidth[1]  = 70.0000;
    cent_freq[2]  = 190.000;   bandwidth[2]  = 70.0000;
    cent_freq[3]  = 260.000;   bandwidth[3]  = 70.0000;
    cent_freq[4]  = 330.000;   bandwidth[4]  = 70.0000;
    cent_freq[5]  = 400.000;   bandwidth[5]  = 70.0000;
    cent_freq[6]  = 470.000;   bandwidth[6]  = 70.0000;
    cent_freq[7]  = 540.000;   bandwidth[7]  = 77.3724;
    cent_freq[8]  = 617.372;   bandwidth[8]  = 86.0056;
    cent_freq[9] =  703.378;   bandwidth[9] =  95.3398;
    cent_freq[10] = 798.717;   bandwidth[10] = 105.411;
    cent_freq[11] = 904.128;   bandwidth[11] = 116.256;
    cent_freq[12] = 1020.38;   bandwidth[12] = 127.914;
    cent_freq[13] = 1148.30;   bandwidth[13] = 140.423;
    cent_freq[14] = 1288.72;   bandwidth[14] = 153.823;
    cent_freq[15] = 1442.54;   bandwidth[15] = 168.154;
    cent_freq[16] = 1610.70;   bandwidth[16] = 183.457;
    cent_freq[17] = 1794.16;   bandwidth[17] = 199.776;
    cent_freq[18] = 1993.93;   bandwidth[18] = 217.153;
    cent_freq[19] = 2211.08;   bandwidth[19] = 235.631;
    cent_freq[20] = 2446.71;   bandwidth[20] = 255.255;
    cent_freq[21] = 2701.97;   bandwidth[21] = 276.072;
    cent_freq[22] = 2978.04;   bandwidth[22] = 298.126;
    cent_freq[23] = 3276.17;   bandwidth[23] = 321.465;
    cent_freq[24] = 3597.63;   bandwidth[24] = 346.136;


    W=np.array([0.003,0.003,0.003,0.007,0.010,0.016,0.016,0.017,0.017,0.022,0.027,0.028,0.030,0.032,0.034,0.035,0.037,0.036,0.036,0.033,0.030,0.029,0.027,0.026,
    0.026])

    bw_min=bandwidth[0]
    min_factor = np.exp (-30.0 / (2.0 * 2.303));#      % -30 dB point of filter

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    j = np.arange(0,n_fftby2)


    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i,:] = np.exp (-11 *(((j - np.floor(f0))/bw)**2) + norm_factor)
        crit_filter[i,:] = crit_filter[i,:]*(crit_filter[i,:] > min_factor)

    num_frames = len(cleanSig)/skiprate-(winlength/skiprate)# number of frames
    start      = 1 # starting sample
    #window     = 0.5*(1 - cos(2*pi*(1:winlength).T/(winlength+1)));


    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    f,t,Zxx=stft(cleanSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec=np.abs(Zxx)
    clean_spec=clean_spec[:-1,:]
    clean_spec=(clean_spec/clean_spec.sum(0))
    f,t,Zxx=stft(enhancedSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    enh_spec=np.abs(Zxx)
    enh_spec=enh_spec[:-1,:]
    enh_spec=(enh_spec/enh_spec.sum(0))

    clean_energy=(crit_filter.dot(clean_spec))
    processed_energy=(crit_filter.dot(enh_spec))
    error_energy=np.power(clean_energy-processed_energy,2)
    error_energy[error_energy<eps]=eps
    W_freq=np.power(clean_energy,gamma)
    SNRlog=10*np.log10((clean_energy**2)/error_energy)
    fwSNR=np.sum(W_freq*SNRlog,0)/np.sum(W_freq,0)
    distortion=fwSNR.copy()
    distortion[distortion<-10]=-10
    distortion[distortion>35]=35

    return np.mean(distortion)
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ^^^^ FROM PYSEPM PACKAGE ^^^^  https://github.com/schmiph2/pysepm
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
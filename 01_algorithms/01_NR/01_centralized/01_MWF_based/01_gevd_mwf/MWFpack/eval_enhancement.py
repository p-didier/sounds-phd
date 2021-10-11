import numpy as np
import pysepm

def eval(clean_speech, enhanced_or_noisy_speech, Fs):

    # Check dimensionalities
    if clean_speech.shape != enhanced_or_noisy_speech.shape:
        raise ValueError('The input array shapes do not match')
    if len(clean_speech.shape) == 1:
        clean_speech = clean_speech[:, np.newaxis()]
    if len(enhanced_or_noisy_speech.shape) == 1:
        enhanced_or_noisy_speech = enhanced_or_noisy_speech[:, np.newaxis()]

    # Number of channels to evaluate
    nChannels = clean_speech.shape[1]

    fwSNRseg = np.zeros(nChannels)
    SNRseg = np.zeros(nChannels)
    stoi = np.zeros(nChannels)
    for ii in range(nChannels):
        # Frequency-weight segmental SNR
        fwSNRseg[ii] = pysepm.fwSNRseg(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
        # Segmental SNR
        SNRseg[ii] = pysepm.SNRseg(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
        # Short-Time Objective Intelligibility (STOI)
        stoi[ii] = pysepm.stoi(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
    
    return fwSNRseg,SNRseg,stoi
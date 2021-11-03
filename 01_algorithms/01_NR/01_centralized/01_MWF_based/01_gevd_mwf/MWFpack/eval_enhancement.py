import numpy as np
from numpy.lib.arraysetops import isin
import pysepm

def eval(clean_speech, enhanced_or_noisy_speech, Fs, gamma_fwSNRseg=0.2, frameLen=0.03):

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
    for ii in range(nChannels):
        # Frequency-weight segmental SNR
        for jj, gamma in enumerate(gamma_fwSNRseg):
            for kk, lenf in enumerate(frameLen):
                fwSNRseg[ii,jj,kk] = pysepm.fwSNRseg(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs, frameLen=lenf, gamma=gamma)
        # Short-Time Objective Intelligibility (STOI)
        stoi[ii] = pysepm.stoi(clean_speech[:,ii], enhanced_or_noisy_speech[:,ii], Fs)
    
    return fwSNRseg,stoi
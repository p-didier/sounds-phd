# Purpose of script:
# Utilities for online DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np

def get_window(winType, nfft):
    
    if winType == 'hann':
        win = np.hanning(nfft)
    elif winType == 'sqrt-hann':
        win = np.sqrt(np.hanning(nfft))
    elif winType == 'hamming':
        win = np.hamming(nfft)
    elif winType == 'rect':
        win = np.ones(nfft)
    else:
        raise ValueError('Window type not recognized')

    return win
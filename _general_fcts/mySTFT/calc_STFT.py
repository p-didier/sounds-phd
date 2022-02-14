from multiprocessing.managers import ValueProxy
import numpy as np
import scipy.fft

def calcSTFT(x, Fs, win, N_STFT, R_STFT, sides='onesided'):
    # X, f = calcSTFT(x, fs, win, N_STFT, R_STFT, sides)
    # performs the STFT.
    #
    # IN:
    # x         signal - samples x channels
    # Fs        sampling frequency
    # win       window function
    # N_STFT    frame length
    # R_STFT    frame shift
    # sides     {'onesided', 'twosided'}, return either onesided or twosided STFT
    # 
    # OUT:
    # x_STFT    STFT tensor - freqbins x frames x channels
    # f         frequency vector
    #
    # Original MATLAB implementation by Thomas Dietzen (calc_STFT.m).
    # Translated to Python by Paul Didier (First: Sept. 2021).

    # Check input format
    if len(x.shape) == 2:
        if x.shape[0] < x.shape[1]:
            print('<calcSTFT>: Input <x> seems transposed --> flipping dimensions.')
            x = x.T
    if N_STFT > x.shape[0]:
        raise ValueError('The chosen STFT frame length is larger than the total signal length.')

    # Use only half of the FFT spectrum  
    N_STFT_half = int(N_STFT/2 + 1)

    # Get frequency vector
    f = np.linspace(0, Fs/2, N_STFT_half)
    if sides == 'twosided':
        f = np.concatenate((f, -np.flip(f[1:-1])))

    # Init
    numFrames = int(np.floor((x.shape[0] - N_STFT + R_STFT)/R_STFT))
    if len(x.shape) == 2:
        numChannels = x.shape[1]
    else: 
        numChannels = 1

    if sides == 'onesided':
        x_STFT = np.zeros((N_STFT_half, numFrames, numChannels), dtype=complex) 
    elif sides == 'twosided':
        x_STFT = np.zeros((N_STFT, numFrames, numChannels), dtype=complex)   

    # Compute STFT frame-by-frame
    for m in range(numChannels):
        for l in range(numFrames): 
            idxx = range(int(l*R_STFT), int(l*R_STFT + N_STFT))
            if len(x.shape) == 2:
                x_frame = x[idxx, m]
            else:
                x_frame = x[idxx]
                
            X_frame = scipy.fft.fft(win * x_frame)
            if sides == 'onesided':
                x_STFT[:,l,m] = X_frame[:N_STFT_half]
            elif sides == 'twosided':              
                x_STFT[:,l,m] = X_frame

    # Reduce dimension if m == 1
    x_STFT = np.squeeze(x_STFT)

    return x_STFT, f

def calcISTFT(X, win, N_STFT, R_STFT, sides='onesided'):
    # x = calcISTFT(X, win, N_STFT, R_STFT, sides)
    # performs the inverse STFT.
    #
    # IN:
    # X         STFT tensor - freqbins x frames x channels
    # win       window function
    # N_STFT    frame length
    # R_STFT    frame shift
    # sides     {'onesided', 'twosided'}, return either onesided or twosided STFT
    # 
    # OUT:
    # x         signal - samples x channels
    #
    # Original MATLAB implementation by Thomas Dietzen (calc_STFT.m).
    # Translated to Python by Paul Didier (First: Sept. 2021).

    raise ValueError('calcISTFT: THIS FUNCTION IS WRONGLY IMPLEMENTED. DO NOT USE. (paul: 20/01/2022')

    L = X.shape[1]
    M = X.shape[2]
    if sides == 'onesided':
        X = np.concatenate((X, np.flip(X[1:-1,:,:].conj(), axis=0)))
    x_frames = np.real(scipy.fft.ifft(X, axis=0))
    
    # Apply synthesis window
    x_frames = x_frames * win[:,np.newaxis,np.newaxis]
    x_frames = x_frames[:N_STFT,:,:]

    # Init output
    x = np.zeros((int(R_STFT*(L-1)+N_STFT), M))
    # OLA processing
    for l in range(L):
        sampIdx = range(int(l*R_STFT), int(l*R_STFT+N_STFT))
        x[sampIdx,:] += x_frames[:,l,:]

    return x
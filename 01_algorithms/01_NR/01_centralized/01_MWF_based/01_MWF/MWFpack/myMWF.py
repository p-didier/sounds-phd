import numpy as np
import scipy.signal as sig
import time
from numba import jit

def MWF(y,Fs,win,L,R,VAD,beta,min_covUpdates):
    # MWF -- Compute the standard MWF for a given set of sensor signals
    # capturing the same acoustic scenario from different locations (speech +
    # noise in rectangular, damped room), in the STFT domain, using WOLA. 
    #
    # >>> Inputs: 
    # -y [Nt*J (complex) float matrix, -] - Raw sensor signals. 
    # -Fs [int, samples/s] - Sampling frequency. 
    # -win [L*1 float vector, -] - STFT window. 
    # -L [int, samples] - Length of individual STFT time frame.
    # -R [int, samples] - Overlap size between STFT time frames.
    # -VAD [Nt*1 binary vector] - Voice Activity Detector.
    # -beta [float [[0,1]], -] - Covariance matrices exponential average constant.
    # -min_covUpdates [int, -] - Minimum # of covariance matrices updates
    #                            before first filter weights update.
    # >>> Outputs:
    # -d_hat [Nt*J (complex) float matrix, -] - Estimated desired signals. 

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Number of frames needed to cover the whole signal
    nframes = int(np.ceil(y.shape[0]/(L-R))) + 1
    nbins = int(L/2)+1
    
    # Compute STFT
    print('Computing STFTs of sensor observations with %i frames and %i bins...' % (nframes,nbins))    
    if y.shape[1] > 1:
        Y = np.zeros((nbins,nframes,y.shape[1]), dtype=complex)
        for ii in range(y.shape[1]):
            Y[:,:,ii] = sig.stft(y[:,ii], fs=Fs, window=win,\
                 nperseg=L, noverlap=R, return_onesided=True)[2]
    else:
        Y = sig.stft(y, fs=Fs, window=win, nperseg=L, noverlap=R, return_onesided=True)[2]

    print('Filtering sensor signals...\n\n')

    # Init arrays
    VAD_l = np.zeros(nframes)

    # Number of sensors
    J = y.shape[1]

    # Init arrays
    d_hat = np.zeros((y.shape[0],J))          # initiate desired signal estimates matrix
    Ryy = np.zeros((J,J,nbins),dtype=complex)                   # initiate sensor signals covariance matrix estimate
    Rnn = np.zeros((J,J,nbins),dtype=complex)                   # initiate noise covariance matrix estimate
    Rss = np.zeros((J,J,nbins),dtype=complex)                   # initiate speech covariance matrix estimate
    VAD_l = np.zeros(nframes)                               # initiate frame-wise VAD
    w_hat = np.identity(J,dtype=complex)
    w_hat = np.repeat(w_hat[:, :, np.newaxis], nbins, axis=2)   # initiate filter weight estimates 
    nUpdatesRnn = np.zeros(nbins)                
    nUpdatesRyy = np.zeros(nbins)

    D_hat = np.zeros_like(Y, dtype=complex) 
    updateWeights = np.zeros(nbins)    # flag to know whether or not to update 

    # Loop over time frames
    for l in range(nframes):
        
        t0 = time.time()
        
        # Time-frame samples' indices
        idxChunk = np.arange((l-1)*(L - R), np.amin([l*(L - R), y.shape[0]]), dtype=int)
        
        # Current frame VAD (majority of speech-active or speech-inactive samples?)
        VAD_l[l] = np.count_nonzero(VAD[idxChunk]) > len(idxChunk)/2
        
        # Loop over frequency bins
        for kp in range(nbins):

            Ytf = np.squeeze(Y[kp,l,:])

            if VAD_l[l]:        # "speech + noise" time frame
                Ryy[:,:,kp] = expavg_covmat(np.squeeze(Ryy[:,:,kp]), beta, Ytf)
                nUpdatesRyy[kp] += 1   
            else:                # "noise only" time frame
                Rnn[:,:,kp] = expavg_covmat(np.squeeze(Rnn[:,:,kp]), beta, Ytf)
                nUpdatesRnn[kp] += 1

            # ---- Check quality of covariance estimates ----
            if not updateWeights[kp]:
                # Check #1 - Need full rank covariance matrices
                if np.linalg.matrix_rank(np.squeeze(Ryy[:,:,kp])) == J and\
                    np.linalg.matrix_rank(np.squeeze(Rnn[:,:,kp])) == J:
                    # Check #2 - Must have had a min. # of updates on each matrix
                    if nUpdatesRyy[kp] >= min_covUpdates and nUpdatesRnn[kp] >= min_covUpdates:
                        updateWeights[kp] = True

            # Update filter coefficients
            if updateWeights[kp]:
                w_hat[:,:,kp] = update_w(np.squeeze(Ryy[:,:,kp]), np.squeeze(Rnn[:,:,kp]))

            # Desired signal estimates
            D_hat[kp,l,:] = np.squeeze(w_hat[:,:,kp]).conj().T @ Ytf

        t1 = time.time()
        # print('Processed time frame %i/%i in %2f s...' % (l,nframes,t1-t0))

    print('MW-filtering done.')

    return D_hat


@jit(nopython=True)
def expavg_covmat(R, beta, Y):
    return beta*R + (1 - beta)*np.outer(Y, Y.conj())

@jit(nopython=True)
def update_w(Ryy,Rnn):
    # Estimate speech covariance matrix
    Rss = Ryy - Rnn
    # LMMSE weights
    return np.linalg.pinv(Ryy) @ Rss  # eq.(19) in ruiz2020a

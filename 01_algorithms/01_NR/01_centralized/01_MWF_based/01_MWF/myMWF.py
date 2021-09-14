import numpy as np
import scipy

def MWF(y,VAD,beta,L,N,min_covUpdates):
    # MWF -- Compute the standard MWF for a given set of sensor signals
    # capturing the same acoustic scenario from different locations (speech +
    # noise in rectangular, damped room), in the STFT domain. 
    #
    # >>> Inputs: 
    # -y [Nt*J (complex) float matrix, -] - Raw sensor signals. 
    # -VAD [Nt*1 binary vector] - Voice Activity Detector.
    # -beta [float [[0,1]], -] - Covariance matrices exponential average constant.
    # -L [int, samples] - Length of individual signal chunks (for (W)OLA).
    # -N [int, samples] - Length of (W)OLA's full window (inc. 0-padding).
    # -min_covUpdates [int, -] - Minimum # of covariance matrices updates
    #                            before first filter weights update.
    # >>> Outputs:
    # -d_hat [Nt*J (complex) float matrix, -] - Estimated desired signals. 

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------
    
    # WOLA parameters for long signals filtering
    M = N - L + 1              # overlap size [samples]
    print('   INFO: Using %i frequency lines for FFT computations.' % N)

    # Number of frames needed to cover the whole signal
    nframes = int(np.ceil(y.shape[0]/L))
    print('   INFO: Whole signal is divided in %i time frames (%i samples overlap)' % (nframes,M))    

    print('Filtering sensor signals...\n\n')

    # Init arrays
    VAD_l = np.zeros(nframes)

    # Number of sensors
    J = y.shape[1]
    d_hat = np.zeros((y.shape[0],J))          # initiate desired signal estimates matrix
    Ryy = np.zeros((J,J,N),dtype=complex)                   # initiate sensor signals covariance matrix estimate
    Rnn = np.zeros((J,J,N),dtype=complex)                   # initiate noise covariance matrix estimate
    Rss = np.zeros((J,J,N),dtype=complex)                   # initiate speech covariance matrix estimate
    VAD_l = np.zeros(nframes)                               # initiate frame-wise VAD
    w_hat = np.identity(J,dtype=complex)
    w_hat = np.repeat(w_hat[:, :, np.newaxis], N, axis=2)   # initiate filter weight estimates 
    nUpdatesRnn = np.zeros(N)                
    nUpdatesRyy = np.zeros(N)

    # Loop over time frames
    for l in range(nframes):
        
        # Time-frame samples' indices
        idxChunk = np.arange(1 + (l-1)*L, np.amin([l*L, y.shape[0]]))
        
        # Current frame VAD (majority of speech-active or speech-inactive samples?)
        VAD_l[l] = np.count_nonzero(VAD[idxChunk]) > len(idxChunk)/2
        
        # Build sensor observations matrix in freq. domain for current time frame
        Y_l = np.zeros((J,N),dtype=complex)
        for k in range(J):
            chunk_y = np.concatenate((y[idxChunk,k], np.zeros(N - len(idxChunk))))   # chunk of sensor signals
            Chunk_y = 1/N * scipy.fft.fft(chunk_y, n=N)     # FFT (sensor)
            Y_l[k,:] = Chunk_y

        # Loop over frequency bins
        D_hat_l = np.zeros((N,J),dtype=complex)
        for kp in range(N):
            if VAD_l[l]:        # "speech + noise" time frame
                Ryy[:,:,kp] = beta*np.squeeze(Ryy[:,:,kp]) + (1 - beta)*np.outer(Y_l[:,kp], Y_l[:,kp].conj())
                nUpdatesRyy[kp] += 1   
            else:                # "noise only" time frame
                Rnn[:,:,kp] = beta*np.squeeze(Rnn[:,:,kp]) + (1 - beta)*np.outer(Y_l[:,kp], Y_l[:,kp].conj())
                nUpdatesRnn[kp] += 1

            # ---- Check quality of covariance estimates ----
            updateWeights = False
            # Check #1 - Need full rank covariance matrices
            if np.linalg.matrix_rank(np.squeeze(Ryy[:,:,kp])) == J and\
                 np.linalg.matrix_rank(np.squeeze(Rnn[:,:,kp])) == J:
                # Check #2 - Must have had a min. # of updates on each matrix
                if nUpdatesRyy[kp] >= min_covUpdates and nUpdatesRnn[kp] >= min_covUpdates:
                    updateWeights = True


            # Update filter coefficients
            if updateWeights is True:
                # Estimate speech covariance matrix
                Rss[:,:,kp] = np.squeeze(Ryy[:,:,kp]) - np.squeeze(Rnn[:,:,kp])
                # LMMSE weights
                w_hat[:,:,kp] = np.linalg.pinv(np.squeeze(Ryy[:,:,kp])) @ np.squeeze(Rss[:,:,kp])   # eq.(19) in ruiz2020a

            # Desired signal estimates
            D_hat_l[kp,:] = np.squeeze(w_hat[:,:,kp]).conj().T @ Y_l[:,kp]

        # Back to time-domain (+ getting rid of computation-precision-small imaginary part)
        d_hat_l = N * np.real(scipy.fft.ifft(D_hat_l, N, axis=0))

        # Build desired signal estimates (OLA)
        idxOutput =  np.arange(1 + (l-1)*L, np.amin([l*M - 1, y.shape[0]]))
        d_hat[idxOutput,:] += d_hat_l[:len(idxOutput),:]

    print('MW-filtering done.')

    return d_hat
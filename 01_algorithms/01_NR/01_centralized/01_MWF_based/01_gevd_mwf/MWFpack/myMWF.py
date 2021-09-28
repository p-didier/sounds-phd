import numpy as np
import scipy.signal as sig
import time
from numba import jit
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT, calcISTFT

def MWF(y,Fs,win,L,R,VAD,beta,min_covUpdates,useGEVD=False,GEVDrank=1):
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
    # -useGEVD [bool] - If true, use the GEVD, do not otherwise (standard MWF).
    # -GEVDrank [int, -] - GEVD rank approximation (default: 1).
    # >>> Outputs:
    # -d_hat [Nt*J (complex) float matrix, -] - Estimated desired signals. 

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Number of frames needed to cover the whole signal
    nframes = int(np.floor(y.shape[0]/(L-R))) - 1
    nbins = int(L/2)+1
    
    # Compute STFT
    print('Computing STFTs of sensor observations with %i frames and %i bins...' % (nframes,nbins))    
    Y = calcSTFT(y, Fs, win, L, R, 'onesided')[0]

    print('Filtering sensor signals...\n\n')

    # Init arrays
    VAD_l = np.zeros(nframes)

    # Number of sensors
    J = y.shape[1]

    # Init arrays
    Ryy = np.zeros((J,J,nbins),dtype=complex)                   # initiate sensor signals covariance matrix estimate
    Rnn = np.zeros((J,J,nbins),dtype=complex)                   # initiate noise covariance matrix estimate
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
                if useGEVD:
                    # Perform generalized eigenvalue decomposition
                    Sigma_yy,Q = mygevd(np.squeeze(Ryy[:,:,kp]),np.squeeze(Rnn[:,:,kp]))
                    # Sort eigenvalues in descending order
                    idx = np.flip(np.argsort(np.diag(Sigma_yy)))
                    Sigma_yy = np.diag(np.flip(np.sort(np.diag(Sigma_yy))))
                    Q = Q[:,idx]
                    # LMMSE weights 
                    w_hat[:,:,kp] = update_w_GEVDMWF(Sigma_yy,Q,GEVDrank) 
                else:
                    w_hat[:,:,kp] = update_w_MWF(np.squeeze(Ryy[:,:,kp]), np.squeeze(Rnn[:,:,kp]))

            # Desired signal estimates
            D_hat[kp,l,:] = np.squeeze(w_hat[:,:,kp]).conj().T @ Ytf

        t1 = time.time()
        print('Processed time frame %i/%i in %2f s...' % (l,nframes,t1-t0))

    print('MW-filtering done.')

    return D_hat


@jit(nopython=True)
def expavg_covmat(Ryy, beta, Y):
    return beta*Ryy + (1 - beta)*np.outer(Y, Y.conj())

@jit(nopython=True)
def update_w_MWF(Ryy,Rnn):
    # Estimate speech covariance matrix
    Rss = Ryy - Rnn
    # LMMSE weights
    return np.linalg.pinv(Ryy) @ Rss  # eq.(19) in ruiz2020a

@jit(nopython=True)
def update_w_GEVDMWF(S,Q,GEVDrank):
    # Estimate speech covariance matrix
    sig_nn = np.ones(S.shape[0])
    sig_yy = np.diag(S)
    diagveig = 1 - sig_nn[:GEVDrank] / sig_yy[:GEVDrank]   # rank <R> approximation
    diagveig = np.append(diagveig, np.zeros(S.shape[0] - GEVDrank))
    # LMMSE weights
    return np.linalg.pinv(Q.conj().T) @ np.diag(diagveig) @ Q.conj().T

@jit(nopython=True)
def mygevd(A,B):
    # mygevd -- Compute the generalized eigenvalue decomposition
    # of the matrix pencil {A,B}. 
    #
    # >>> Inputs: 
    # -A [N*N (complex) float matrix, -] - Symmetrical matrix.
    # -B [N*N (complex) float matrix, -] - Symmetrical positive-definite matrix.
    # >>> Outputs:
    # -S [N*N (complex) float matrix, -] - Diagonal matrix of generalized eigenvalues. 
    # -Q [N*N (complex) float matrix, -] - Corresponding matrix of generalized eigenvectors. 

    # (c) Paul Didier - 24-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # Based on Algorithm 8.7.1 in Golub's "Matrix Computations" 4th edition (2013).
    # ------------------------------------

    # Cholesky factorization
    G = np.linalg.cholesky(B - 1j*np.imag(B))       # B = G*G^H

    C1 = np.linalg.solve(G,A)
    C = np.linalg.solve(G.conj(),C1.T)    # C = G^(-1)*A*G^(-H)

    s, Qa = np.linalg.eig(C)
    S = np.diag(s)

    X = np.linalg.solve(G.conj().T, Qa)   # X = G^(-H)*Qa
    Q = np.linalg.pinv(X.conj().T)

    return S,Q
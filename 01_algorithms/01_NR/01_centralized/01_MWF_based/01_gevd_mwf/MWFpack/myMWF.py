import numpy as np
from numpy.lib.shape_base import column_stack
import scipy.signal as sig
import scipy
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
    y_STFT = calcSTFT(y, Fs, win, L, R, 'onesided')[0]
    # y2 = calcISTFT(y_STFT, win, L, R, 'onesided')  # Check ISTFT outcome

    print('Filtering sensor signals...\n\n')

    # Init arrays
    VAD_l = np.zeros(nframes)

    # Number of sensors
    nNodes = y.shape[1]

    # Init arrays
    Ryy = np.zeros((nNodes,nNodes,nbins),dtype=complex)                   # initiate sensor signals covariance matrix estimate
    Rnn = np.zeros((nNodes,nNodes,nbins),dtype=complex)                   # initiate noise covariance matrix estimate
    #
    Sigma_yy = np.zeros((nNodes,nNodes,nbins),dtype=complex)              # initiate GEVD eigenvalues matrix 
    Qmat = np.zeros((nNodes,nNodes,nbins),dtype=complex)                  # initiate GEVD eigenvectors matrix 
    #
    VAD_l = np.zeros(nframes)                               # initiate frame-wise VAD
    W_hat = np.identity(nNodes,dtype=complex)
    W_hat = np.repeat(W_hat[:, :, np.newaxis], nbins, axis=2)   # initiate filter weight estimates 
    nUpdatesRnn = np.zeros(nbins)                
    nUpdatesRyy = np.zeros(nbins)
    #
    D_hat = np.zeros_like(y_STFT, dtype=complex) 
    updateWeights = np.zeros(nbins)    # flag to know whether or not to update 

    # TMP
    Sigma_nn = np.zeros((nNodes,nNodes,nbins),dtype=complex)              # initiate GEVD eigenvalues matrix FOR NOISE ONLY PERIODS
    Qmat_n = np.zeros((nNodes,nNodes,nbins),dtype=complex)                  # initiate GEVD eigenvectors matrix FOR NOISE ONLY PERIODS

    # Loop over time frames
    for l in range(nframes):
        
        t0 = time.time()
        
        # Time-frame samples' indices
        idxChunk = np.arange(l*(L - R), np.amin([l*(L - R) + L, y.shape[0]]), dtype=int)
        
        # Current frame VAD (majority of speech-active or speech-inactive samples?)
        VAD_l[l] = np.count_nonzero(VAD[idxChunk]) > len(idxChunk)/2
        
        # Loop over frequency bins
        for kp in range(nbins):

            Ytf = np.squeeze(y_STFT[kp,l,:])

            if VAD_l[l]:        # "speech + noise" time frame
                Ryy[:,:,kp] = expavg_covmat(np.squeeze(Ryy[:,:,kp]), beta, Ytf)
                nUpdatesRyy[kp] += 1   
            else:                # "noise only" time frame
                Rnn[:,:,kp] = expavg_covmat(np.squeeze(Rnn[:,:,kp]), beta, Ytf)
                nUpdatesRnn[kp] += 1

            # ---- Check quality of covariance estimates ----
            if not updateWeights[kp]:
                # Check #1 - Need full rank covariance matrices
                if np.linalg.matrix_rank(np.squeeze(Ryy[:,:,kp])) == nNodes and\
                    np.linalg.matrix_rank(np.squeeze(Rnn[:,:,kp])) == nNodes:
                    # Check #2 - Must have had a min. # of updates on each matrix
                    if nUpdatesRyy[kp] >= min_covUpdates and nUpdatesRnn[kp] >= min_covUpdates:
                        updateWeights[kp] = True

            # Update filter coefficients
            if updateWeights[kp]:
                if useGEVD:
                    # Perform generalized eigenvalue decomposition
                    sig,Xmat = scipy.linalg.eigh(np.squeeze(Ryy[:,:,kp]),np.squeeze(Rnn[:,:,kp]))
                    q = np.linalg.pinv(Xmat.conj().T)     
                    # Sort eigenvalues in descending order
                    Sigma_yy[:,:,kp], Qmat[:,:,kp] = sortgevd(np.diag(sig),q)
                    W_hat[:,:,kp] = update_w_GEVDMWF(Sigma_yy[:,:,kp], Qmat[:,:,kp], GEVDrank)          # LMMSE weights
                else:
                    W_hat[:,:,kp] = update_w_MWF(np.squeeze(Ryy[:,:,kp]), np.squeeze(Rnn[:,:,kp]))      # LMMSE weights

            # Desired signal estimates for each node separately (last dimension of <D_hat>)
            D_hat[kp,l,:] = np.squeeze(W_hat[:,:,kp]).conj().T @ Ytf

            # # Get output SNR for current TF-bin
            # snrout = SNRout(np.squeeze(W_hat[:,:,kp]), np.squeeze(Ryy[:,:,kp]), np.squeeze(Rnn[:,:,kp]))

        t1 = time.time()
        if l % 10 == 0:
            print('Processed time frame %i/%i in %2f s...' % (l,nframes,t1-t0))

    print('MW-filtering done.')

    return D_hat

def SNRout(w,Rxx,Rnn):
    # Derives output SNR for current TF-bin (see equation 2.59 in Randy's thesis)
    a = w.conj().T @ Rxx @ w
    b = w.conj().T @ Rnn @ w
    snr = a @ np.linalg.pinv(b)
    return snr

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
    sig_yy = np.diag(S)
    diagveig = np.ones(GEVDrank) - np.ones(GEVDrank) / sig_yy[:GEVDrank]   # rank <R> approximation
    diagveig = np.append(diagveig, np.zeros(S.shape[0] - GEVDrank))
    # LMMSE weights
    return np.linalg.pinv(Q.conj().T) @ np.diag(diagveig) @ Q.conj().T

@jit(nopython=True)
def mygevd(A,B):
    # mygevd -- Compute the generalized eigenvalue decomposition
    # of the matrix pencil {A,B}. 
    #
    # >>> Inputs: 
    # -A [N*N (complex) float matrix, -] - Symmetrical matrix, full rank.
    # -B [N*N (complex) float matrix, -] - Symmetrical positive-definite matrix, full rank.
    # >>> Outputs:
    # -S [N*N (complex) float matrix, -] - Diagonal matrix of generalized eigenvalues. 
    # -Q [N*N (complex) float matrix, -] - Corresponding matrix of generalized eigenvectors. 

    # (c) Paul Didier - 24-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # Based on Algorithm 8.7.1 in Golub's "Matrix Computations" 4th edition (2013).
    # ------------------------------------

    # Cholesky factorization
    Gmat = np.linalg.cholesky(B - 1j*np.imag(B))       # B = G*G^H

    C1mat = np.linalg.solve(Gmat,A)
    Cmat = np.linalg.solve(Gmat.conj(),C1mat.T)    # C = G^(-1)*A*G^(-H)

    s, Qa = np.linalg.eig(Cmat)
    Smat = np.diag(s)

    Xmat = np.linalg.solve(Gmat.conj().T, Qa)   # X = G^(-H)*Qa
    Qmat = np.linalg.pinv(Xmat.conj().T)

    # print(Qmat)

    stop = 1 

    return Smat,Qmat

def sortgevd(S,Q,order='descending'):
    # Sorts outcome of GEVD in descending eigenvalues order
    if order == 'descending':
        idx = np.flip(np.argsort(np.diag(S)))
        S_sorted = np.diag(np.flip(np.sort(np.diag(S))))
    elif order == 'ascending':
        idx = np.argsort(np.diag(S))
        S_sorted = np.diag(np.sort(np.diag(S)))
    Q_sorted = Q[:,idx]
    return S_sorted,Q_sorted


# TEMPORARY FUNCTIONS
def compare_cols(A,B):

    # Check inputs format
    if not all(len (row) == len (A) for row in A):
        raise ValueError('First argument must be a square matrix')
    if not all(len (row) == len (B) for row in B):
        raise ValueError('Second argument must be a square matrix')

    nCols = A.shape[1]
    idx = np.zeros(nCols, dtype=int)
    for ii in range(nCols):
        mycol = A[:,ii]

        dist_to_B = np.zeros(nCols)
        for jj in range(nCols):
            dist_to_B[jj] = np.linalg.norm(mycol - B[:,jj])
        
        idx[ii] = np.argmin(dist_to_B)

    return idx
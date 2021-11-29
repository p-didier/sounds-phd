import numpy as np
import scipy
import time
from numba import njit
import sys, os
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT, calcISTFT
from utilities.terminal import loop_progress
from plotting.threedim import set_axes_equal
from plotting.exports import makegif
from general.frequency import noctfr
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\03_signal_gen\\01_acoustic_scenes')))
from rimPypack.rimPy import rimPy
#
from MWFpack import spatial

def MWF(y,Fs,win,L,R,voiceactivity,beta,min_covUpdates,useGEVD=False,GEVDrank=1,desired=None,SPP_thrs=0.5,MWFtype='online'):
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
    # -voiceactivity [Nt*1 binary vector /or/ Nt*Nf matrix] - Voice Activity Detector or Speech Presence Probability.
    # -beta [float [[0,1]], -] - Covariance matrices exponential average constant.
    # -min_covUpdates [int, -] - Minimum # of covariance matrices updates
    #                            before first filter weights update.
    # -useGEVD [bool] - If true, use the GEVD, do not otherwise (standard MWF).
    # -GEVDrank [int, -] - GEVD rank approximation (default: 1).
    # -SPP_thrs [float [[0,1]], -] - SPP threshold above which speech is considered present.
    # -MWFtype [str] - If 'batch', compute the covariance matrices from the entire signal (AND DO NOT USE GEVD).
                     # If 'online', compute the covariance matrices iteratively (possibly using GEVD).
    # >>> Outputs:
    # -D_hat [Nt*Nf*J (complex) float tensor, -] - Estimated desired signals (STFT domain). 

    # (c) Paul Didier - 14-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Useful parameters
    nframes = int(np.floor(y.shape[0]/(L-R))) - 1   # Number of necessary time frames to cover the whole signal
    nbins = int(L/2)+1                              # Number of frequency bins
    flagSPP = len(voiceactivity.shape) == 2         # Flag to know that we will be using a SPP and not a time-domain VAD
    
    # Compute STFT
    print('Computing STFTs of sensor observations with %i frames and %i bins...' % (nframes,nbins))    
    y_STFT, freqs = calcSTFT(y, Fs, win, L, R, 'onesided')
    if desired is not None:
        d_STFT = calcSTFT(desired, Fs, win, L, R, 'onesided')[0]

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
    # TEMPORARY
    sig_export = np.zeros_like(D_hat)
    
    if MWFtype == 'batch':
        if desired is None:
            raise ValueError('Cannot compute exact MWF without knowledge of the desired signal.')
        for kp in range(nbins):
            print('Using the given desired signal to compute the exact MWF - bin %i/%i.' % (kp+1, nbins))
            D_hat[kp,:,:], W_hat[:,:,kp] = get_exact_MWF(y_STFT[kp,:,:], d_STFT[kp,:,:])
    elif MWFtype == 'online':
        # Loop over time frames
        for l in range(nframes):
            t0 = time.time()    

            if not flagSPP:
                # Time-frame samples' indices
                idxChunk = np.arange(l*(L - R), np.amin([l*(L - R) + L, y.shape[0]]), dtype=int)
                # Current frame VAD (majority of speech-active or speech-inactive samples?)
                VAD_l[l] = np.count_nonzero(voiceactivity[idxChunk]) > len(idxChunk)/2
            
            # Loop over frequency bins
            for kp in range(nbins):

                Ytf = np.squeeze(y_STFT[kp,l,:]) # Current TF bin
                
                # Speech activity detection (VAD (time-domain) or SPP (STFT-domain))
                if not flagSPP:
                    speech_now = VAD_l[l]
                else:
                    speech_now = voiceactivity[kp,l] >= SPP_thrs

                if speech_now:        # "speech + noise" time frame
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
                        sig_export[kp,l,:] = np.diag(Sigma_yy[:,:,kp])# TMP --------------
                        W_hat[:,:,kp] = update_w_GEVDMWF(Sigma_yy[:,:,kp], Qmat[:,:,kp], GEVDrank)          # LMMSE weights
                    else:
                        W_hat[:,:,kp] = update_w_MWF(np.squeeze(Ryy[:,:,kp]), np.squeeze(Rnn[:,:,kp]))      # LMMSE weights

                # Desired signal estimates for each node separately (last dimension of <D_hat>)
                D_hat[kp,l,:] = np.squeeze(W_hat[:,:,kp]).conj().T @ Ytf

            t1 = time.time()
            if l % 10 == 0:
                print('Processed time frame %i/%i in %2f s...' % (l,nframes,t1-t0))

    print('MW-filtering done.')

    # # TEMPORARY
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,3)
    # plottype = ''
    # plottype = 'norm'
    # # tmp = np.std(sig_export, axis=-1)
    # for ii in range(6):
    #     # Current subplot indexing
    #     iax = int(np.ceil((ii+1)/3))-1
    #     jax = ii % 3
    #     # Plot
    #     if plottype != 'norm':
    #         tmp = np.real(sig_export[:,:,ii])
    #         mapp = ax[iax,jax].imshow(20*np.log10(tmp), vmin=0, vmax=70)
    #         ax[iax,jax].set(title='%i$^\mathrm{th}$ largest EVL [dB-scale]' % (ii+1))
    #     else:
    #         tmp = np.real(sig_export[:,:,ii]) / np.real(sig_export[:,:,0])
    #         tmp[tmp == np.nan] = 0
    #         mapp = ax[iax,jax].imshow(tmp, vmin=0, vmax=1)
    #         ax[iax,jax].set(title='%i$^\mathrm{th}$ largest EVL' % (ii+1))
    #     ax[iax,jax].invert_yaxis()
    #     ax[iax,jax].set_aspect('auto')
    #     # ax[iax,jax].grid()
    #     fig.colorbar(mapp, ax=ax[iax,jax])
    #     if ii > 2:
    #         ax[iax,jax].set(xlabel='Frame index $l$')
    #     if ii == 0 or ii == 3:
    #         ax[iax,jax].set(ylabel='Freq. bin index $\kappa$')
    # if plottype != 'norm':
    #     plt.suptitle('$\{\hat{\mathbf{R}}_\mathbf{yy},\hat{\mathbf{R}}_\mathbf{nn}\}$-GEVLs')
    # else:
    #     plt.suptitle('$\{\hat{\mathbf{R}}_\mathbf{yy},\hat{\mathbf{R}}_\mathbf{nn}\}$-GEVLs, normalized to largest GEVL')
    # #     plt.savefig('GEVD_EVLs_norm.png')
    # #     plt.savefig('GEVD_EVLs.png')
    # plt.show()

    stop = 1

    return D_hat, W_hat, freqs

def SNRout(w,Rxx,Rnn):
    # Derives output SNR for current TF-bin (see equation 2.59 in Randy's thesis)
    a = w.conj().T @ Rxx @ w
    b = w.conj().T @ Rnn @ w
    snr = a @ np.linalg.pinv(b)
    return snr


# @njit
def get_exact_MWF(Yf, Df):
    # Computes the exact MWF
    
    # Check dimensions of input arrays
    transposed = False
    if Yf.shape[0] > Yf.shape[1]:
        Yf = Yf.T
        transposed = True
    if Df.shape[0] > Df.shape[1]:
        Df = Df.T

    Ryy = 1/Yf.shape[1] * Yf @ Yf.conj().T  # Sensor signals autocorrelation
    Rss = 1/Df.shape[1] * Df @ Df.conj().T  # Desired signals autocorrelation
    Wfilt = np.linalg.solve(Ryy, Rss)
    Dout = Wfilt.conj().T @ Yf              # Apply filter to sensor signals

    if transposed:
        Dout = Dout.T

    return Dout, Wfilt


@njit
def expavg_covmat(Ryy, beta, Y):
    return beta*Ryy + (1 - beta)*np.outer(Y, Y.conj())

@njit
def update_w_MWF(Ryy,Rnn):
    # Estimate speech covariance matrix
    Rss = Ryy - Rnn
    # LMMSE weights
    return np.linalg.pinv(Ryy) @ Rss  # eq.(19) in ruiz2020a

@njit
def update_w_GEVDMWF(S,Q,GEVDrank):
    # Estimate speech covariance matrix
    sig_yy = np.diag(S)
    diagveig = np.ones(GEVDrank) - np.ones(GEVDrank) / sig_yy[:GEVDrank]   # rank <R> approximation
    diagveig = np.append(diagveig, np.zeros(S.shape[0] - GEVDrank))
    # LMMSE weights
    What = np.linalg.pinv(Q.conj().T) @ np.diag(diagveig) @ Q.conj().T
    return What

@njit
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


def spatial_visu_MWF(W,freqs,rd,alpha,r,Fs,rir_dur,targetSources=None,noiseSources=None,\
    exportit=False,exportname='',noise_spatially_white=False, stoi_imp=None, fwSNRseg_imp=None):
    # Compute the MWF output as function of frequency and spatial location.

    print('\nComputing spatial visualization of MWF effect...\n')

    # Check inputs
    if len(W.shape) == 2:
        if len(freqs) > 1:
            raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        W = W[np.newaxis, :, :]    # Single frequency bin case
    if W.shape[0] != len(freqs):
        raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        
    # ------------- HARD CODED PARAMETERS -------------
    # RIR parameters
    refCoeff = -1*np.sqrt(1 - alpha)       # Reflection coefficient 
    # Grid
    gridres = 0.1   # Grid spatial resolution [m]
    # gridres = 2   # Grid spatial resolution [m]
    # -------------------------------------------------

    # Gridify room or room-slice
    xx,yy,zz = spatial.gridify(rd, targetSources[:,-1], gridres, plotgrid=False)
    print('Computing over %i 2-D z-slice(s)...' % xx.shape[-1])

    # Get energy over grid
    en, en_pf, freqs = spatial.getenergy(xx,yy,zz,rir_dur,Fs,r,rd,refCoeff,W,bins_of_interest=freqs) 
    # ~~~~~~~~~~ PLOT ~~~~~~~~~~
    spatial.plotspatialresp(xx,yy,en,targetSources,r,\
        dBscale=1,exportit=exportit,exportname=exportname,\
            freqs=freqs,noiseSource=noiseSources,multichannel=True,noise_spatially_white=noise_spatially_white,\
                stoi_imp=stoi_imp, fwSNRseg_imp=fwSNRseg_imp)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    return 0


def applyMWF_tdomain(y, W_hat, Fs,win,L_fft,R_fft):
    # Applies a MWF derived in the frequency domain to a signal in the time-domain.

    y_STFT = calcSTFT(y, Fs, win, L_fft, R_fft, 'onesided')[0]

    nbins = y_STFT.shape[0]
    nframes = y_STFT.shape[1]
    
    D_hat = np.zeros_like(y_STFT)
    # Loop over time frames
    for l in range(nframes):
        # Loop over frequency bins
        for kp in range(nbins):
            Ytf = np.squeeze(y_STFT[kp,l,:]) # Current TF bin
            # Desired signal estimates for each node separately (last dimension of <D_hat>)
            D_hat[kp,l,:] = np.squeeze(W_hat[:,:,kp]).conj().T @ Ytf

    d_hat = calcISTFT(D_hat, win, L_fft, R_fft, sides='onesided')
    # Zero-pad if needed to get the same output signal length
    if d_hat.shape[0] < y.shape[0]:
        d_hat = np.concatenate((d_hat, np.zeros((y.shape[0]-d_hat.shape[0], d_hat.shape[1]))), axis=0)

    return d_hat

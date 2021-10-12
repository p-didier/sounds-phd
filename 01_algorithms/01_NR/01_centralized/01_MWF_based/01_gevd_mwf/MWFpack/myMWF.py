import numpy as np
from numpy.lib.shape_base import column_stack
import scipy.signal as sig
import scipy
import time
from numba import jit
import sys, os
from sklearn import preprocessing
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT
from utilities.terminal import loop_progress
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\03_signal_gen\\01_acoustic_scenes')))
from rimPypack.rimPy import rimPy

def MWF(y,Fs,win,L,R,voiceactivity,beta,min_covUpdates,useGEVD=False,GEVDrank=1,desired=None,SPP_thrs=0.5):
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
    #
    # TEMPORARY
    sig_export = np.zeros_like(D_hat)

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
    # # plottype = 'norm'
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
    What = np.linalg.pinv(Q.conj().T) @ np.diag(diagveig) @ Q.conj().T
    return What

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


def spatial_visu_MWF(W,freqs,rd,alpha,r,Fs,win,L,R,targetSources=None,noiseSources=None):
    # Compute the MWF output as function of frequency and spatial location.

    # Check inputs
    if len(W.shape) == 2:
        if len(freqs) > 1:
            raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        W = W[:, :, np.newaxis]    # Single frequency bin case
    if W.shape[-1] != len(freqs):
        raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        
    # ------------- HARD CODED PARAMETERS -------------
    # RIR generation parameters
    rir_dur = 2**10 / Fs                # Duration [s]
    refCoeff = -1*np.sqrt(1 - alpha)    # Reflection coefficient 
    # Signal duration
    Tsig = 0.5        # [s]
    # Mesh
    gridres = 0.25   # Grid spatial resolution [m]
    # -------------------------------------------------

    # Gridify room or room-slice
    x_ = np.linspace(0,rd[0],num=int(np.round(rd[0]/gridres)))
    y_ = np.linspace(0,rd[1],num=int(np.round(rd[1]/gridres)))
    if targetSources is not None:
        z_ = targetSources[:,-1]  # Set the z-coordinates of the target sources as slice heights
        z_ = z_[:,np.newaxis]
        print('%.1f x %.1f x %.1f m^3 room gridified by slices every %.1f m,\nresulting in %i possible source locations on %i 2D planes.' % (rd[0],rd[1],rd[2],gridres,len(x_)*len(y_)*len(z_),len(z_)))
    else:
        z_ = np.linspace(0,rd[2],num=int(np.round(rd[2]/gridres)))
        print('%.1f x %.1f x %.1f m^3 room gridified every %.1f m,\nresulting in %i possible source locations across the 3D space.' % (rd[0],rd[1],rd[2],gridres,len(x_)*len(y_)*len(z_)))
    xx,yy,zz = np.meshgrid(x_, y_, z_, indexing='ij')

    # Make raw signal
    raw = np.random.uniform(low=-1.0, high=1.0, size=(int(Tsig*Fs),))
    raw = preprocessing.scale(raw)   # normalize
    raw = raw[:, np.newaxis]       # make 2-dimensional

    # Loop over grid points
    magout = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],len(freqs),r.shape[0]))      # Full volume

    for ii in range(xx.shape[0]):
        for jj in range(xx.shape[1]):
            for kk in range(xx.shape[2]):
                # Source location (grid point)
                r0 = [xx[ii,jj,kk], yy[ii,jj,kk], zz[ii,jj,kk]]
                # Compute filter output energy
                magout[ii,jj,kk,:,:] = compute_filter_output(r, r0, rd, refCoeff, rir_dur, Fs, raw, win, L, R, W)
                # Monitor loop
                progress_percent = loop_progress([ii,jj,kk],xx.shape)
                print('Computing filter output energy for source at (%.1f,%.1f,%.1f) in room [%i%% done]...' % (r0[0],r0[1],r0[2],progress_percent))

    # PLOT PLOT PLOT
    plotsensor = 2
    plotfreq = 200
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if targetSources is not None:       # Slice
        for ii in range(magout.shape[2]):
            ax.contourf(xx[:,:,ii], yy[:,:,ii], magout[:,:,ii,plotfreq,plotsensor]/np.amax(magout[:,:,:,plotfreq,plotsensor]), offset=z_[ii], zdir='z', alpha=0.5)
        for ii in range(targetSources.shape[0]):
            ax.scatter(targetSources[ii,0],targetSources[ii,1],targetSources[ii,2],c='blue')
        for ii in range(noiseSources.shape[0]):
            ax.scatter(noiseSources[ii,0],noiseSources[ii,1],noiseSources[ii,2],c='red')
        for ii in range(r.shape[0]):
            ax.scatter(r[ii,0],r[ii,1],r[ii,2],c='green')
    else:                               # Full volume
        for ii in range(xx.shape[0]):
            for jj in range(xx.shape[1]):
                for kk in range(xx.shape[2]):
                    ax.scatter(xx[ii,jj,kk], yy[ii,jj,kk], zz[ii,jj,kk], c=str(magout[ii,jj,kk,plotfreq,plotsensor]/np.amax(magout[:,:,:,plotfreq,plotsensor])), marker="o")
        for ii in range(r.shape[0]):
            ax.scatter(r[ii,0],r[ii,1],r[ii,2],c='red')
    ax.set(title='f = %.1f Hz' % freqs[plotfreq])
    #
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(freqs,magout[2,2,2,:,plotsensor])

    return 0


def compute_filter_output(r, r0, rd, refCoeff, rir_dur, Fs, raw, win, L, R, W):
    # Compute MWF filter output energy for a certain combination of source/receivers positions. 

    # Get transfer functions
    h = rimPy(r, r0, rd, refCoeff, rir_dur, Fs)

    # Generate sensors signal 
    y = sig.fftconvolve(raw, h, axes=0)

    # Get STFT
    y_STFT = calcSTFT(y, Fs, win, L, R, 'onesided')[0]
    y_FFT = np.fft.fft(y, axis=0)

    # fig, ax = plt.subplots()
    # ax.plot(20*np.log10(np.abs(y_FFT[:,0])))
    # ax.grid()
    # plt.show()

    # Apply filter to each TF bin
    magout = np.zeros((y_STFT.shape[0], y_STFT.shape[2]))
    for kp in range(y_STFT.shape[0]):
        z_kp = np.zeros((y_STFT.shape[1],y_STFT.shape[2]), dtype=complex)
        for l in range(y_STFT.shape[1]):
            datacurr = np.squeeze(y_STFT[kp,l,:])    
            z_kp[l,:] = np.squeeze(W[:,:,kp]).conj().T @ datacurr
        # Get time-averaged filter-output magnitude
        magout[kp,:] = np.mean(np.abs(z_kp)**2, axis=0)
        
    return magout

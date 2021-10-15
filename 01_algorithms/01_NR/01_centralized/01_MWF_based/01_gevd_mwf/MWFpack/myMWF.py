import pathlib
import numpy as np
import scipy.signal as sig
import scipy
import time
from numba import njit
import sys, os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT
from utilities.terminal import loop_progress
from plotting.threedim import plot_room, set_axes_equal
from playsounds.playsounds import playthis
from plotting.exports import makegif
from general.frequency import noctfr
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\03_signal_gen\\01_acoustic_scenes')))
from rimPypack.rimPy import rimPy

def MWF(y,Fs,win,L,R,voiceactivity,beta,min_covUpdates,useGEVD=False,GEVDrank=1,desired=None,SPP_thrs=0.5,exact=False):
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
    # -exact [bool] - If true, use the exact MWF with the true covariance matrices, not their approximations (overrides everything else).
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
    
    if exact:
        if desired is None:
            raise ValueError('Cannot compute exact MWF without knowledge of the desired signal.')
        for kp in range(nbins):
            print('Using the given desired signal to compute the exact MWF - bin %i/%i.' % (kp+1, nbins))
            D_hat[kp,:,:], W_hat[:,:,kp] = get_exact_MWF(y_STFT[kp,:,:], d_STFT[kp,:,:])
    else:
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


def spatial_visu_MWF(W,freqs,rd,alpha,r,Fs,win,L,R,targetSources=None,noiseSources=None):
    # Compute the MWF output as function of frequency and spatial location.

    print('\nComputing spatial visualization of MWF effect...\n')

    # Check inputs
    if len(W.shape) == 2:
        if len(freqs) > 1:
            raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        W = W[:, :, np.newaxis]    # Single frequency bin case
    if W.shape[-1] != len(freqs):
        raise ValueError('The frequency vector should contain as many elements as the -1 dimension of <W>')
        
    # ------------- HARD CODED PARAMETERS -------------
    Tsig = 0.25        # Signal duration [s]
    # RIR generation parameters
    rir_dur = np.amin([2**12 / Fs, Tsig])  # Duration [s]
    refCoeff = -1*np.sqrt(1 - alpha)       # Reflection coefficient 
    # Mesh
    gridres = 0.25   # Grid spatial resolution [m]
    # gridres = 2   # Grid spatial resolution [m]
    # -------------------------------------------------

    # Gridify room or room-slice
    x_ = np.linspace(0,rd[0],num=int(np.round(rd[0]/gridres)))
    y_ = np.linspace(0,rd[1],num=int(np.round(rd[1]/gridres)))
    if targetSources is not None:
        z_ = targetSources[:,-1]  # Set the z-coordinates of the target sources as slice heights
        z_ = z_[:,np.newaxis]
        print('%.1f x %.1f x %.1f m^3 room gridified in %i slices every %.1f m,\nresulting in %i possible source locations.' % (rd[0],rd[1],rd[2],len(z_),gridres,len(x_)*len(y_)*len(z_)))
    else:
        z_ = np.linspace(0,rd[2],num=int(np.round(rd[2]/gridres)))
        print('%.1f x %.1f x %.1f m^3 room gridified every %.1f m,\nresulting in %i possible source locations across the 3D space.' % (rd[0],rd[1],rd[2],gridres,len(x_)*len(y_)*len(z_)))
    xx,yy,zz = np.meshgrid(x_, y_, z_, indexing='ij')

    # # # TEMPORARY
    # x_ = r[:,0]  # Set the z-coordinates of the target sources as slice heights
    # x_ = x_[:,np.newaxis] + 0.05 * np.ones((r.shape[0],1))
    # y_ = r[:,1]  # Set the z-coordinates of the target sources as slice heights
    # y_ = y_[:,np.newaxis] + 0.05 * np.ones((r.shape[0],1))
    # z_ = r[:,2]  # Set the z-coordinates of the target sources as slice heights
    # z_ = z_[:,np.newaxis] + 0.05 * np.ones((r.shape[0],1))
    # xx,yy,zz = np.meshgrid(x_, y_, z_, indexing='ij')

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
                progress_percent = loop_progress([ii,jj,kk], xx.shape)
                print('Computing filter output energy for source at (%.2f,%.2f,%.2f) in room [%.2f %%]...' % (r0[0],r0[1],r0[2],progress_percent))

    # ~~~~~~~~~~ PLOT ~~~~~~~~~~
    plot_spatial_visu_MWF(xx,yy,zz,magout,freqs,targetSources,noiseSources,r,makeGIF=False,freqAvType='OTOB')
    plot_spatial_visu_MWF(xx,yy,zz,magout,freqs,targetSources,noiseSources,r,makeGIF=False,freqAvType='all')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    return 0


def compute_filter_output(r, r0, rd, refCoeff, rir_dur, Fs, raw, win, L, R, W):
    # Compute MWF filter output energy for a certain combination of source/receivers positions. 

    # Get transfer functions
    h = rimPy(r, r0, rd, refCoeff, rir_dur, Fs)

    # TEMPORARY - NORMALIZE ALL H's
    h /= np.sum(h, axis=0)

    # Generate sensors signal 
    y = sig.fftconvolve(raw, h, axes=0)

    # Get STFT
    y_STFT,f = calcSTFT(y, Fs, win, L, R, 'onesided')

    nSensors = y_STFT.shape[2]
    nframes = y_STFT.shape[1]
    nbins = len(f)

    # Apply filter to each TF bin
    magout = np.zeros((nbins, nSensors))
    magout_justz = np.zeros((nbins, nSensors))
    magout_justy = np.zeros((nbins, nSensors))
    z = np.zeros_like(y_STFT, dtype=complex)
    for kp in range(nbins):
        for l in range(nframes):
            z[kp,l,:] = np.squeeze(W[:,:,kp]).conj().T @ np.squeeze(y_STFT[kp,l,:])
        # Get time-averaged filter-output magnitude
        magout[kp,:] = np.mean(np.abs(z[kp,:,:])**2, axis=0) / np.mean(np.abs(y)**2, axis=0)

    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(f,magout_justy)
        ax.set_aspect('auto')
        ax.grid()
        ax.set(title='Power spectrum input $E\{|y|^2\}$')
        plt.ylim((0, np.amax(magout_justz)))
        ax = fig.add_subplot(212)
        ax.plot(f,magout_justz)
        ax.set_aspect('auto')
        ax.grid()
        ax.set(title='Power spectrum output $E\{|\hat{d}|^2\}$', xlabel='$f$ [Hz]')
        plt.ylim((0, np.amax(magout_justz)))
        plt.show()

    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        mapp = ax.imshow(np.abs(y_STFT[:,:,0])**2)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.grid()
        plt.colorbar(mapp)
        ax.set(title = 'STFT($y_1(\mathbf{r}_0)$)')
        ax = fig.add_subplot(122)
        mapp = ax.imshow(np.abs(z[:,:,0])**2)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.grid()
        ax.set(title = 'STFT($\hat{d}_1(\mathbf{r}_0) = \mathbf{w}_1^Hy_1(\mathbf{r}_0)$)')
        plt.colorbar(mapp)
        plt.show()
        
    return magout


def plot_spatial_visu_MWF(xx,yy,zz,magout,freqs,targetSources,noiseSources,r,makeGIF=False,freqAvType='all'):

    # Export name (HARD-CODED)
    exportname = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\test' % (os.getcwd())

    fig = plt.figure(figsize=(8,6))
    if makeGIF:
        angles = np.linspace(0, 360, len(freqs))
        for idx_gif in range(len(freqs)):
            # Choose frequency
            plotfreq = idx_gif
            for plotsensor in range(magout.shape[-1]):
                plot_subfct(fig,xx,yy,zz,magout,targetSources,noiseSources,r,plotfreq,plotsensor,rotate=True,anglerot=angles[idx_gif])
            plt.suptitle('f = %.1f Hz' % freqs[plotfreq])
            plt.tight_layout()
            fname = '%s_%i.png' % (exportname, int(idx_gif))
            plt.savefig(fname, bbox_inches='tight')
            fig.clear()
        # MAKE GIF
        giffolder = Path(fname).parent
        gifname = Path(exportname).with_suffix('').stem
        makegif(giffolder, gifname)
        print('Spatial MWF output as fct of frequency exported as GIF in:\n"%s\\%s.gif"' % (giffolder, gifname))

    # AVERAGE OVER FREQUENCIES
    fc, fl, fu = noctfr(n=3, fll=freqs[1], ful=freqs[-1], type='exact')
    if freqAvType == 'all':
        fc, fl, fu = [0], [freqs[0]], [freqs[-1]]   # Consider all frequencies
    for idxOTOB in range(len(fc)):
        # Select appropriate data chunk
        idxfreq = [ii for ii in range(len(freqs)) if freqs[ii] >= fl[idxOTOB] and freqs[ii] <= fu[idxOTOB]]
        if len(idxfreq) > 0:
            magout_curr = magout[:,:,:,idxfreq,:]  # Extract only relevant info for current "band"
            for plotsensor in range(magout.shape[-1]):
                plot_subfct(fig,xx,yy,zz,20*np.log10(np.abs(magout_curr)),targetSources,noiseSources,r,\
                    np.arange(magout_curr.shape[3]),plotsensor)
            if freqAvType == 'OTOB':
                plt.suptitle('Average 1/3-octave band centered on %i Hz (%i to %i Hz)' % (fc[idxOTOB], fl[idxOTOB], fu[idxOTOB]))
            elif freqAvType == 'all':
                plt.suptitle('Average over all frequencies (%i to %i Hz)' % (fl[idxOTOB], fu[idxOTOB]))
            plt.tight_layout()
            if freqAvType == 'all':
                fname = '%s_allf.png' % exportname  
            elif freqAvType == 'OTOB':
                fname = '%s_OTOBfc%iHz.png' % (exportname, int(np.round(fc[idxOTOB])))   
            plt.savefig(fname, bbox_inches='tight')
            fig.clear()
        else:
            print('No frequency bins in band centered on %.2f Hz. Skipping band.' % (fc[idxOTOB]))
    plt.show()

    return None


def plot_subfct(fig,xx,yy,zz,magout,targetSources,noiseSources,r,plotfreq,plotsensor,rotate=False,anglerot=0):

    # Generate sub-plot axes
    if magout.shape[-1] > 2:
        ax = fig.add_subplot(2, np.ceil(magout.shape[-1]/2), plotsensor+1, projection='3d')
    else:
        ax = fig.add_subplot(1, magout.shape[-1], plotsensor+1, projection='3d')
    
    # Rotate
    if rotate:
        ax.view_init(35, anglerot)   
    if (r[:,-1] == targetSources[0,-1]).all():
        ax.view_init(90, 0)

    if targetSources is not None:       # Slice
        for ii in range(magout.shape[2]):
            ax.contourf(xx[:,:,ii], yy[:,:,ii],\
                 np.abs(np.mean(magout[:,:,ii,plotfreq,plotsensor], axis=2)/np.amax(magout[:,:,:,plotfreq,plotsensor])),\
                      offset=zz[0,0,ii], zdir='z', alpha=0.5)
        for ii in range(targetSources.shape[0]):
            ax.scatter(targetSources[ii,0],targetSources[ii,1],targetSources[ii,2],c='blue')
        for ii in range(noiseSources.shape[0]):
            ax.scatter(noiseSources[ii,0],noiseSources[ii,1],noiseSources[ii,2],c='red')
        for ii in range(r.shape[0]):
            if ii == plotsensor:
                ax.scatter(r[ii,0],r[ii,1],r[ii,2],c='green',edgecolors='black')
            else:
                ax.scatter(r[ii,0],r[ii,1],r[ii,2],c='green')
    else:                               # Full volume
        for ii in range(xx.shape[0]):
            for jj in range(xx.shape[1]):
                for kk in range(xx.shape[2]):
                    ax.scatter(xx[ii,jj,kk], yy[ii,jj,kk], zz[ii,jj,kk], c=str(magout[ii,jj,kk,plotfreq,plotsensor]/np.amax(magout[:,:,:,plotfreq,plotsensor])), marker="o")
        for ii in range(r.shape[0]):
            ax.scatter(r[ii,0],r[ii,1],r[ii,2],c='red')
    # if alpha < 1:
    #     plot_room(ax, rd)       # plot room boundaries
    ax.set(title='Sensor #%i' % (plotsensor+1)) # title
    if (r[:,-1] == targetSources[0,-1]).all():
        ax.set_xticks([])       # get rid of x-axis ticks
        ax.set_yticks([])       # get rid of y-axis ticks
        ax.set_zticks([])       # get rid of z-axis ticks
    else:
        set_axes_equal(ax)      # set axes equal

    return None
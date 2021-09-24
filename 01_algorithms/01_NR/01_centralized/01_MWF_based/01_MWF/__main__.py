# Standard library imports
from operator import index, xor
import numpy as np
import os, time
from numpy.core.defchararray import title
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import cProfile
from pstats import Stats, SortKey
import scipy.signal as sig
import pandas as pd
# Third party imports
#
# Local application imports
from MWFpack import myMWF, sig_gen, VAD

# Classic Multichannel Wiener Filter (MWF) for noise reduction, 
# using oracle VAD and full communication (i.e. a fusion center for all nodes data).
#
# (c) Paul Didier - 08-Sept-2021
# SOUNDS ETN - KU Leuven ESAT STADIUS

SAVEFIGS = False

def run_script():

    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % os.getcwd()  # path to acoustic scenarios
    speech_in = 'libri'     # name of speech signals library to be used
    Tmax = 10               # maximum signal duration [s]
    noise_type = 'white'    # type of noise to be used
    baseSNR = 10            # SNR pre-RIR application [dB]
    pauseDur = 1            # duration of pauses in-between speech segments [s]
    pauseSpace = 1          # duration of speech segments (btw. pauses) [s]
    # ----- Acoustic scenario and speech signal specific selection
    # ASref = 'AS9_J5_Ns1_Nn1'  # acoustic scenario (if empty, random selection)
    ASref = 'testAS'  # acoustic scenario (if empty, random selection)
    ASref = 'testAS_anechoic'  # acoustic scenario (if empty, random selection)
    # speech = ''                    # speech signals (if empty, random selection)
    speech1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0000.flac'
    speech2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0001.flac'
    speech = [speech1,speech2]
    noise1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise1.wav'
    noise2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise2.wav'
    noise3 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise3.wav'
    noise = [noise1,noise2,noise3]

    # STFT
    L = 2**9              # Time frame length [samples]
    R = L/2                # Inter-frame overlap length [samples]
    win = np.hanning(L)    # STFT time window

    # Covariance estimates
    beta = 1 - 1/16e3       # autocorrelation matrices time-avg. update constant
    min_cov_updates = 10    # min. number of covariance matrices updates before 1st filter weights update

    # VAD
    tw = 40e-3              # window length [s]
    ref_sensor = 0          # index of reference sensor
    VAD_fact = 400          # VAD threshold factor w.r.t. max(y**2)

    # I) Generate microphone signals
    print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref)
    y,ds,ny,t,Fs = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                            pauseDur,pauseSpace,ASref,speech,noise)
    print('Microphone signals created using "%s"' % ASref)

    # Set useful data as variables
    J = y.shape[-1]

    # II) Oracle VAD
    thrs_E = np.amax(ds[:,ref_sensor]**2)/VAD_fact  
    print('\nComputing oracle VAD from clean speech signal...')
    myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
    print('Oracle VAD computed')

    # III) Compute MWF 
    t0 = time.time()
    D_hat = myMWF.MWF(y,Fs,win,L,R,myVAD,beta,min_cov_updates)
    t1 = time.time()
    print('Enhanced signals (%is x %i sensors) computed in %3f s' % (Tmax,J,t1-t0))
    # Get corresponding time-domain signal(s)
    d_hat = np.zeros_like(y)
    for ii in range(J):
        ts = sig.istft(np.squeeze(D_hat[:,:,ii]),fs=Fs,window=win,nperseg=L,noverlap=R,input_onesided=True)[-1]
        d_hat[:,ii] = ts[:y.shape[0]]       # Get rid of last overlap

    # # TMP - Export data for comparison with MATLAB implementation
    # for ii in range(D_hat.shape[-1]):
    #     dfr = pd.DataFrame(np.squeeze(np.real(D_hat[:,:,ii])))
    #     dfi = pd.DataFrame(np.squeeze(np.imag(D_hat[:,:,ii])))
    #     dfr.to_csv('DhatPy_sensor%i_real.csv' % ii)
    #     dfi.to_csv('DhatPy_sensor%i_imag.csv' % ii)

    # ------------ QUICK PLOT COMMANDS ------------ 
    # fig, ax = plt.subplots()
    # ax.imshow(20*np.log10(np.abs(np.squeeze(D_hat[:,:,0]))), extent=[0, t[-1],Fs/2, 0])        
    # ax.invert_yaxis()
    # ax.set_aspect('auto')
    # ax.grid()
    # plt.show()
    # ------------ ------------ ------------ -------

    # IV) SNR improvement estimates
    SNRd_hat = np.zeros(J)
    SNRimp = np.zeros(J)
    SNRy = np.zeros(J)
    for k in range(J):
        SNRd_hat[k] = VAD.SNRest(d_hat[:,k],myVAD)
        SNRy[k] = VAD.SNRest(y[:,k],myVAD)
        SNRimp[k] = SNRd_hat[k] - SNRy[k]

    print('Raw sensor SNRs [dB]:')
    print(SNRy)
    print('Post-processed sensor SNRs [dB]:')
    print(SNRd_hat)
    print('SNR improvements (per sensor, in [dB]):')
    print(SNRimp)

    # Visualize microphone signals before/after MWF enhancement
    if 1:
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(12, 4.5, forward=True)   # figure size
        vminn = -120.0
        vmaxx = -31.0

        # Y = sig.stft(y[:,ii], fs=Fs, nperseg=L, nfft=Nfft)[2]
        Y = sig.stft(y[:,ref_sensor], fs=Fs, nperseg=L, noverlap=R, return_onesided=True)[2]
        Ds = sig.stft(ds[:,ref_sensor], fs=Fs, nperseg=L, noverlap=R, return_onesided=True)[2]
        # Raw microphone signal
        mapp = ax[0].imshow(20*np.log10(np.abs(Y)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[0].invert_yaxis()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel='$t$ [s]', ylabel='$f$ [kHz]',
            title='$y_%i(t,f)$' % (ref_sensor+1))
        ax[0].axes.yaxis.set_ticks(np.arange(0,Fs/2+1e3,1e3))
        ax[0].axes.yaxis.set_ticklabels([str(ii) for ii in np.arange(0,Fs/1e3/2 + 1)])
        # Raw microphone signal
        mapp = ax[1].imshow(20*np.log10(np.abs(np.squeeze(D_hat[:,:,ref_sensor]))),\
             extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[1].invert_yaxis()
        ax[1].set_aspect('auto')
        ax[1].set(xlabel='$t$ [s]',
            title='$\hat{d}_%i(t,f)$' % (ref_sensor+1))
        ax[1].axes.yaxis.set_ticklabels([])
        # Raw microphone signal
        mapp = ax[2].imshow(20*np.log10(np.abs(Ds)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[2].invert_yaxis()
        ax[2].set_aspect('auto')
        ax[2].set(xlabel='$t$ [s]',
            title='$d_%i(t,f)$' % (ref_sensor+1))
        ax[2].axes.yaxis.set_ticklabels([])
        fig.colorbar(mapp, ax=ax[2])

        plt.suptitle('Centralized MWF noise reduction -- $\beta$ = %2f' % beta)

        if SAVEFIGS:
            plt.savefig('%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_MWF\\00_figs\\01_for_20210916meeting\\STFTs_%ip_ev%i_%s.pdf' % (os.getcwd(),pauseDur,pauseSpace,ASref))
            plt.show()

    # Bar chart for SNR improvements at each sensor
    if 1:
        barw = 0.3
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 3, forward=True)   # figure size
        b1 = ax.bar(np.arange(1,J+1)-barw/2, height=SNRy, width=barw)
        b2 = ax.bar(np.arange(1,J+1)+barw/2, height=SNRd_hat, width=barw)
        plt.legend([b1,b2],['SNR($y_k$)','SNR($\hat{d}_k$)'])
        ax.set(title='Average SNR improvement: %.1f dB' % (np.round(np.mean(SNRimp),1)))
        
        if SAVEFIGS:
            plt.savefig('%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_MWF\\00_figs\\01_for_20210916meeting\\barchart_%ip_ev%i_%s.pdf' % (os.getcwd(),pauseDur,pauseSpace,ASref))
            plt.show()

run_script()
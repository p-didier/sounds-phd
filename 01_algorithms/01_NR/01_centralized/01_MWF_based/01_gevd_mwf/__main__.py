# Standard library imports
from operator import index, xor
import numpy as np
import os, time, sys
from numpy.core.defchararray import title
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import scipy.signal as sig
import pandas as pd
# Third party imports
#
# Local application imports
from MWFpack import myMWF, sig_gen, VAD
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT, calcISTFT
import playsounds.playsounds as ps

# Multichannel Wiener Filter (MWF) for noise reduction, using Generalized Eigenvalue Decomposition (GEVD)
# with oracle VAD and full communication (i.e. a fusion center for all nodes data).
#
# (c) Paul Didier - 08-Sept-2021
# SOUNDS ETN - KU Leuven ESAT STADIUS

# Global variables
SAVEFIGS = 0        # If true, saves figures as PNG and PDF files
EXPORTDATA = 0      # If true, exports I/O SNRs as CSV files
LISTEN_TO_MICS = 0  # If true, plays examples of target and raw mic signal
SHOW_WAVEFORMS = 0  # If true, plots waveforms of target and raw mic signals


def run_script():

    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % os.getcwd()  # path to acoustic scenarios
    speech_in = 'libri'     # name of speech signals library to be used
    noise_type = 'white'    # type of noise to be used
    #
    Tmax = 5               # maximum signal duration [s]
    baseSNR = 10            # SNR pre-RIR application [dB]
    #
    pauseDur = 1            # duration of pauses in-between speech segments [s]
    pauseSpace = 3          # duration of speech segments (btw. pauses) [s]
    #
    useGEVD = True          # if True, use GEVD, do not otherwise
    GEVDrank = 2

    # Exports
    # exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\onesource_reverberant' % os.getcwd()
    # exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\onesource_anechoic' % os.getcwd()
    exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\twosources_GEVDrank1' % os.getcwd()
    # exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\twosources_GEVDrank2' % os.getcwd()

    # ----- Acoustic scenario + specific speech/noise signal(s) selection
    # ASref = 'J5_Ns1_Nn1\\AS9'       # acoustic scenario (if empty, random selection)
    ASref = 'J5_Ns2_Nn3\\AS0'       # acoustic scenario (if empty, random selection)
    # ASref = 'J5_Ns2_Nn3\\testAS_anechoic'       # acoustic scenario (if empty, random selection)
    # ASref = ''                    # acoustic scenario (if empty, random selection)
    # speech = ''                    # speech signals (if empty, random selection)
    speech1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0000.flac'
    speech2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\3570\\5694\\3570-5694-0007.flac'
    speech = [speech1,speech2]
    noise1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise1.wav'
    noise2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise2.wav'
    noise3 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise3.wav'
    noise = [noise1,noise2,noise3]

    # STFT
    L = 2**9                        # Time frame length [samples]
    R = L/2                         # Inter-frame overlap length [samples]
    win = np.sqrt(np.hanning(L))    # STFT time window

    # Covariance estimates
    beta = 1 - 1/16e3       # autocorrelation matrices time-avg. update constant
    min_cov_updates = 10    # min. number of covariance matrices updates before 1st filter weights update

    # VAD
    tw = 40e-3              # window length [s]
    ref_sensor = 0          # index of reference sensor
    VAD_fact = 400          # VAD threshold factor w.r.t. max(y**2)

    # ----------------------------------------------------------------------------------------
    # -------------------------------------- PROCESSING --------------------------------------
    # ----------------------------------------------------------------------------------------

    # I) Generate microphone signals
    print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref)
    y,ds,ny,t,Fs,reftxt = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                            pauseDur,pauseSpace,ASref,speech,noise,plotAS=False)
    print('Microphone signals created using "%s"' % ASref)

    # Set useful data as variables
    J = y.shape[-1]

    # I.2) Checks on input parameters
    if not (Tmax*Fs/(L-R)).is_integer():
        print('WARNING: the chosen <Tmax>=%is conflicts with the frame length <L-R>=%i\n' % (Tmax, L-R))

    # II) Oracle VAD
    thrs_E = np.amax(ds[:,ref_sensor]**2)/VAD_fact  
    print('\nComputing oracle VAD from clean speech signal...')
    myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
    print('Oracle VAD computed')

    # II.2) Initial SNR estimates
    SNRy = np.zeros(J)
    for k in range(J):
        SNRy[k] = VAD.SNRest(y[:,k],myVAD)

    # II.3) Get STFTs
    ref_sensor = np.argmin(SNRy)
    Ymat = calcSTFT(y[:,ref_sensor], Fs, win, L, R, 'onesided')[0]
    Ds = calcSTFT(ds[:,ref_sensor], Fs, win, L, R, 'onesided')[0]

    if LISTEN_TO_MICS:
        # Listen
        sPbIdx = np.argmin(SNRy)
        maxplaytime = 5     # Maximal play-time in seconds
        print('Playing the target signal for sensor #%i...' % (sPbIdx+1))
        ps.playthis(ds[:int(maxplaytime*Fs),sPbIdx], Fs)
        print('Playing the raw mic. signal for sensor #%i...' % (sPbIdx+1))
        ps.playthis(y[:int(maxplaytime*Fs),sPbIdx], Fs)

    if SHOW_WAVEFORMS:
        # PLot
        fig, ax = plt.subplots()
        ax.plot(t, y[:,ref_sensor])
        ax.plot(t, ds[:,ref_sensor])
        ax.set(xlabel='time (s)',
            title='Waveforms')
        plt.legend('Raw mic signal', 'Target signal (clean)')
        ax.grid()
        plt.show()

    # III) Compute (GEVD-)MWF 
    t0 = time.time()
    D_hat = myMWF.MWF(y,Fs,win,L,R,myVAD,beta,min_cov_updates,useGEVD,GEVDrank)
    t1 = time.time()
    print('Enhanced signals (%is x %i sensors) computed in %3f s' % (Tmax,J,t1-t0))
    # Get corresponding time-domain signal(s)
    d_hat = calcISTFT(D_hat, win, L, R, 'onesided')

    # IV) SNR improvement estimates
    SNRd_hat = np.zeros(J)
    SNRimp = np.zeros(J)
    for k in range(J):
        SNRd_hat[k] = VAD.SNRest(d_hat[:,k],myVAD)
        SNRimp[k] = SNRd_hat[k] - SNRy[k]

    if EXPORTDATA:
        # Export SNRs in time domain
        npa = np.stack((SNRy,SNRd_hat,SNRimp))
        pd.DataFrame(npa, index=['y', 'dhat', 'imp']).to_csv('%s\\SNRs_%ip_ev%i__%s.csv' % (exportDir,pauseDur,pauseSpace,reftxt))

    print('Raw sensor SNRs (time-domain only) [dB]')
    print(SNRy)
    print('Post-processed sensor SNRs (time-domain only) [dB]:')
    print(SNRd_hat)
    print('SNR improvements (time-domain only) (per sensor, in [dB]):')
    print(SNRimp)

    if 0:
        # V) SNR estimates per bin
        Ns = 20       # Nr. of STFT frames in time section
        Nss = Ns/2    # Nr. of STFT frames overlapping between time sections
        SNRm_acNodes,SNRstd_acNodes = SNR_perband(D_hat,Ds,Ns,Nss,VAD_fact,Fs)

        # Plot statistics across nodse of SNR improvement per bin
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(5, 3, forward=True)   # figure size
        # Mean across nodes
        mapp = ax[0].imshow(SNRm_acNodes.T, extent=[0, SNRm_acNodes.shape[0], Fs/2, 0], interpolation='none')
        ax[0].invert_yaxis()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel='Time frame nr.', ylabel='$f$ [kHz]',
            title='$\\mathrm{E}_{\\mathrm{Nodes}}\\{SNR(t,\\kappa)\\}$')
        ax[0].axes.yaxis.set_ticks(np.arange(0,Fs/2+1e3,1e3))
        ax[0].axes.yaxis.set_ticklabels([str(ii) for ii in np.arange(0,Fs/1e3/2 + 1)])
        # Mean across nodes
        mapp = ax[1].imshow(SNRstd_acNodes.T, extent=[0, SNRm_acNodes.shape[0], Fs/2, 0], interpolation='none')
        ax[1].invert_yaxis()
        ax[1].set_aspect('auto')
        ax[1].set(xlabel='Time frame nr.', ylabel='$f$ [kHz]',
            title='$\\sigma_{\\mathrm{Nodes}}\\{SNR(t,\\kappa)\\}$')
        ax[1].axes.yaxis.set_ticks(np.arange(0,Fs/2+1e3,1e3))
        ax[1].axes.yaxis.set_ticklabels([str(ii) for ii in np.arange(0,Fs/1e3/2 + 1)])
        plt.show()

    # ----------------------------------------------------------------------------------------
    # -------------------------------------- PLOTTING --------------------------------------
    # ----------------------------------------------------------------------------------------

    if 1:
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(12, 4.5, forward=True)   # figure size
        vminn = -120.0
        vmaxx = 10.0

        # Raw microphone signal
        mapp = ax[0].imshow(20*np.log10(np.abs(Ymat)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[0].invert_yaxis()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel='$t$ [s]', ylabel='$f$ [kHz]',
            title='$y_%i(t,f)$ - SNR = %.2fdB' % (ref_sensor+1, SNRy[ref_sensor]))
        ax[0].axes.yaxis.set_ticks(np.arange(0,Fs/2+1e3,1e3))
        ax[0].axes.yaxis.set_ticklabels([str(ii) for ii in np.arange(0,Fs/1e3/2 + 1)])
        fig.colorbar(mapp, ax=ax[0])
        # Raw microphone signal
        mapp = ax[1].imshow(20*np.log10(np.abs(np.squeeze(D_hat[:,:,ref_sensor]))),\
             extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[1].invert_yaxis()
        ax[1].set_aspect('auto')
        ax[1].set(xlabel='$t$ [s]',
            title='$\hat{d}_%i(t,f)$ - SNR = %.2fdB' % (ref_sensor+1, SNRd_hat[ref_sensor]))
        ax[1].axes.yaxis.set_ticklabels([])
        fig.colorbar(mapp, ax=ax[1])
        # Raw microphone signal
        mapp = ax[2].imshow(20*np.log10(np.abs(Ds)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[2].invert_yaxis()
        ax[2].set_aspect('auto')
        ax[2].set(xlabel='$t$ [s]',
            title='$d_%i(t,f)$' % (ref_sensor+1))
        ax[2].axes.yaxis.set_ticklabels([])
        fig.colorbar(mapp, ax=ax[2])

        plt.suptitle('Centralized MWF, sensor #%i (worst initial SNR) -- $\\beta$ = %2f -- SNR imp.: +%.2fdB' % (ref_sensor+1,beta,SNRimp[ref_sensor]))

        if SAVEFIGS:
            if not os.path.isdir(exportDir):
                os.mkdir(exportDir)
                time.sleep(0.1)
            plt.savefig('%s\\STFTs_%ip_ev%i_%s.pdf' % (exportDir,pauseDur,pauseSpace,reftxt))
            plt.savefig('%s\\STFTs_%ip_ev%i_%s.png' % (exportDir,pauseDur,pauseSpace,reftxt))

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
            if not os.path.isdir(exportDir):
                os.mkdir(exportDir)
                time.sleep(0.1)
            plt.savefig('%s\\barchart_%ip_ev%i_%s.pdf' % (exportDir,pauseDur,pauseSpace,reftxt))
            plt.savefig('%s\\barchart_%ip_ev%i_%s.png' % (exportDir,pauseDur,pauseSpace,reftxt))

        plt.show()

# ---------------------------------------------------------------------------------------
# ------------------------------------ SUB-FUNCTIONS ------------------------------------
# ---------------------------------------------------------------------------------------

def SNR_perband(Ymat,Dmat,Ns,Nss,VAD_fact,Fs):
    # SNR_perband -- Computes the SNR per frequency bin across time sections.

    # Time sections
    Nts = int(np.floor(Ymat.shape[1]/Nss))   # Nr. of time sections

    # Get SNR per frequency line for each time section
    snr = np.zeros((Nts,Ymat.shape[0],Ymat.shape[-1]))

    for tt in range(Nts):
        if tt % 10 == 0:
            print('Computing SNR on time section %i/%i...' % (tt+1,Nts))
        idx_ts = np.arange(tt*(Ns - Nss), np.amin((tt*(Ns - Nss) + Ns, Ymat.shape[1])), dtype=int)
        Ymat_ts = Ymat[:,idx_ts,:]
        Dmat_ts = Dmat[:,idx_ts,:]
        
        oVAD = np.zeros_like(Dmat_ts, dtype=float)
        
        for m in range(Ymat.shape[-1]):
            for ii in range(Ymat.shape[0]):
                Ybin = np.squeeze(Ymat_ts[ii,:,m])
                Dbin = np.squeeze(Dmat_ts[ii,:,0])
                thrs_E = np.amax(Dbin**2)/VAD_fact 

                oVAD[ii,:,0] = VAD.oracleVAD(Dbin, 0 ,thrs_E,Fs,plotVAD=False)[0]
                snr[tt,ii,m] = VAD.getSNR(Ybin,oVAD[ii,:,0],silent=True)

    print('SNR computed on all time sections. Deriving statistics across nodes...')

    snr[snr == np.inf] = np.nan
    snr[snr == -np.inf] = np.nan
    SNRstd_acrossNodes = np.nanstd(snr, axis=2)
    SNRm_acrossNodes = np.nanmean(snr, axis=2)

    print('All done. Ready for plotting.')

    return SNRm_acrossNodes,SNRstd_acrossNodes

# ------------------------------------------------------------------------------------
# ------------------------------------ RUN SCRIPT ------------------------------------
# ------------------------------------------------------------------------------------

run_script()
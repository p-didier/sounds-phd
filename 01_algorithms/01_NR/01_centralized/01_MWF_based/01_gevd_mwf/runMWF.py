# Standard library imports
import numpy as np
import os, time, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import scipy.signal as sig
# Third party imports
#
# Local application imports
from MWFpack import myMWF, sig_gen, VAD, eval_enhancement
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
LISTEN_OUTPUT = 0   # If true, plays enhanced and original signals
CWD = os.getcwd()
PLOT_RESULTS = 1    # If true, plots speech enhancement results on figures (+ exports if SAVEFIGS) 


def main():

    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % CWD # path to acoustic scenarios
    speech_in = 'libri'         # name of speech signals library to be used
    noise_type = 'white'        # type of noise to be used
    multispeakers = 'overlap'   # option for multi-speakers speech signal generation
    # multispeakers = 'distinct'   # option for multi-speakers speech signal generation
                                #   -'overlap': the speakers may speak simultaneously.
                                #   -'distinct': the speakers may never speak simultaneously.
    voicedetection = 'VAD'      # type of voice activity detection mechanism to use (VAD or SPP)
    #
    Tmax = 1               # maximum signal duration [s]
    baseSNR = 10            # SNR pre-RIR application [dB]
    #
    pauseDur = 0            # duration of pauses in-between speech segments [s]
    pauseSpace = 3          # duration of speech segments (btw. pauses) [s]
    #
    useGEVD = True          # if True, use GEVD, do not otherwise
    GEVDrank = 1
    #
    SPP_threshold = 0.8
    # Exports
    exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20211007meeting\\onesource_anechoic' % CWD
    # ----- Acoustic scenario + specific speech/noise signal(s) selection (if empty, random selection) 
    ASref = 'J3Mk2_Ns2_Nn3\\testAS_anechoic'      
    # ASref = 'J3Mk2_Ns2_Nn3\\testAS'      
    # Specific speech/noise files 
    speech1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0000.flac'
    speech2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\3570\\5694\\3570-5694-0007.flac'
    speech = [speech1,speech2]
    noise1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise1.wav'
    noise2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise2.wav'
    noise3 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\99_noises\\white_Fs16e3\\whitenoise3.wav'
    noise = [noise1,noise2,noise3]

    # STFT
    L_fft = 2**9                        # Time frame length [samples]
    R_fft = L_fft/2                     # Inter-frame overlap length [samples]
    win = np.sqrt(np.hanning(L_fft))    # STFT time window

    # Covariance estimates
    Tavg = 0.5              # Time constant for exp. averaging of corr. mats
    min_cov_updates = 10    # min. number of covariance matrices updates before 1st filter weights update

    # VAD
    tw = 40e-3              # window length [s]
    ref_sensor = 0          # index of reference sensor
    VAD_fact = 400          # VAD threshold factor w.r.t. max(y**2)

    # ----------------------------------------------------------------------------------------
    # -------------------------------------- PROCESSING --------------------------------------
    # ----------------------------------------------------------------------------------------

    # ~~~~~~~~~~~~~~~~~~ Generate signals ~~~~~~~~~~~~~~~~~~~
    print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref)
    y,ds,ny,t,Fs,J,rd,r,rs,rn,alpha,reftxt = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                                            pauseDur,pauseSpace,ASref,speech,noise,plotAS=None,\
                                            plot_AS_dir='%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20211007meeting\\GIFs\\onesource' % CWD,\
                                            ms=multispeakers)
    print('Microphone signals created using "%s"' % ASref)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    # Set useful data as variables
    M = y.shape[-1]                 # total number of sensors
    Mk = M/J                        # number of sensors per node
    beta = 1 - (R_fft/(Fs*Tavg))    # Forgetting factor
    # beta = 1 - 1/16e3    # Forgetting factor (TMP)
    # Check on input parameters
    if not (Tmax*Fs/(L_fft-R_fft)).is_integer():
        print('WARNING: the chosen <Tmax>=%is conflicts with the frame length <L_fft-R_fft>=%i\n' % (Tmax, L_fft-R_fft))

    # ~~~~~~~~~~~~~~~~~~ Oracle VAD ~~~~~~~~~~~~~~~~~~~
    print('\nComputing oracle VAD from clean speech signal...')
    thrs_E = np.amax(ds[:,ref_sensor]**2)/VAD_fact  
    myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
    print('Oracle VAD computed')

    # Initial SNR estimates
    SNRy = VAD.SNRest(y,myVAD)
    # Get Speech Presence Probability
    if voicedetection == 'SPP':
        Y_best_sensor = calcSTFT(y[:,np.argmax(SNRy)], Fs, win, L_fft, R_fft, 'onesided')[0]
        spp = VAD.oracleSPP(Y_best_sensor,plotSPP=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # User feedback
    if LISTEN_TO_MICS:              # Listen to input and target signals
        sPbIdx = np.argmin(SNRy)
        maxplaytime = 10     # Maximal play-time in seconds
        print('Playing target signal for sensor #%i...' % (sPbIdx+1))
        ps.playthis(ds[:int(maxplaytime*Fs),sPbIdx], Fs)
        print('Playing raw mic. signal for sensor #%i (lowest input SNR)...' % (sPbIdx+1))
        ps.playthis(y[:int(maxplaytime*Fs),sPbIdx], Fs)
    if SHOW_WAVEFORMS:              # Visualize input and target signals
        fig, ax = plt.subplots()
        ax.plot(t, y[:,ref_sensor])
        ax.plot(t, ds[:,ref_sensor])
        ax.set(xlabel='time (s)',
            title='Waveforms for sensor #%i (node #%i)' % (ref_sensor, int(np.floor(ref_sensor/Mk))))
        plt.legend('Raw mic signal', 'Target signal (clean)')
        ax.grid()
        plt.show()

    # ~~~~~~~~~~~~~~~~~~ (GEVD-)MWF ~~~~~~~~~~~~~~~~~~~ 
    print('Entering MWF routine...')
    t0 = time.time()
    if voicedetection == 'VAD':
        D_hat, W_hat, freqs = myMWF.MWF(y,Fs,win,L_fft,R_fft,myVAD,beta,min_cov_updates,useGEVD,GEVDrank,desired=ds)
    elif voicedetection == 'SPP':
        D_hat, W_hat, freqs = myMWF.MWF(y,Fs,win,L_fft,R_fft,spp,beta,min_cov_updates,useGEVD,GEVDrank,desired=ds,SPP_thrs=SPP_threshold)
    t1 = time.time()
    print('Enhanced signals (%i s for %i sensors) computed in %3f s' % (Tmax,J,t1-t0))
    # Get corresponding time-domain signal(s)
    d_hat = calcISTFT(D_hat, win, L_fft, R_fft, 'onesided')
    # Zero-pad if needed to get the same output signal length
    if d_hat.shape[0] < y.shape[0]:
        d_hat = np.concatenate((d_hat, np.zeros((y.shape[0]-d_hat.shape[0], d_hat.shape[1]))), axis=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if LISTEN_OUTPUT:
        # Listen
        sPbIdx = np.argmin(SNRy)
        print('Playing the raw mic. signal for sensor #%i (lowest input SNR)...' % (sPbIdx+1))
        ps.playthis(y[:,sPbIdx], Fs)
        print('Playing the enhanced signal for sensor #%i...' % (sPbIdx+1))
        ps.playthis(d_hat[:,sPbIdx], Fs)

    # ~~~~~~~~~~~~~~~~~~ Speech enhancement performance evaluation (metrics - <pysepm> module) ~~~~~~~~~~~~~~~~~~~ 
    fwSNRseg_noisy,SNRseg_noisy,stoi_noisy = eval_enhancement.eval(ds, y, Fs)
    fwSNRseg_enhanced,SNRseg_enhanced,stoi_enhanced = eval_enhancement.eval(ds, d_hat, Fs)

    print('fwSNRseg improvement:')
    print(fwSNRseg_enhanced - fwSNRseg_noisy)
    print('SNRseg improvement:')
    print(SNRseg_enhanced - SNRseg_noisy)
    print('STOI improvement:')
    print(stoi_enhanced - stoi_noisy)

    # if EXPORTDATA:
    #     # Export SNRs in time domain
    #     npa = np.stack((SNRy,SNRd_hat,SNRimp))
    #     pd.DataFrame(npa, index=['y', 'dhat', 'imp']).to_csv('%s\\SNRs_%ip_ev%i__%s.csv' % (exportDir,pauseDur,pauseSpace,reftxt))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


    # ~~~~~~~~~~~~~~~~~~ MWF SPATIAL EFFECT ~~~~~~~~~~~~~~~~~~~ 
    myMWF.spatial_visu_MWF(W_hat, freqs, rd, alpha, r, Fs, win, L_fft, R_fft, targetSources=rs, noiseSources=rn)

    # ----------------------------------------------------------------------------------------
    # -------------------------------------- PLOTTING --------------------------------------
    # ----------------------------------------------------------------------------------------

    if PLOT_RESULTS:
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(12, 4.5, forward=True)   # figure size
        vminn = -50.0
        vmaxx = 10.0

        # Get STFTs
        ref_sensor = np.argmin(SNRy)
        Ymat = calcSTFT(y[:,ref_sensor], Fs, win, L_fft, R_fft, 'onesided')[0]
        Ds = calcSTFT(ds[:,ref_sensor], Fs, win, L_fft, R_fft, 'onesided')[0]

        # Raw microphone signal
        mapp = ax[0].imshow(20*np.log10(np.abs(Ymat)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[0].invert_yaxis()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel='$t$ [s]', ylabel='$f$ [kHz]',
            title='$y_%i(t,f)$ - STOI = %.2f' % (ref_sensor+1, stoi_noisy[ref_sensor]))
        ax[0].axes.yaxis.set_ticks(np.arange(0,Fs/2+1e3,1e3))
        ax[0].axes.yaxis.set_ticklabels([str(ii) for ii in np.arange(0,Fs/1e3/2 + 1)])
        fig.colorbar(mapp, ax=ax[0])
        # Raw microphone signal
        mapp = ax[1].imshow(20*np.log10(np.abs(np.squeeze(D_hat[:,:,ref_sensor]))),\
             extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
        ax[1].invert_yaxis()
        ax[1].set_aspect('auto')
        ax[1].set(xlabel='$t$ [s]',
            title='$\hat{d}_%i(t,f)$ - STOI = %.2f' % (ref_sensor+1, stoi_enhanced[ref_sensor]))
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

        plt.suptitle('Centralized MWF, sensor #%i (worst initial SNR) -- $\\beta$ = %2f -- STOI imp.: +%.2f' % (ref_sensor+1,beta,stoi_enhanced[ref_sensor] - stoi_noisy[ref_sensor]))

        if SAVEFIGS:
            if not os.path.isdir(exportDir):
                os.mkdir(exportDir)
                time.sleep(0.1)
            plt.savefig('%s\\STFTs_%ip_ev%i_%s.pdf' % (exportDir,pauseDur,pauseSpace,reftxt))
            plt.savefig('%s\\STFTs_%ip_ev%i_%s.png' % (exportDir,pauseDur,pauseSpace,reftxt))

        plt.show()

        # # Bar chart for SNR improvements at each sensor
        # barw = 0.3
        # fig, ax = plt.subplots()
        # fig.set_size_inches(5, 3, forward=True)   # figure size
        # b1 = ax.bar(np.arange(1,M+1)-barw/2, height=SNRy, width=barw)
        # b2 = ax.bar(np.arange(1,M+1)+barw/2, height=SNRd_hat, width=barw)
        # plt.legend([b1,b2],['SNR($y_k$)','SNR($\hat{d}_k$)'])
        # ax.set(title='Average SNR improvement: %.1f dB' % (np.round(np.mean(SNRimp),1)))
        
        # if SAVEFIGS:
        #     if not os.path.isdir(exportDir):
        #         os.mkdir(exportDir)
        #         time.sleep(0.1)
        #     plt.savefig('%s\\barchart_%ip_ev%i_%s.pdf' % (exportDir,pauseDur,pauseSpace,reftxt))
        #     plt.savefig('%s\\barchart_%ip_ev%i_%s.png' % (exportDir,pauseDur,pauseSpace,reftxt))

        # plt.show()

    return None

# ------------------------------------------------------------------------------------
# ------------------------------------ RUN SCRIPT ------------------------------------
# ------------------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit(main())
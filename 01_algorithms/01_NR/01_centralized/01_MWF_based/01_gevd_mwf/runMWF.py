# Standard library imports
import numpy as np
import os, time, sys
import matplotlib.pyplot as plt
import matplotlib
from numpy.core.numeric import zeros_like
from scipy.signal.signaltools import decimate
matplotlib.rcParams['text.usetex'] = False
import pandas as pd
import scipy.io.wavfile
# Third party imports
#
# Local application imports
from MWFpack import myMWF, sig_gen, VAD, spatial
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT, calcISTFT
import playsounds.playsounds as ps
from general.frequency import divide_in_bands
from metrics import eval_enhancement

# Multichannel Wiener Filter (MWF) for noise reduction, using Generalized Eigenvalue Decomposition (GEVD)
# with oracle VAD and full communication (i.e. a fusion center for all nodes data).
#
# (c) Paul Didier - 08-Sept-2021
# SOUNDS ETN - KU Leuven ESAT STADIUS

# Global variables
SAVEFIGS =        0  # If true, saves figures as PNG and PDF files
EXPORTDATA =      0  # If true, exports I/O speech enhancement quality measures (SNRs, STOIs) as CSV files
LISTEN_TO_MICS =  0  # If true, plays examples of target and raw mic signal
SHOW_WAVEFORMS =  0  # If true, plots waveforms of target and raw mic signals
LISTEN_OUTPUT =   0  # If true, plays enhanced and original signals
PLOT_RESULTS =    1  # If true, plots speech enhancement results on figures (+ exports if SAVEFIGS) 
COMPUTE_SPATIAL = 1  # If true, computes the spatial response of the MWF
CWD = os.getcwd()


def main():

    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % CWD # path to acoustic scenarios
    speech_in = 'libri'             # name of speech signals library to be used
    noise_type = 'white'            # type of noise to be used
    multispeakers = 'overlap'       # option for multi-speakers speech signal generation
    # multispeakers = 'distinct'    # option for multi-speakers speech signal generation
                                    #   -'overlap': the speakers may speak simultaneously.
                                    #   -'distinct': the speakers may never speak simultaneously.
    noise_spatially_white = 1       # Option for use of spatially-white noise:
    noise_spatially_white = 'combined'       # Option for use of spatially-white noise:
                #   -false, do not assume any spatially white noise and use specific noise source positions.
                #   -true, assume spatially white noise and ignore entirely specific noise source positions.
                #   -'combined', use specific noise source positions and add some spatially white noise.
    voicedetection = 'VAD'          # type of voice activity detection mechanism to use (VAD or SPP)
    #
    MWFtype = 'batch'               # if 'batch', compute the covariance matrices from the entire signal (AND DO NOT USE GEVD)
    # MWFtype = 'online'            # if 'online', compute the covariance matrices iteratively (possibly using GEVD)
    #
    useGEVD = True                  # if True, use GEVD, do not otherwise
    GEVDrank = 1
    #
    Tmax = 15                       # maximum signal duration [s]
    baseSNR = 10                    # SNR pre-RIR application [dB]
    #
    pauseDur = 1                    # duration of pauses in-between speech segments [s]
    pauseSpace = 3                  # duration of speech segments (btw. pauses) [s]
    #
    SPP_threshold = 0.8
    # Spatial visualization
    bandtype = 'OTOB'   # If 'OTOB', process in 1/3-octave bands
    bandtype = 'OB'     # If 'OB', process in octave bands
    bandtype = None     # If None, do not divide in bands (all freq. at once)
    # bandtype = [None, 'OB']     

    # Exports
    # exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\export_tests' % CWD
    exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\04_for_20211104meeting\\metrics' % (CWD)
    exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\04_for_20211104meeting\\MWFspatial' % (CWD)
    exportDir = '%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\04_for_20211104meeting\\MWF3D_perf' % (CWD)
    # ----- Acoustic scenario + specific speech/noise signal(s) selection (if empty, random selection) 
    # ASref = 'J3Mk2_Ns2_Nn3\\testAS_anechoic'      
    # ASref = 'J3Mk2_Ns2_Nn3\\testAS'            
    # ASref = 'J3Mk2_Ns1_Nn1\\testAS_anechoic'      
    # ASref = 'J5Mk1_Ns1_Nn2\\testAS_anechoic_2D'      
    # ASref = 'J10Mk1_Ns1_Nn2\\testAS_anechoic_2D'      
    # ASref = 'J10Mk1_Ns1_Nn1\\testAS_anechoic_2D'      
    # ASref = 'J1Mk5_Ns1_Nn1\\testAS_anechoic_2D_array'   
    # ASref = 'J1Mk5_Ns1_Nn3\\AS1_anechoic_2D_array' 
    # ASref = 'J5Mk1_Ns1_Nn3\\AS1_anechoic_2D'     
    # ASref = 'J5Mk1_Ns1_Nn1\\AS7_anechoic_2D'       
    # ASref = 'J5Mk1_Ns1_Nn1\\AS6_anechoic_2D'     
    # ASref = 'J5Mk1_Ns2_Nn1\\AS5_anechoic_2D'     
    # ASref = 'J5Mk1_Ns1_Nn2\\AS2_anechoic_2D'     
    # ASref = '2D\\J5Mk1_Ns2_Nn3'   
    # ASref = '2D\\J5Mk1_Ns2_Nn3\\AS0_anechoic_s2_n123'   
    # ASref = '2D\\J5Mk1_Ns2_Nn3\\AS0_anechoic_s12_n1'
    # ASref = '2D\\J5Mk1_Ns2_Nn3\\AS0_anechoic_s12_n3'
    # ASref = '2D\\J5Mk1_Ns2_Nn3\\AS0_anechoic_s12_n123'
    # ASref = 'J5Mk1_Ns2_Nn3\\AS0_anechoic'
    # plotAS = 'PDF'
    plotAS = 'plot'
    # plotAS = None
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
    Tavg = 1              # Time constant for exp. averaging of corr. mats
    min_cov_updates = 10    # min. number of covariance matrices updates before 1st filter weights update

    # VAD
    tw = 40e-3              # window length [s]
    ref_sensor = 0          # index of reference sensor
    VAD_fact = 400          # VAD threshold factor w.r.t. max(y**2)

    if '2D\\' in ASref and ASref.count('\\') == 1:
        allASs = next(os.walk('%s\\%s' % (path_acoustic_scenarios,ASref)))[2]
        nAS = len(allASs)
    else:
        nAS = 1
        

    for idxAS in range(nAS):

        if nAS > 1:
            ASref_full = ASref + '\\%s' % (allASs[idxAS])
        else:
            ASref_full = ASref

        # ----------------------------------------------------------------------------------------
        # -------------------------------------- PROCESSING --------------------------------------
        # ----------------------------------------------------------------------------------------

        # ~~~~~~~~~~~~~~~~~~ Generate signals ~~~~~~~~~~~~~~~~~~~
        print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref_full)
        y,ds,d,ny,y_single_sources,t,Fs,J,rd,r,rs,rn,alpha,reftxt = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                                                pauseDur,pauseSpace,ASref_full,speech,noise,plotAS=plotAS,\
                                                plot_AS_dir=exportDir,\
                                                ms=multispeakers,prewhiten=False,spatially_white_noise=noise_spatially_white)
        print('Microphone signals created using "%s"' % ASref_full)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        # Set useful data as variables
        M = y.shape[-1]                 # total number of sensors
        Mk = M/J                        # number of sensors per node
        beta = 1 - (R_fft/(Fs*Tavg))    # Forgetting factor
        beta = 1 - 1/16e3    # Forgetting factor (TMP)  --- for PhDSU_N03/N04/N05

        # Check on input parameters
        if not (Tmax*Fs/(L_fft-R_fft)).is_integer():
            print('WARNING: the chosen <Tmax>=%is conflicts with the frame length <L_fft-R_fft>=%i\n' % (Tmax, L_fft-R_fft))

        # ~~~~~~~~~~~~~~~~~~ Oracle VAD ~~~~~~~~~~~~~~~~~~~
        print('\nComputing oracle VAD from clean speech signal...')
        thrs_E = np.amax(ds[:,ref_sensor]**2)/VAD_fact  
        myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
        print('Oracle VAD computed')

        # Initial SNR estimates
        SNRy = eval_enhancement.SNRest(y,myVAD)
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
                title='Waveforms for sensor #%i (node #%i)' % (ref_sensor+1, int(np.floor((ref_sensor+1)/Mk))))
            ax.grid()
            ax.legend(['Raw mic signal', 'Target signal (clean)'])
            plt.show()

        # ~~~~~~~~~~~~~~~~~~ (GEVD-)MWF ~~~~~~~~~~~~~~~~~~~ 
        print('Entering MWF routine...')
        t0 = time.time()
        if voicedetection == 'VAD':
            D_hat, W_hat, freqs = myMWF.MWF(y,Fs,win,L_fft,R_fft,myVAD,beta,min_cov_updates,useGEVD,GEVDrank,desired=ds,MWFtype=MWFtype)
        elif voicedetection == 'SPP':
            D_hat, W_hat, freqs = myMWF.MWF(y,Fs,win,L_fft,R_fft,spp,beta,min_cov_updates,useGEVD,GEVDrank,desired=ds,SPP_thrs=SPP_threshold,MWFtype=MWFtype)
        t1 = time.time()
        print('Enhanced signals (%i s for %i sensors) computed in %3f s' % (Tmax,J,t1-t0))
        # Get corresponding time-domain signal(s)
        d_hat = calcISTFT(D_hat, win, L_fft, R_fft, 'onesided')
        # Zero-pad if needed to get the same output signal length
        if d_hat.shape[0] < y.shape[0]:
            d_hat = np.concatenate((d_hat, np.zeros((y.shape[0]-d_hat.shape[0], d_hat.shape[1]))), axis=0)

        d_hat_single_sources = None
        Ns = y_single_sources.shape[-1]
        if Ns > 1:    # in the case where there are several speech sources
            print('Several target sources detected, applying output MWF on single-speech-source synthetized signals\nfor subsequent STOI calculation...')
            # Apply filter to single-speech microphone signals
            d_hat_single_sources = np.zeros_like(y_single_sources)
            for ii in range(Ns):
                d_hat_single_sources[:,:,ii] = myMWF.applyMWF_tdomain(y_single_sources[:,:,ii], W_hat, freqs,win,L_fft,R_fft)
            
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        sPbIdx = np.argmin(SNRy)
        scipy.io.wavfile.write("%s\\before_enhancement.wav" % exportDir, Fs, y[:,sPbIdx])
        scipy.io.wavfile.write("%s\\after_enhancement.wav" % exportDir, Fs, d_hat[:,sPbIdx])
        if LISTEN_OUTPUT:
            # Listen
            print('Playing the raw mic. signal for sensor #%i (lowest input SNR)...' % (sPbIdx+1))
            ps.playthis(y[:,sPbIdx], Fs)
            print('Playing the enhanced signal for sensor #%i...' % (sPbIdx+1))
            ps.playthis(d_hat[:,sPbIdx], Fs)

        # ~~~~~~~~~~~~~~~~~~ Speech enhancement performance evaluation (metrics - <pysepm> module) ~~~~~~~~~~~~~~~~~~~ 
        gamma = np.arange(0.0, 2, 0.5)
        fwSNRseg_L = np.arange(0.01, 0.04, 0.01)
        gamma = [0.2]           # FOR QUICK TESTS
        fwSNRseg_L = [0.03]   # FOR QUICK TESTS
        #
        print('Deriving speech enhancement performance metrics for %i combinations of parameters...' % (len(gamma)*len(fwSNRseg_L))) 
        fwSNRseg_noisy,sisnr_noisy,stoi_noisy = eval_enhancement.eval(ds, y, Fs, myVAD, gamma_fwSNRseg=gamma, frameLen=fwSNRseg_L)
        fwSNRseg_enhanced,sisnr_enhanced,stoi_enhanced = eval_enhancement.eval(ds, d_hat, Fs, myVAD, gamma_fwSNRseg=gamma,  frameLen=fwSNRseg_L)
        if d_hat_single_sources is not None:
            stoi_single_sources_noisy = np.zeros((M, Ns))
            stoi_single_sources_enhanced = np.zeros((M, Ns))
            for ii in range(Ns):
                print('Single speech source case #%i/%i' % (ii+1, Ns))
                stoi_single_sources_noisy[:,ii] = eval_enhancement.eval(ds, y_single_sources[:,:,ii], Fs, myVAD, onlySTOI=1)[-1]
                stoi_single_sources_enhanced[:,ii] = eval_enhancement.eval(ds, d_hat_single_sources[:,:,ii], Fs, myVAD, onlySTOI=1)[-1]

        # Average improvements
        fwSNRseg_imp = np.mean(fwSNRseg_enhanced - fwSNRseg_noisy, axis=0)
        stoi_imp = np.mean(stoi_enhanced - stoi_noisy, axis=0)
        if d_hat_single_sources is not None:    # if several sources, compute improvements per indiv. speaker
            stoi_imp = np.mean(stoi_single_sources_enhanced - stoi_single_sources_noisy, axis=0)
            
        print('Speech enhancement performance metrics calculated.') 

        if EXPORTDATA:
            # Export fwSNRseg values for each value of frame length and gamma exponent
            for ii, g in enumerate(gamma):
                subdirpath = '%s\\fwSNRseg_gamma%s' % (exportDir, str(np.round(g, decimals=2)).replace('.','p'))
                if not os.path.isdir(subdirpath):
                    os.mkdir(subdirpath)  # Make sub-directory
                for jj, l in enumerate(fwSNRseg_L):
                    subdirpath_full = '%s\\frameLen_%s' % (subdirpath, str(np.round(l, decimals=4)).replace('.','p'))
                    if not os.path.isdir(subdirpath_full):
                        os.mkdir(subdirpath_full)  # Make sub-directory
                    npa = np.stack((fwSNRseg_noisy[:,ii,jj],fwSNRseg_enhanced[:,ii,jj]))
                    pd.DataFrame(npa, index=['fwSNRseg_noisy', 'fwSNRseg_enhanced']).to_csv('%s\\fwSNRseg_%ip_ev%i__%s.csv' % (subdirpath_full,pauseDur,pauseSpace,reftxt))
            # Export STOI values
            if d_hat_single_sources is not None:
                for ii in range(Ns):
                    npa = np.stack((stoi_single_sources_noisy[:,ii],stoi_single_sources_enhanced[:,ii]))
                    pd.DataFrame(npa, index=['stoi_noisy', 'stoi_enhanced']).to_csv('%s\\STOI_source%iof%i_%ip_ev%i__%s.csv' % (exportDir,ii+1, Ns, pauseDur,pauseSpace,reftxt))
            else:
                npa = np.stack((stoi_noisy,stoi_enhanced))
                pd.DataFrame(npa, index=['stoi_noisy', 'stoi_enhanced']).to_csv('%s\\STOI_%ip_ev%i__%s.csv' % (exportDir,pauseDur,pauseSpace,reftxt))
            # Export SI-SNR values
            npa = np.stack((sisnr_noisy,sisnr_enhanced))
            pd.DataFrame(npa, index=['sisnr_noisy', 'sisnr_enhanced']).to_csv('%s\\SISNR_%ip_ev%i__%s.csv' % (exportDir,pauseDur,pauseSpace,reftxt))
            print('fwSNRseg, STOI, and SISNR before/after enhancement exported in CSV files\n(in folder <%s>).' % exportDir) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

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

        # ~~~~~~~~~~~~~~~~~~ MWF SPATIAL EFFECT ~~~~~~~~~~~~~~~~~~~ 
        if COMPUTE_SPATIAL:
            rir_dur = L_fft / Fs   # RIR duration [s], based on STFT processing window size. 
            # Re-shape <W_hat>
            W_hat = np.transpose(W_hat, [2,0,1])   #... necessary for the way the spatial visu is implemented (calculation of filter output power)

            if not isinstance(bandtype, list):
                bandtype = [bandtype]
            for idx_bandtype, bt in enumerate(bandtype):
                bands_indices, fc = divide_in_bands(bt, freqs)[:2]
                # Divide in bands
                for ii, band_indices in enumerate(bands_indices):
                    if len(band_indices) > 0:
                        print('\nEXPORTING spatial visualization for band %i/%i' % (ii+1, len(bands_indices)))
                        W_hat_curr = W_hat[band_indices,:,:]
                        freqs_curr = freqs[band_indices]
                        idx_slash_beg = ASref_full.find('\\')
                        idx_slash_end = ASref_full.rfind('\\')
                        if idx_slash_beg == idx_slash_end:
                            idx_slash_beg = -1
                        if bt is None:
                            fname = '%s\\%s\\sv_allf_%s' % (exportDir, ASref_full[idx_slash_beg+1:idx_slash_end], ASref_full[idx_slash_end+1:])
                        else:
                            fname = '%s\\%s\\sv_%s%iHz_%s' % (exportDir, ASref_full[idx_slash_beg+1:idx_slash_end],\
                                bt, np.round(fc[ii]), ASref_full[idx_slash_end+1:])
                        #
                        myMWF.spatial_visu_MWF(W_hat_curr, freqs_curr, rd, alpha, r, Fs, rir_dur, targetSources=rs, noiseSources=rn,\
                            exportit=1, exportname=fname, noise_spatially_white=noise_spatially_white, fwSNRseg_imp=fwSNRseg_imp, stoi_imp=stoi_imp)
                    else:
                        print('\nNo frequencies in band centered on %.1f Hz' % fc[ii])

        print('\n\n---------------- ALL DONE ----------------')
        stop = 1

        plt.close('all') 

    return None

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
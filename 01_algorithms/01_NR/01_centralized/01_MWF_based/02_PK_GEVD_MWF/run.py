# Standard library imports
import sys,os,time
import numpy as np
from pathlib import Path
# Local application imports
currdir = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(currdir.parent, '01_GEVD_MWF')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from mySTFT.calc_STFT import calcSTFT, calcISTFT
from MWFpack import sig_gen, VAD, myMWF

CWD = os.getcwd()

def main():
    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % CWD # path to acoustic scenarios
    speech_in = 'libri'         # name of speech signals library to be used
    noise_type = 'white'        # type of noise to be used
    multispeakers = 'overlap'   # option for multi-speakers speech signal generation
    # multispeakers = 'distinct'   # option for multi-speakers speech signal generation
                                #   -'overlap': the speakers may speak simultaneously.
                                #   -'distinct': the speakers may never speak simultaneously.
    #
    Tmax = 15               # maximum signal duration [s]
    baseSNR = 10            # SNR pre-RIR application [dB]
    #
    pauseDur = 0            # duration of pauses in-between speech segments [s]
    pauseSpace = 3          # duration of speech segments (btw. pauses) [s]
    #
    useGEVD = True          # if True, use GEVD, do not otherwise
    GEVDrank = 1
    # ----- Acoustic scenario + specific speech/noise signal(s) selection (if empty, random selection) 
    ASref = 'J3Mk2_Ns2_Nn3\\testAS_anechoic'      
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


    # ~~~~~~~~~~~~~~~~~~ Generate signals ~~~~~~~~~~~~~~~~~~~
    print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref)
    y,ds,ny,t,Fs,nNodes,reftxt = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                                            pauseDur,pauseSpace,ASref,speech,noise,plotAS=None,\
                                            plot_AS_dir='%s\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20211007meeting\\GIFs\\onesource' % CWD,\
                                            ms=multispeakers)
    print('Microphone signals created using "%s"' % ASref)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Set useful data as variables
    nSensors = y.shape[-1]          # total number of sensors
    Mk = nSensors/nNodes            # number of sensors per node
    beta = 1 - (R_fft/(Fs*Tavg))    # Forgetting factor

    # ~~~~~~~~~~~~~~~~~~ Oracle VAD ~~~~~~~~~~~~~~~~~~~
    print('\nComputing oracle VAD from clean speech signal...')
    thrs_E = np.amax(ds[:,ref_sensor]**2)/VAD_fact  
    myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
    print('Oracle VAD computed')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~ PK-GEVD-MWF ~~~~~~~~~~~~~~~~~~~ 
    print('Entering MWF routine...')
    t0 = time.time()
    D_hat = myMWF.MWF(y,Fs,win,L_fft,R_fft,myVAD,beta,min_cov_updates,useGEVD,GEVDrank,desired=ds)
    t1 = time.time()
    print('Enhanced signals (%i s for %i sensors) computed in %3f s' % (Tmax,J,t1-t0))
    # Get corresponding time-domain signal(s)
    d_hat = calcISTFT(D_hat, win, L_fft, R_fft, 'onesided')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return 0


if __name__ == '__main__':
    sys.exit(main())
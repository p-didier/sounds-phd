from operator import index, xor
import random
import matplotlib
import numpy as np
import random
import scipy.signal as sig
# Custom tools
from sig_gen import sig_gen
import VAD, myMWF
# Plotting
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False

# Classic Multichannel Wiener Filter for noise reduction, 
# using oracle VAD and full communication (i.e. a fusion center for all nodes data).
#
# (c) Paul Didier - 08-Sept-2021
# SOUNDS ETN - KU Leuven ESAT STADIUS

path_acoustic_scenarios = 'C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\02_data\\01_acoustic_scenarios'  # path to acoustic scenarios
speech_in = 'libri'     # name of speech signals library to be used
Tmax = 15               # maximum signal duration [s]
noise_type = 'white'    # type of noise to be used
baseSNR = 10            # SNR pre-RIR application [dB]
pauseDur = 1            # duration of pauses in-between speech segments [s]
pauseSpace = 1          # duration of speech segments (btw. pauses) [s]
# ----- Acoustic scenario and speech signal specific selection
ASref = 'AS9_J5_Ns1_Nn1'  # acoustic scenario (if empty, random selection)
# speech = ''                    # speech signals (if empty, random selection)
speech1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0000.flac'
speech2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0001.flac'
speech = [speech1,speech2]


# Covariance estimates
beta = 1 - 1/16e3       # autocorrelation matrices time-avg. update constant
L = 2**9                # time-frame size [samples]
min_cov_updates = 10    # min. number of covariance matrices updates before 1st filter weights update

# VAD
tw = 40e-3              # window length [s]
ref_sensor = 1          # index of reference sensor
VAD_fact = 200          # VAD threshold factor w.r.t. max(y**2)

# General
Nfft = 2**10            # Number of FFT points (when computing STFTs)

# I) Generate microphone signals
print('\nGenerating mic. signals, using acoustic scenario "%s"' % ASref)
y,ds,ny,t,Fs = sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                        pauseDur,pauseSpace,ASref,speech)
print('Microphone signals created using "%s"' % ASref)
# Set useful data as variables
J = y.shape[-1]

# II) Oracle VAD
thrs_E = np.amax((ds[:,ref_sensor])**2)/VAD_fact  
print('\nComputing oracle VAD from clean speech signal...')
myVAD = VAD.oracleVAD(ds[:,0], tw, thrs_E, Fs, plotVAD=0)[0]
print('Oracle VAD computed')

# III) Compute MWF 
d_hat = myMWF.MWF(y, myVAD, beta, L, Nfft, min_cov_updates)

# IV) SNR improvement estimates
SNRy = np.zeros(J)
SNRd_hat = np.zeros(J)
SNRimp = np.zeros(J)
for k in range(J):
    SNRy[k] = VAD.SNRest(y[:,k],myVAD)
    SNRd_hat[k] = VAD.SNRest(d_hat[:,k],myVAD)
    SNRimp[k] = SNRd_hat[k] - SNRy[k]

print('Raw sensor SNRs [dB]:')
print(SNRy)
print('Post-processed sensor SNRs [dB]:')
print(SNRd_hat)
print('SNR improvements (per sensor, in [dB]):')
print(SNRimp)

# fig, ax = plt.subplots()
# ax.plot(t, d_hat)
# ax.grid()
# plt.show()

# Visualize microphone signals
if 1:
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(12, 4.5, forward=True)   # figure size
    vminn = -150.0
    vmaxx = 0.0

    # Y = sig.stft(y[:,ii], fs=Fs, nperseg=L, nfft=Nfft)[2]
    Y = sig.stft(y[:,ref_sensor], fs=Fs, nperseg=L, nfft=Nfft)[2]
    Ds = sig.stft(ds[:,ref_sensor], fs=Fs, nperseg=L, nfft=Nfft)[2]
    Dhat = sig.stft(d_hat[:,ref_sensor], fs=Fs, nperseg=L, nfft=Nfft)[2]
    # Raw microphone signal
    mapp = ax[0].imshow(20*np.log10(np.abs(Y)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
    ax[0].invert_yaxis()
    ax[0].set_aspect('auto')
    ax[0].set(xlabel='$t$ [s]',
        title='$y_%i(t,f)$' % ref_sensor)
    # Raw microphone signal
    mapp = ax[1].imshow(20*np.log10(np.abs(Dhat)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
    ax[1].invert_yaxis()
    ax[1].set_aspect('auto')
    ax[1].set(xlabel='$t$ [s]',
        title='$\hat{d}_%i(t,f)$' % ref_sensor)
    # Raw microphone signal
    mapp = ax[2].imshow(20*np.log10(np.abs(Ds)), extent=[0, t[-1], Fs/2, 0], vmin=vminn, vmax=vmaxx)
    ax[2].invert_yaxis()
    ax[2].set_aspect('auto')
    ax[2].set(xlabel='$t$ [s]', ylabel='$f$ [Hz]',
        title='$d_%i(t,f)$' % ref_sensor)
    fig.colorbar(mapp, ax=ax[2])

    plt.show()
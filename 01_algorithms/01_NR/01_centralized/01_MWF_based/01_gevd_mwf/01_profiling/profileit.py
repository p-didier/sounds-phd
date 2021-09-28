import cProfile
import sys, os
sys.path.append('C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_MWF')
# print(sys.path)
from MWFpack import sig_gen

def run():
    path_acoustic_scenarios = '%s\\02_data\\01_acoustic_scenarios' % os.getcwd()  # path to acoustic scenarios
    speech_in = 'libri'     # name of speech signals library to be used
    Tmax = 15               # maximum signal duration [s]
    noise_type = 'white'    # type of noise to be used
    baseSNR = 10            # SNR pre-RIR application [dB]
    pauseDur = 1            # duration of pauses in-between speech segments [s]
    pauseSpace = 1          # duration of speech segments (btw. pauses) [s]
    # ----- Acoustic scenario and speech signal specific selection
    ASref = 'AS9_J5_Ns1_Nn1'  # acoustic scenario (if empty, random selection)
    # ASref = 'testAS'  # acoustic scenario (if empty, random selection)
    # ASref = 'testAS_anechoic'  # acoustic scenario (if empty, random selection)
    # speech = ''                    # speech signals (if empty, random selection)
    speech1 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0000.flac'
    speech2 = 'C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals\\01_LibriSpeech_ASR\\test-clean\\61\\70968\\61-70968-0001.flac'
    speech = [speech1,speech2]

    # I) Generate microphone signals
    import time
    t0 = time.time()

    y,ds,ny,t,Fs = sig_gen.sig_gen(path_acoustic_scenarios,speech_in,Tmax,noise_type,baseSNR,\
                            pauseDur,pauseSpace,ASref,speech)
    t1 = time.time()
    print('Time elapsed: %f s' % (t1-t0))

# cProfile.run('run()')
run()
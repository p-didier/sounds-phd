# >>> Purpose of script:
# Generate custom-length speech signals from Librispeech speech snippets.
# To be used, e.g., as non-stationary interfering signal for noise reduction.

import sys, os
import copy
import numpy as np
from pathlib import Path, PurePath
import soundfile as sf
import matplotlib.pyplot as plt
import simpleaudio as sa
from scipy.io import wavfile

# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
from VAD import oracleVAD

# Paths
PATH_TO_LIBRI = 'C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/02_research/03_simulations/99_datasets/01_signals/01_LibriSpeech_ASR/test-clean'
EXPORT_PATH = 'C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/02_data/00_raw_signals/02_noise/speech'
# Numerical global variables
SEED = 12345  # for random generators
T = 30  # second(s)
N_FILES = 10    # number of files to generate
# Booleans
PLOT_AND_LISTEN = False  # if True, shows plot of the signal and plays it back
EXPORT_FILE = True
ELIMINATE_INITIAL_SILENCE = True
# ^^^ if True, ensure that the audio file starts on a speech segment
ELIMINATE_ALL_SILENCES = True
# ^^^ if True, ensures that the audio file is full of speech segments

def main():

    # Get random generator
    rng = np.random.default_rng(seed=SEED)

    for ii in range(N_FILES):
        print(f'Generating signal {ii + 1}/{N_FILES}...')

        # Select random folder
        folder, ref = select_random_folder(
            dir=PATH_TO_LIBRI,
            rng=rng,
        )

        # Infer sampling frequency
        fs = infer_sampling_freq(folder)

        # Build signal
        signal = build_signal(
            dir=folder,
            fs=fs,
            T=T,
            elimInitialSilence=ELIMINATE_INITIAL_SILENCE,
            elimAllSilences=ELIMINATE_ALL_SILENCES
        )

        # Plot
        if PLOT_AND_LISTEN:
            plot_and_hear_signal(signal, fs)

        # Save
        if EXPORT_FILE:
            amplitude = np.iinfo(np.int16).max
            nparrayNormalized = (amplitude * signal / np.amax(signal) * 0.5)\
                .astype(np.int16)  # 0.5 to avoid clipping
            wavfile.write(f'{EXPORT_PATH}/speech_{ref}_{T}s.wav', fs, nparrayNormalized)

    return 0


def plot_and_hear_signal(sig, fs):
    """
    Plots generated signal and plays it back. 
    """

    # Time vector (x-axis)
    t = np.arange(len(sig)) / fs

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(t, sig)
    axes.grid()
    axes.set_xlabel('$t$ [s]')
    plt.tight_layout()	
    plt.show(block=False)

    # Listen
    audio_array = copy.copy(sig)
    audio_array *= 32767 / max(abs(sig))
    audio_array = audio_array.astype(np.int16)
    sa.play_buffer(audio_array,1,2,fs)

    return None


def build_signal(dir, fs, T, elimInitialSilence=False, elimAllSilences=False):
    """
    Builds speech signal of desired length `T` by concatenating fragments
    contained in folder `dir`. 
    """
    # Target number of samples
    desiredLength = int(np.floor(T * fs))

    # List audio files
    p = Path(dir).glob('**/*')
    files = [x for x in p if x.is_file()]  # https://stackoverflow.com/a/40216619

    signal = np.array([])
    idx = 0
    while len(signal) < desiredLength:
        while files[idx].name[-5:] != '.flac':
            idx += 1
        fragment, _ = sf.read(str(files[idx]))
        # Eliminate initial silence
        if len(signal) == 0 and elimInitialSilence:
            antiVAD = np.array(fragment <= 0.5 * np.amax(fragment), dtype=int)
            # vvv find spot where the first 0 appears in `antiVAD`
            idxStart = list(np.diff(antiVAD)).index(-1)
            fragment = fragment[idxStart:]
        if elimAllSilences:
            # Eliminate all silences (all VAD=0)
            # -- useful to get a minimum-power interfering signal
            # (e.g., for DWACD-based SRO estimation)
            # -- see ICASSP2023/OJSP rebuttal, reviewer #4.
            thrsVAD = np.amax(fragment ** 2) / 4000
            vad, _ = oracleVAD(fragment, tw=40e-3, thrs=thrsVAD, Fs=fs)
            fragment = fragment[vad == 1]
        signal = np.concatenate((signal, np.array(fragment)))
        idx += 1
        if idx >= len(files) - 1:
            idx = 0  # loop back

    # Truncate extra samples
    if len(signal) > desiredLength:
        signal = signal[:desiredLength]

    return signal


def infer_sampling_freq(dir):
    """
    Infer sampling frequency from audio files in folder `dir`.
    """

    p = Path(dir).glob('**/*')
    files = [x for x in p if x.is_file()]  # https://stackoverflow.com/a/40216619
    # Infer
    _, fs = sf.read(str(files[0]))

    return fs


def select_random_folder(dir, rng):
    """
    Select random "end-of-line" subfolder within `directory`.
    A reference string is also returned to name files further down the line. 
    """
    flagStop = False
    currDir = dir
    while not flagStop:
        immediateSubdirs = [f.path for f in os.scandir(currDir) if f.is_dir()]
        if len(immediateSubdirs) == 0:
            # We have reach a leaf of the folder structure
            flagStop = True
        elif len(immediateSubdirs) == 1:
            # There is a single subfolder
            folder = immediateSubdirs[0]
            currDir = copy.copy(folder)
        else:
            # Select a random subfolder
            idx = rng.integers(low=0, high=len(immediateSubdirs) - 1, size=1)
            folder = immediateSubdirs[idx[0]]
            currDir = copy.copy(folder)

    # Build reference string
    ref = Path(folder).parent.stem + '_' + Path(folder).stem

    return folder, ref

if __name__== "__main__" :
    sys.exit(main())
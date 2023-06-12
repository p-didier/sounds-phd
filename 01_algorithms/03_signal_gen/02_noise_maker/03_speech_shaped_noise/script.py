import numpy as np
from numpy import fft
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Globals
# FILEPATH = '02_data/00_raw_signals/01_speech/speech2.wav'
FILEPATH = 'C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/02_research/03_simulations/99_datasets/01_signals/01_LibriSpeech_ASR/test-clean'
EXPORTFOLDER = '02_data/00_raw_signals/02_noise/ssn'
SEED = 12346  # only used if `FILEPATH` is a folder
NOISE_DURATION = 60*3  # [s]

def main(filePath=FILEPATH, noiseDuration=NOISE_DURATION):

    np.random.seed(SEED)  # Set random seed

    # Case where the filepath is a directory
    if Path(filePath).is_dir():
        flagStop = False
        while not flagStop:
            # Check if folder contains sound files (wav or flac)
            soundFiles = list(Path(filePath).glob('**/*.wav')) +\
                list(Path(filePath).glob('**/*.flac'))
            if len(soundFiles) > 0:
                # If so select a random file
                filePath = soundFiles[
                    int(np.random.randint(0, len(soundFiles)))
                ]
                print(f'Randomly selected file: {filePath}')
                flagStop = True
            else:
                # If not, check for immediate subfolders
                subfolders = list(Path(filePath).glob('**/*'))
                subfolders = [x for x in subfolders if x.is_dir()]
                if len(subfolders) > 0:
                    filePath = subfolders[
                        int(np.random.randint(0, len(subfolders)))
                    ]
                else:
                    print(f'The folder {filePath} is empty.')
                    raise ValueError
    
    speech, fs = sf.read(
        file=filePath
    )

    noise = _noise_from_signal(
        x=speech,
        fs=fs,
        dur=noiseDuration
    )

    # Look at spectra
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(20 * np.log10(np.abs(np.fft.fft(speech))))
    axes.plot(20 * np.log10(np.abs(np.fft.fft(noise))))
    axes.grid()
    plt.tight_layout()	
    plt.show()

    # Export files
    if 1:
        sf.write(
            file=f'{EXPORTFOLDER}/ssn_{Path(filePath).stem}.wav',
            data=noise,
            samplerate=fs
        )

    stop = 1


# From: https://github.com/timmahrt/pyAcoustics/blob/main/pyacoustics/speech_filters/speech_shaped_noise.py 
def next_pow_2(x):
    """Calculates the next power of 2 of a number."""
    return int(pow(2, np.ceil(np.log2(x))))


# From: https://github.com/timmahrt/pyAcoustics/blob/main/pyacoustics/speech_filters/speech_shaped_noise.py 
def _noise_from_signal(x, fs=40000, dur=1, keep_env=False):
    """Create a noise with same spectrum as the input signal.
    Parameters
    ----------
    x : array_like
        Input signal.
    fs : int
        Sampling frequency of the input signal. (Default value = 40000)
    dur : float
        Duration of output noise file [s]
    keep_env : bool
        Apply the envelope of the original signal to the noise. (Default
        value = False)
    Returns
    -------
    ndarray
        Noise signal.
    """
    x = np.asarray(x)
    n_x = int(dur * fs)
    # n_x = x.shape[-1]
    n_fft = next_pow_2(n_x)
    Xfft = fft.rfft(x, next_pow_2(n_fft))
    # Randomize phase.
    noise_mag = np.abs(Xfft) * np.exp(
        2 * np.pi * 1j * np.random.random(Xfft.shape[-1]))
    noise = np.real(fft.irfft(noise_mag, n_fft))
    out = noise[:n_x]

    if keep_env:
        env = np.abs(signal.hilbert(x))
        [bb, aa] = signal.butter(6, 50 / (fs / 2))  # 50 Hz LP filter
        env = signal.filtfilt(bb, aa, env)
        out *= env

    return out


if __name__=='__main__':
    sys.exit(main())
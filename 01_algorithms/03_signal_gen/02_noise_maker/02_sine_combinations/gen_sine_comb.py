
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

FS = 16e3
AUDIOLENGTH = 30
MAXFREQ = FS / 2
SEED = 12345        # random generator seed
PAUSEEVERY = 3      # [s]
PAUSEDURATION = 2   # [s]
exportFolder = 'U:\\py\\sounds-phd\\01_algorithms\\03_signal_gen\\02_noise_maker\\02_sine_combinations\\sounds'
NSIGNALS = 5

def main():
    """Generate some "noise": Sum of randomly weighted sine-functions
    -- From Taewoong Lee's MATLAB function `genscodatar.m`
    """
    # Base checks
    if MAXFREQ > FS / 2:
        return ValueError(f'Too high max. freq. chosen (fmax={MAXFREQ} Hz) for the current sampling freq. (fs={FS} Hz).')

    # General seeds
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(low=1, high=9999, size=(NSIGNALS,))

    for ii in range(NSIGNALS):
        print(f'Generating sine-combination {ii+1}/{NSIGNALS}...')

        # Set random generator seed
        np.random.seed(seeds[ii])

        x, t = gen_sincomb()

        # Introduce pauses
        x = add_pauses(x, pauseDur=PAUSEDURATION, pauseEvery=PAUSEEVERY, fs=FS)

        if 0:
            # Plot
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            ax.plot(t, x)
            ax.grid()
            plt.tight_layout()	
            plt.show()

        # Export as WAV
        amplitude = np.iinfo(np.int16).max
        data = (amplitude * x / np.amax(x) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
        wavfile.write(f'{exportFolder}/mySineCombination{ii+1}.wav', int(FS), data)

    return None


def add_pauses(x, pauseDur, pauseEvery, fs):

    nSamplesPerPause = int(fs * pauseDur)
    nSamplesBetweenPauses = int(fs * pauseEvery)

    xWithPauses = np.empty((0,))
    idxStart = 0
    while len(xWithPauses) < len(x):
        idxEnd = idxStart + nSamplesBetweenPauses
        xWithPauses = np.concatenate((xWithPauses, x[idxStart:idxEnd], np.zeros(nSamplesPerPause)))
        idxStart = idxEnd

    if len(xWithPauses) > len(x):
        xWithPauses = xWithPauses[:len(x)]


    return xWithPauses


def gen_sincomb():

    n = np.arange(FS * AUDIOLENGTH)

    t = 1 / FS * n  # time vector

    x = np.zeros(len(t))

    freqs = np.arange(start=20, stop=MAXFREQ, step=5)     # frequencies of sines

    # Random amplitudes and phases
    amps = (2 * np.random.random(len(freqs)) - 1)
    phases = np.pi / 180 * (2 * np.random.random(len(freqs)) - 1)
    
    for ii in range(len(freqs)):
        fvar = np.random.randn(1)
        fvar = 0.1 * (fvar - np.round(fvar))
        fvar += freqs[ii]

        x += amps[ii] * np.sin(2 * np.pi * fvar * t + phases[ii])

    return x, t

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
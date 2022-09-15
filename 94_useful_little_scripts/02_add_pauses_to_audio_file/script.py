import sys
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings():
    pauseEvery : float # [s]
    pauseDur : float # [s]
    totalDur : float # [s]
    smoothenPauses : bool = False # if True, smoothen pauses (avoid abrupt cuts in audio)
    smoothingDur : float = 0.25  # [s]
    export : bool = False 

def main():

    settings = Settings(
        pauseEvery=1.5,
        pauseDur=1.,
        totalDur=60.,
        smoothenPauses=True,
        smoothingDur=0.25,
        export=True
    )

    # Beethoven's Für Elise
    filepath = 'C:/Users/pdidier/Dropbox/BELGIUM/KU Leuven/SOUNDS_PhD/02_research/99_useful/sound_files/fur_elise.wav'

    audioWithPauses, fs = add_pauses(filepath, settings)

    # Plot
    t = np.arange(audioWithPauses.shape[0]) / fs
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(t, audioWithPauses)
    axes.grid()
    plt.tight_layout()	
    plt.show()

    if settings.export:
        sf.write(f'{Path(filepath).parent}/{Path(filepath).stem}_wPauses.wav', audioWithPauses, fs)


def add_pauses(path, s: Settings):

    # Read file
    x, fs = librosa.load(path)

    # Check num. channels
    if x.ndim == 1:
        x = x[:, np.newaxis]
    nChannels = np.amin(x.shape)
    
    audio = np.zeros((0, nChannels))
    idxEndChunk = 0  # init
    while audio.shape[0] < int(np.floor(s.totalDur * fs)):

        idxBegChunk = idxEndChunk + int(s.pauseEvery * fs)
        idxEndChunk = idxBegChunk + int(s.pauseEvery * fs)
        newChunk = x[idxBegChunk:idxEndChunk, :]

        if s.smoothenPauses:
            nSamplesSmoothing = int(s.smoothingDur * fs)
            win = np.hanning(2 * nSamplesSmoothing)
            win = win[:, np.newaxis]
            newChunk[-nSamplesSmoothing:, :] *= win[-nSamplesSmoothing:]
            newChunk[:nSamplesSmoothing, :] *= win[:nSamplesSmoothing]

        audio = np.concatenate((audio, newChunk), axis=0)
        # Add silence
        audio = np.concatenate((audio, np.zeros((int(s.pauseDur * fs), nChannels))), axis=0)

    return audio, fs

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
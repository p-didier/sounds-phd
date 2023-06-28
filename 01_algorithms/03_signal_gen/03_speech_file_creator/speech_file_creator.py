# Purpose of script:
# Create custom speech files from library of speech files. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import resampy
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass

PATH_TO_LIB_ROOT = 'C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/02_research/03_simulations/99_datasets/01_signals/01_LibriSpeech_ASR/test-clean'
SEED = 12345
EXPORT_PATH = '02_data/00_raw_signals/01_speech/custom'

N_FILES_TO_GENERATE = 10

@dataclass
class SpeechFileParameters:
    pathToLibRoot: str = PATH_TO_LIB_ROOT  # Path to root of library of speech files
    T: float = 60       # Duration of speech file in seconds
    fs: int = 16000     # Sampling frequency in Hz
    speechActiveMinDuration: float = 0.5  # Minimum duration of speech active segment in seconds
    speechActiveMaxDuration: float = 5.0  # Maximum duration of speech active segment in seconds
    speechInactiveMinDuration: float = 0.5  # Minimum duration of speech inactive segment in seconds
    speechInactiveMaxDuration: float = 5.0  # Maximum duration of speech inactive segment in seconds
    durFadeOut: float = 0.01  # Duration of fade out in seconds
    exportFormat: str = 'wav'  # Export format of speech file
    exportPath: str = ''  # Path to export speech file to
    nTalkers: int = 1  # Number of speakers in speech file


def main():
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(SEED)

    # Instantiate parameters
    p = SpeechFileParameters(
        T=60,
        fs=16000,
        exportPath=EXPORT_PATH
    )

    for i in range(N_FILES_TO_GENERATE):
        print(f'Generating speech file {i+1}/{N_FILES_TO_GENERATE}...')
        create_speech_file(p)


def create_speech_file(p: SpeechFileParameters):

    # Select folders to choose from
    folders = list(Path(p.pathToLibRoot).glob('*'))
    folders = [f for f in folders if f.is_dir()]
    folders = np.random.choice(folders, p.nTalkers, replace=False)

    speech = np.zeros(0)
    while len(speech) < p.fs * p.T:
        print(f'Current length of speech file: {len(speech) / p.fs} seconds...')
        # Select random speech file from library
        y = get_speech_snippet(folders, p.fs, format='flac')
        # Check if speech file is long enough
        while len(y) < p.fs * p.speechActiveMinDuration:
            y2 = get_speech_snippet(folders, p.fs, format='flac')
            y = np.append(y, y2)
        # Make sure the speech file is not too long
        if len(y) > p.fs * p.speechActiveMaxDuration:
            y = y[:int(p.fs * p.speechActiveMaxDuration)]
            # Fade out to avoid clicks
            fadeOut = np.linspace(1, 0, int(p.fs * p.durFadeOut))
            y[-len(fadeOut):] = y[-len(fadeOut):] * fadeOut

        # Add silence after speech file
        silence = np.zeros(int(np.random.uniform(
            p.fs * p.speechInactiveMinDuration,
            p.fs * p.speechInactiveMaxDuration
        )))
        currSnippet = np.append(y, silence)

        # Add snippet to speech file
        speech = np.append(speech, currSnippet)

    # Make sure speech file is not too long
    speech = speech[:int(p.fs * p.T)]
    
    # Save speech file
    fname =f'speech_{array_id(speech)}_{p.nTalkers}talkers.{p.exportFormat}'
    sf.write(f'{p.exportPath}/{fname}', speech, p.fs)
    print(f'Speech file saved to {p.exportPath}/{fname}')


def get_speech_snippet(folders, fs, format='flac'):
    """Get random speech snippet from library of speech files."""
    # Select random speech file from library of speech files with the
    # right format from the given folders.
    speechFiles = []
    for folder in folders:
        speechFiles += list(folder.rglob(f'*.{format}'))
    speechFile = np.random.choice(speechFiles)
    if speechFiles == []:
        raise ValueError('No speech file found in library.')
    
    y, fsSnippet = sf.read(speechFile)
    # Resample if necessary
    if fsSnippet != fs:
        y = resampy.resample(y, fsSnippet, fs)
    # Normalize 
    y = y / np.max(np.abs(y))
    return y


def array_id(
        a: np.ndarray, *,
        include_dtype=False,
        include_shape=False,
        algo = 'xxhash'
    ):
    """
    Computes a unique ID for a numpy array.
    From: https://stackoverflow.com/a/64756069

    Parameters
    ----------
    a : np.ndarray
        The array to compute the ID for.
    include_dtype : bool, optional
        Whether to include the dtype in the ID.
    include_shape : bool, optional
    """
    data = bytes()
    if include_dtype:
        data += str(a.dtype).encode('ascii')
    data += b','
    if include_shape:
        data += str(a.shape).encode('ascii')
    data += b','
    data += a.tobytes()
    if algo == 'sha256':
        import hashlib
        return hashlib.sha256(data).hexdigest().upper()
    elif algo == 'xxhash':
        import xxhash
        return xxhash.xxh3_64(data).hexdigest().upper()
    else:
        assert False, algo


if __name__ == '__main__':
    sys.exit(main())
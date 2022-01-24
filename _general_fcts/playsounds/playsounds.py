from multiprocessing.managers import ValueProxy
from sys import path_importer_cache
import simpleaudio as sa
import numpy as np
from pathlib import Path
import time

def playthis(nparray, Fs):
    """Plays audio from numpy array"""
    amplitude = np.iinfo(np.int16).max
    audio_data = (amplitude*nparray/np.amax(nparray)).astype(np.int16)

    nChannels = sum(np.array(audio_data.shape) > 1)

    play_obj = sa.play_buffer(audio_data, nChannels, 2, Fs)
    play_obj.wait_done()

    return None


def playwavfile(pathToFile, listeningMaxDuration):
    """Plays audio contained in wav file
    Parameters
    ----------
    pathToFile : str
        Path to the file to be read.
    listeningMaxDuration : float
        Maximal playback duration [s].
    """
    # Check path correctness
    if not Path(pathToFile).is_file():
        raise ValueError(f'The file\n"{pathToFile}"\ndoes not exist.')
    # Import sound
    wave_obj = sa.WaveObject.from_wave_file(pathToFile)
    # Listen
    tStartPB = time.perf_counter()
    play_obj = wave_obj.play()
    while play_obj.is_playing(): 
        if (time.perf_counter() - tStartPB) >= listeningMaxDuration:
            play_obj.stop()
    return None
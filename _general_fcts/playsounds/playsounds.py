import simpleaudio as sa
import numpy as np

def playthis(nparray, Fs):
    # Plays audio from numpy array

    amplitude = np.iinfo(np.int16).max
    audio_data = (amplitude*nparray).astype(np.int16)

    nChannels = len(audio_data.shape)

    play_obj = sa.play_buffer(audio_data, nChannels, 2, Fs)
    play_obj.wait_done()

    return None
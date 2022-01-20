import simpleaudio as sa
import numpy as np

def playthis(nparray, Fs):
    """Plays audio from numpy array"""
    amplitude = np.iinfo(np.int16).max
    audio_data = (amplitude*nparray/np.amax(nparray)).astype(np.int16)

    nChannels = sum(np.array(audio_data.shape) > 1)

    play_obj = sa.play_buffer(audio_data, nChannels, 2, Fs)
    play_obj.wait_done()

    return None


def playwavfile(filename):
    """Plays audio contained in wav file"""
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing
    return None

# %%

import numpy as np
from IPython.display import Audio
# import sounddevice as sd  # <-- "PortAudio library not found"
# import pyaudio  # <-- won't install via pip...
from scipy.io import wavfile
# from playsound import playsound  <-- unknown dependencies "gi" + others
from pathlib import Path
import simpleaudio as sa

wave_audio = np.sin(np.linspace(0, 3000, 20000))

# Audio(wave_audio, rate=20000)   # Option 1 (doesn't seem to work -- 
#             # shows widget in Jupyter Interactive window but won't play)

# wav_wave = np.array(wave_audio, dtype=np.int16)
# sd.play(wav_wave, 20000)   # Option 2 doesn't work at all -- "PortAudio library not found"


# sr, wdata = wavfile.read('test_sound.wav')

# p = pyaudio.PyAudio()
# stream = p.open(format = p.get_format_from_width(1), channels = 1, rate = sr, output = True)
# stream.write(wdata)
# stream.stop_stream()
# stream.close()
# p.terminate()

# playsound(f'{Path(__file__).parent}/test_sound.wav')


filename = '/users/sista/pdidier/py/sounds-phd/01_algorithms/01_NR/02_distributed/dev/test_sound.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing


# %%
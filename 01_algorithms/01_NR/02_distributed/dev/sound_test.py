
# %%

import numpy as np
from IPython.display import Audio
# import sounddevice as sd  # <-- "PortAudio library not found"
# import pyaudio  # <-- won't install via pip...
from scipy.io import wavfile
# from playsound import playsound  <-- unknown dependencies "gi" + others
from pathlib import Path, PurePath
import simpleaudio as sa
import sys
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from playsounds.playsounds import playwavfile

# wave_audio = np.sin(np.linspace(0, 3000, 20000))

# Audio(wave_audio, rate=20000)   # Option 1 (doesn't seem to work -- 
#             # shows widget in Jupyter Interactive window but won't play)

# wav_wave = np.array(wave_audio, dtype=np.int16)
# sd.play(wav_wave, 20000)   # Option 2 doesn't work at all -- "PortAudio library not found"


# sr, wdata = wavfile.read('test_sound.wav')

# p = pyaudio.PyAudio()
# stream = p.open(format = p.get_format_from_width(1), channels = 1, rate = sr, output = True)
# stream.write(wdata)
# stream.stop_stream()s
# stream.close()
# p.terminate()

# playsound(f'{Path(__file__).parent}/test_sound.wav')


listeningMaxDuration = 3
filename = 'C:/Users/u0137935/source/repos/PaulESAT/sounds-phd/01_algorithms/01_NR/02_distributed/dev/test_sound.wav'
playwavfile(filename, listeningMaxDuration)


# %%
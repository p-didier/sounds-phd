from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import numpy as np
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from mySTFT.calc_STFT import calcISTFT, calcSTFT


Fs = 16e3
T = 2
x = np.random.random(int(T*Fs))
N_STFT = 1024
win = np.hanning(N_STFT)
R_STFT = N_STFT / 2

X, freqs = calcSTFT(x, Fs, win, N_STFT, R_STFT, sides='onesided')
if len(X.shape) == 2:
    X = X[:, :, np.newaxis]

# x2 = calcISTFT(X, win, N_STFT, R_STFT, sides='onesided')

# plt.figure
# plt.plot(x)
# plt.plot(x2)
# plt.show()

# stop = 1

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background


seed = 1
rng = np.random.default_rng(seed)

fs = 16e3
N = 2**12
y = 2 * rng.random(N) - 1
t = np.arange(N) / fs
win = np.hanning(N)
Y = np.fft.fft(y * win, n=N)
Ynowin = np.fft.fft(y, n=N)
fhalf = np.arange(N/2)/N*fs
f = np.concatenate((-np.flip(fhalf), fhalf))

fig = plt.figure()
ax = fig.add_subplot(311)
plt.plot(t, y, label='$y$')
plt.plot(t, y * win, label='$y\cdot w$')
plt.legend()
plt.grid()
ax = fig.add_subplot(312)
plt.plot(f, np.abs(Ynowin))
plt.grid()
plt.ylim([0, np.amax(np.abs(Ynowin))])
plt.title('FFT($y$)')
ax = fig.add_subplot(313)
plt.plot(f, np.abs(Y))
plt.grid()
plt.ylim([0, np.amax(np.abs(Y))])
plt.title('FFT($y\cdot w$)')
fig.tight_layout()

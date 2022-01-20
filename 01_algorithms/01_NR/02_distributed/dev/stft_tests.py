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

x2 = calcISTFT(X, win, N_STFT, R_STFT, sides='onesided')

plt.figure
plt.plot(x)
plt.plot(x2)
plt.show()

stop = 1


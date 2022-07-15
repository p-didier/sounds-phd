from numba import njit
import time
from rimPy import rimPy
import numpy as np

mic_pos = np.array([1,2,3])
source_pos = np.array([3,4,5])
room_dim = np.array([7,7,7])
beta = np.zeros((2,3))
rir_length = 2**4
Fs = 16e3

for ii in range(1000):
    t = time.perf_counter()
    rimPy(mic_pos, source_pos, room_dim, beta, rir_length, Fs)
    print(f'{time.perf_counter() - t} seconds')



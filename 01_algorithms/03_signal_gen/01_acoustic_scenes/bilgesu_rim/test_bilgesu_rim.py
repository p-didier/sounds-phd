
from bilgesu_rim import ISM
import numpy as np

def main():

    alpha = 0.5
    mic_pos = [[0.1,0.1,0.1],]
    source_pos = [1,2,3]
    room_dim = [5,6,7]
    rir_length = 2**11
    Fs = 16e3
    beta = -np.sqrt(1 - alpha)

    h, Sr = ISM(mic_pos, source_pos, room_dim, beta, rir_length, Rd, Sr, Tw, Fc, Fs, c)

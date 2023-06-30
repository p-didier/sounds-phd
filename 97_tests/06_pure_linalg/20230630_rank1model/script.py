# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

TARGET_SIGNAL = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
N_SENSORS = 2
RIR_LENGTH = 1000
SELFNOISE_FACTOR = 0.05
DURATION = 5  # seconds

def main():
    """Main function (called by default when running script)."""
    
    # Load target signal
    targetSignal, fs = sf.read(TARGET_SIGNAL)
    targetSignal = targetSignal[:int(DURATION * fs)]
    
    # Generate clean signals
    mat = np.tile(targetSignal, (3, 1)).T
    cleanSigs = mat @ np.diag([1, 0.75, 0.5])  # rank-1 matrix (scalings only)
    r = np.linalg.matrix_rank(cleanSigs.T @ cleanSigs)
    print(f'Rank of clean-signals matrix Rss: {r}')

    # # Generate RIRs
    # rirs = np.zeros((RIR_LENGTH, N_SENSORS))
    # for n in range(N_SENSORS):
    #     rir = np.zeros(RIR_LENGTH)
    #     # rir[np.random.randint(low=0, high=RIR_LENGTH // 2, dtype=int)] =\
    #     #     np.random.uniform(low=0.5, high=1)
    #     rir[0] = np.random.uniform(low=0.5, high=1)
    #     # rir[0] = 1
    #     rirs[:, n] = rir

    # Generate noisy signals
    noisySignals = np.zeros((len(targetSignal), N_SENSORS))
    for n in range(N_SENSORS):
        noisySignals[:, n] = cleanSigs[:, n] + SELFNOISE_FACTOR *\
            np.random.uniform(low=-1, high=1, size=len(targetSignal))
    
    r = np.linalg.matrix_rank(noisySignals.T @ noisySignals)
    print(f'Rank of noisy-signals matrix Ryy: {r}')
    
    # # Plot signals
    # fig, axs = plt.subplots(1, 1)
    # for n in range(N_SENSORS):
    #     axs.plot(noisySignals[:, n], label=f'Sensor {n}')
    # plt.legend()
    # plt.show()

    stop = 1

if __name__ == '__main__':
    sys.exit(main())
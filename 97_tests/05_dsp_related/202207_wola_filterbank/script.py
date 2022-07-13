import sys
import core.utils as c
import soundfile as sf
import numpy as np

p = c.Params(
    T=2,    # signal duration [s]
    Nh=512, # analysis/synthesis windows size [samples]
    R=256,  # window shift [samples]
    seed=12345, # random generator seed
)

def main():
    signalPath = 'U:/py/sounds-phd/02_data/00_raw_signals/01_speech/speech1.wav'

    # Load signal
    x, fs = sf.read(signalPath)
    x = x[:int(p.T * fs)]

    # Create random generator
    rng = np.random.default_rng(p.seed)

    # Generate random complex coefficients (freq.-domain filter)
    w = 2 * rng.random(p.Nh // 2 + 1) - 1 + 1j * (2 * rng.random(p.Nh // 2 + 1) - 1)
    w = w[:, np.newaxis]
    w[0, :] = w[0, :].real      # Set DC to real value
    w[-1, :] = w[-1, :].real    # Set Nyquist to real value
    w = np.concatenate((w, np.flip(w[:-1, :].conj(), axis=0)[:-1, :]), axis=0)  # make symmetric
    w = np.squeeze(w)

    # Run WOLA
    z = c.wola_fd_broadcast(x, p.Nh, p.R, w)

    # Run TD-WOLA "equivalent"
    z2 = c.wola_td_broadcast(x, p.Nh, p.R, w)

    # Plot
    c.plotit(x, y, w)


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
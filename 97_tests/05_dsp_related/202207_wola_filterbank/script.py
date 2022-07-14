import sys
import core.utils as c
import soundfile as sf
import numpy as np
import copy

p = c.Params(
    T=1,    # signal duration [s]
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

    # w = np.ones_like(w)

    # Run WOLA
    z, chunks = c.wola_fd_broadcast(x, p.Nh, p.R, w)

    # Run TD-WOLA "equivalent"
    ztd = c.wola_td_broadcast(x, p.Nh, w, updateFilterEveryXsamples=1000)
    # Subdivide in blocks and go to freq. domain
    nchunks = z.shape[1] - 1
    zchunkedtd = np.zeros((p.Nh, nchunks))
    zchunked = np.zeros((p.Nh, nchunks), dtype=complex)
    for ii in range(nchunks):
        zchunkedtd[:, ii] = copy.copy(ztd[(ii * p.R):(ii * p.R + p.Nh)])
        zchunked[:, ii] = np.fft.fft(zchunkedtd[:, ii] * np.sqrt(np.hanning(p.Nh)), p.Nh, axis=0)

    # import simpleaudio as sa
    # import time
    # audio_array = x * 32767 / max(abs(x))
    # audio_array = audio_array.astype(np.int16)
    # sa.play_buffer(audio_array,1,2,fs)
    # time.sleep(p.T)
    # audio_array = ztd * 32767 / max(abs(ztd))
    # audio_array = audio_array.astype(np.int16)
    # sa.play_buffer(audio_array,1,2,fs)

    
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # # axes.plot(ztd / np.amax(ztd))
    # # axes.plot(x / np.amax(x))
    # # axes[0].plot(chunks.T / np.amax(chunks))
    # # axes[1].plot(zchunkedtd / np.amax(zchunkedtd))
    # # axes.plot(20*np.log10(np.abs(zchunked)))
    # # axes.plot(20*np.log10(np.abs(z)))
    # # axes.grid()
    # plt.tight_layout()	
    # plt.show()

    # Plot
    c.plotit(z, zchunked, 100, unwrapPhase=True)

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
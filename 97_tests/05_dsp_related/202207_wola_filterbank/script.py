import sys
import core.utils as c
import soundfile as sf
import numpy as np
import copy
import matplotlib.pyplot as plt

p = c.Params(
    T=1,    # signal duration [s]
    Nh=512, # analysis/synthesis windows size [samples]
    R=256,  # window shift [samples]
    seed=12345, # random generator seed
)

def main():
    # # Load signal
    # signalPath = 'U:/py/sounds-phd/02_data/00_raw_signals/01_speech/speech1.wav'
    # x, fs = sf.read(signalPath)
    # x = x[20000:20000+int(p.T * fs)]
    # # x = x[:int(p.T * fs)]

    # Create tonal signal 
    freq = 1000  # frequency [Hz]
    fs = 16000  # sampling rate [Hz]
    x = np.sin(2 * np.pi * freq * np.arange(int(p.T * fs)) / fs)

    # Define window
    h = np.sqrt(np.hanning(p.Nh))   # root-Hann
    # h = np.ones(p.Nh)               # rectangular

    # Create random generator
    rng = np.random.default_rng(p.seed)

    # # Generate random complex coefficients (freq.-domain filter)
    # w = 2 * rng.random(p.Nh // 2 + 1) - 1 + 1j * (2 * rng.random(p.Nh // 2 + 1) - 1)
    # w = w[:, np.newaxis]
    # w[0, :] = w[0, :].real      # Set DC to real value
    # w[-1, :] = w[-1, :].real    # Set Nyquist to real value
    # w = np.concatenate((w, np.flip(w[:-1, :].conj(), axis=0)[:-1, :]), axis=0)  # make symmetric
    # w = np.squeeze(w)

    # Load RIR
    wTD, fsRIR = sf.read('U:/py/sounds-phd/97_tests/05_dsp_related/00_signals/rir1.wav')
    wTD = wTD[:p.Nh]                    # truncate
    w = np.fft.fft(wTD, p.Nh, axis=0)   # freq-domain coefficients
    # w = np.ones_like(w)   # uncomment for no filtering at all

    # Run WOLA
    z, xchunks, zchunks = c.wola_fd_broadcast(x, p.Nh, p.R, w, h)

    # Run TD-WOLA "equivalent"
    ztd = c.wola_td_broadcast(x, p.Nh, w, h, updateFilterEveryXsamples=None)
    # ztd = c.wola_td_broadcast(x, p.Nh, w, updateFilterEveryXsamples=1000)

    # Run TD-WOLA "naive" (direct filtering by `w`)
    ztd_naive = c.wola_td_broadcast_naive(x, w)

    # Subdivide in blocks and go to freq. domain
    nchunks = z.shape[1] - 1
    zchunkedtd = np.zeros((p.Nh, nchunks))
    zchunked = np.zeros((p.Nh, nchunks), dtype=complex)
    zchunkedtd_naive = np.zeros((p.Nh, nchunks))
    zchunked_naive = np.zeros((p.Nh, nchunks), dtype=complex)
    xchunksfd = np.zeros((p.Nh, nchunks), dtype=complex)
    for ii in range(nchunks):
        idxBeg = ii * p.R
        idxEnd = idxBeg + p.Nh
        zchunkedtd[:, ii] = copy.copy(ztd[idxBeg:idxEnd])
        zchunked[:, ii] = np.fft.fft(zchunkedtd[:, ii] * h, p.Nh, axis=0)
        # zchunked[:, ii] = np.fft.fft(zchunkedtd[:, ii], p.Nh, axis=0)
        zchunkedtd_naive[:, ii] = copy.copy(ztd_naive[idxBeg:idxEnd])
        zchunked_naive[:, ii] = np.fft.fft(zchunkedtd_naive[:, ii] * h, p.Nh, axis=0)
        xchunksfd[:, ii] = np.fft.fft(x[idxBeg:idxEnd] * h, p.Nh, axis=0)

    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # idx = 10
    # axes.plot(zchunks[idx, :], label='$F^{{-1}}\\{w\\cdot F\\{ x(l)\\}\\}$')
    # axes.plot(zchunkedtd_naive[:, idx], label='$(x \\ast F^{{-1}}\\{ w \\})[l]$')
    # axes.legend()
    # plt.tight_layout()	
    # plt.show()

    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(np.real(w))
    # axes.plot(np.imag(w))
    # axes.legend()
    # plt.tight_layout()	
    # plt.show()

    # import simpleaudio as sa
    # import time
    # audio_array = x * 32767 / max(abs(x))
    # audio_array = audio_array.astype(np.int16)
    # sa.play_buffer(audio_array,1,2,fs)
    # time.sleep(p.T)
    # audio_array = ztd * 32767 / max(abs(ztd))
    # audio_array = audio_array.astype(np.int16)
    # sa.play_buffer(audio_array,1,2,fs)
    # time.sleep(p.T)
    # audio_array = ztd_naive * 32767 / max(abs(ztd_naive))
    # audio_array = audio_array.astype(np.int16)
    # sa.play_buffer(audio_array,1,2,fs)

    # fig, axes = plt.subplots(1,1)
    # fig.set_size_inches(8.5, 3.5)
    # axes.plot(x, label='original $x$')
    # axes.plot(ztd, label='$z$ TD, equivalent WOLA')
    # # axes.plot(ztd_naive, label='$z$ TD, naive')
    # axes.legend()
    # plt.tight_layout()	
    # plt.show()

    # Plot
    # c.plotit(xchunksfd, z, zchunked, 10, unwrapPhase=False, plotFullSpectrum=False)
    c.plotit(xchunksfd, z, zchunked_naive, 10, unwrapPhase=False)

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
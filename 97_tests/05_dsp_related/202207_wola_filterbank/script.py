import sys

from regex import F
import core.utils as c
import numpy as np
import copy
import matplotlib.pyplot as plt

p = c.Params(
    T=1,    # signal duration [s]
    Nh=512, # analysis/synthesis windows size [samples]
    R=256,  # window shift [samples]
    #
    signalType='speech',        # speech input signal (loaded from file)
    # signalType='tone',          # tonal input signal
    #
    filterType='rir',       # room impulse response (loaded from file)
    # filterType='random',    # random filter
    # filterType='none',       # no filtering
    #
    windowType='roothann',  # square-root-Hann window
    # windowType='rect'       # rectangular window
    #
    seed=12345, # random generator seed
)

def main():

    # Load signal 
    x, fs = c.load_signal(p.signalType, p.T, 16000, plotit=False, listen=False)

    # Generate filter
    w = c.get_filter(p.filterType, p.Nh, p.seed)

    # Generate window
    h = c.get_window(p.windowType, p.Nh)

    # Run regular WOLA (same as in DANSE scripts)
    z, xchunks, zchunks = c.wola_fd_broadcast(x, p.Nh, p.R, w, h)

    # Run TD-WOLA "equivalent"
    ztd = c.wola_td_broadcast(x, p.Nh, w, h, plotDistFct=False)

    # Run TD-WOLA "naive" (direct filtering by `w`)
    ztd_naive = c.wola_td_broadcast_naive(x, w)

    # Subdivide in blocks and go to freq. domain
    nchunks = z.shape[1]
    zchunkedtd = np.zeros((p.Nh, nchunks))
    zchunked = np.zeros((p.Nh, nchunks), dtype=complex)
    zchunkedtd_naive = np.zeros((p.Nh, nchunks))
    zchunked_naive = np.zeros((p.Nh, nchunks), dtype=complex)
    xchunksfd = np.zeros((p.Nh, nchunks), dtype=complex)
    for ii in range(nchunks):
        idxBeg = ii * p.R
        idxEnd = idxBeg + p.Nh
        if idxEnd > len(ztd):
            break
        zchunkedtd[:, ii] = copy.copy(ztd[idxBeg:idxEnd])
        zchunked[:, ii] = np.fft.fft(zchunkedtd[:, ii] * h, p.Nh, axis=0)
        # zchunked[:, ii] = np.fft.fft(zchunkedtd[:, ii], p.Nh, axis=0)
        zchunkedtd_naive[:, ii] = copy.copy(ztd_naive[idxBeg:idxEnd])
        zchunked_naive[:, ii] = np.fft.fft(zchunkedtd_naive[:, ii] * h, p.Nh, axis=0)
        xchunksfd[:, ii] = np.fft.fft(x[idxBeg:idxEnd] * h, p.Nh, axis=0)

    # Plot
    c.plotit(xchunksfd, z, zchunked, 10, unwrapPhase=False, plotFullSpectrum=False)
    c.plotit(xchunksfd, z, zchunked_naive, 10, unwrapPhase=False)

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
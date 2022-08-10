
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

Nh = 2048
R = Nh // 2
h = np.sqrt(np.hanning(Nh))
T = 0.5
filteringType = 'rir'
# filteringType = 'none'

def main():

    x, fs = sf.read('U:/py/sounds-phd/02_data/00_raw_signals/01_speech/speech1.wav')
    x = x[25000:25000 + int(T * fs)]

    if filteringType == 'rir':
        wTD, _ = sf.read('U:/py/sounds-phd/97_tests/05_dsp_related/00_signals/rir1.wav')
        wTD = wTD[:Nh]
        wTD /= np.amax(np.abs(wTD))
        wTD *= 0.25
    elif filteringType == 'none':
        wTD = np.zeros(Nh)
        wTD[0] = 1
    w = np.fft.fft(wTD, Nh, axis=0)

    x_WOLAdom, y_td, y_tdChunks, x_WOLAdom2, y_td2, y_tdChunks2 = wola_two_passes(x, R, Nh, h, w)

    # Plot
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(x, label='Input')
    axes.plot(y_td, label='1st pass output')
    axes.plot(y_td2, label='2nd pass output')
    axes.legend()
    axes.grid()
    plt.tight_layout()

    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(6.5, 2.5)
    mapp = axes[0].pcolormesh(20*np.log10(np.abs(x_WOLAdom[:int(Nh // 2), :])))
    plt.colorbar(mapp, ax=axes[0])
    mapp = axes[1].pcolormesh(20*np.log10(np.abs(x_WOLAdom2[:int(Nh // 2), :])))
    plt.colorbar(mapp, ax=axes[1])
    plt.tight_layout()	

    plt.show()

    return 0


def wola_two_passes(x, R, Nh, h, w):

    numChunks = int(len(x) // R - 1)
    x_WOLAdom = np.zeros((Nh, numChunks), dtype=complex)
    y_td = np.zeros_like(x)
    y_tdChunks = np.zeros((Nh, numChunks))
    for ii in range(numChunks):

        idxBeg = ii * R
        idxEnd = idxBeg + Nh
        Xcurr = np.fft.fft(x[idxBeg:idxEnd] * h, Nh, axis=0)

        Xcurr *= w
        x_WOLAdom[:, ii] = Xcurr

        ycurr = np.fft.ifft(Xcurr, Nh, axis=0)
        ycurr = np.real_if_close(ycurr)
        y_tdChunks[:, ii] = ycurr

        y_td[idxBeg:idxEnd] += ycurr * h

    x_WOLAdom2 = np.zeros((Nh, numChunks), dtype=complex)
    y_td2 = np.zeros_like(x)
    y_tdChunks2 = np.zeros((Nh, numChunks))
    for ii in range(numChunks):

        idxBeg = ii * R
        idxEnd = idxBeg + Nh
        Xcurr = np.fft.fft(y_td[idxBeg:idxEnd] * h, Nh, axis=0)

        x_WOLAdom2[:, ii] = Xcurr

        ycurr = np.fft.ifft(Xcurr, Nh, axis=0)
        ycurr = np.real_if_close(ycurr)
        y_tdChunks2[:, ii] = ycurr

        y_td2[idxBeg:idxEnd] += ycurr * h

    return x_WOLAdom, y_td, y_tdChunks, x_WOLAdom2, y_td2, y_tdChunks2


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
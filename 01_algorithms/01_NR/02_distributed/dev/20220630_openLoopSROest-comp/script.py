import numpy as np
import matplotlib.pyplot as plt
import sys
import fcns.myfcns as f
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search

# Simulation parameters
p = f.simParams(
    T=10,        # signal duration [s]
    eps=100,
    blockSize=2048,
    overlap=0.5,
    ld=1,
    alpha=.95
)

def main():

    # Generate signals
    y = f.Signal(T=p.T, nChannels=2)
    y.apply_sro(p.eps, channelIdx=1)
    # y.plot_waveform()

    # WOLA params
    analysisWin = np.sqrt(np.hanning(p.blockSize))
    analysisWin = analysisWin[:, np.newaxis]

    # Block-wise processing
    l = 0
    acr = 0     # initial average coherences product
    nChunks = int(y.data.shape[0] // (p.blockSize * (1 - p.overlap)))
    yyH = np.zeros((nChunks, int(p.blockSize / 2 + 1), y.nChannels, y.nChannels), dtype=complex)
    sroEst = np.zeros(nChunks)

    flagFirstEstimate = True
    while 1:
        # Corresponding time instant
        t = int(l * p.blockSize * (1 - p.overlap) + p.blockSize) / y.fs[0]
        idxEnd = int(t * y.fs[0])
        idxBeg = idxEnd - p.blockSize
        if idxEnd > y.data.shape[0] - 1:
            break   # breaking point

        # Current chunk
        ycurr = y.data[idxBeg:idxEnd, :]

        # Go to freq. domain
        ycurrhat = 1 / analysisWin.sum() * np.fft.fft(ycurr * analysisWin, p.blockSize, axis=0)
        ycurrhat = ycurrhat[:int(ycurrhat.shape[0] / 2 + 1), :]     # keep only >0 freqs.

        # Compute correlation matrix
        yyH[l, :, :, :] = np.einsum('ij,ik->ijk', ycurrhat, ycurrhat.conj())

        if l >= p.ld:

            cohPosteriori = yyH[l, :, 0, 1] / np.sqrt(yyH[l, :, 0, 0] * yyH[l, :, 1, 1])     # a posteriori coherence
            cohPriori = yyH[l - p.ld, :, 0, 1] / np.sqrt(yyH[l - p.ld, :, 0, 0] * yyH[l - p.ld, :, 1, 1])     # a posteriori coherence
                
            res_prod = cohPosteriori * cohPriori.conj()
            # Prep for ISTFT (negative frequency bins too)
            res_prod = np.concatenate(
                [res_prod[:-1],
                    np.conj(res_prod)[::-1][:-1]],
                -1
            )
            # Update the average coherence product
            if flagFirstEstimate:
                acr = res_prod     # <-- 1st SRO estimation, no exponential averaging (initialization)
                flagFirstEstimate = False
            else:
                acr = p.alpha * acr + (1 - p.alpha) * res_prod  

            # Estimate SRO
            sroEst[l] = - max_time_lag_search(acr) / (p.ld * int(p.blockSize * (1 - p.overlap)))

        l += 1

    # Plot
    f.plotit(sroEst, gt=p.eps)

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------

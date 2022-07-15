from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
import fcns.myfcns as f

# Simulation parameters
p = f.SimParams(
    T=10,        # signal duration [s]
    eps=100,
    blockSize=2048,
    overlap=0.5,
    # ld=np.linspace(1, 10, num=5, dtype=int),
    ld=5,
    # alpha=np.linspace(0, 1, num=5, endpoint=False)
    alpha=0.95,
    compensateSRO=True,
    alphaEps=0.01,
    # compType='closedloop',
    compType='openloop',
)

def main():

    # Generate signals
    y = f.Signal(T=p.T, nChannels=2)
    y.apply_sro(p.eps, channelIdx=1)
    # y.plot_waveform()

    lenAlpha = 1
    lenLd = 1
    legLabels = None
    if isinstance(p.alpha, np.ndarray):
        lenAlpha = len(p.alpha)
        legLabels = [f'$\\alpha$ = {a}' for a in p.alpha]
    elif isinstance(p.ld, np.ndarray):
        lenLd = len(p.ld)
        legLabels = [f'$l_\\mathrm{{d}}$ = {a}' for a in p.ld]

    # Number of signal frames
    nFrames = int(y.data.shape[0] // (p.blockSize * (1 - p.overlap)) + 1)
    # Initialize results array
    sroEst = np.zeros((nFrames, lenAlpha, lenLd))
    sroResEst = np.zeros((nFrames, lenAlpha, lenLd))
    pcopy = copy(p)
    for ii in range(lenAlpha):
        if lenAlpha > 1:
            pcopy.alpha = p.alpha[ii]            
        for jj in range(lenLd):
            if lenLd > 1:
                pcopy.ld = p.ld[jj]
            print(f'Computing SRO for alpha={pcopy.alpha} & ld={pcopy.ld}...')
            sroEst[:, ii, jj], sroResEst[:, ii, jj] = f.run(y, pcopy)     # <-- run SRO estimation
    sroEst = sroEst.squeeze()           # eliminate unused dimensions
    sroResEst = sroResEst.squeeze()     # eliminate unused dimensions
    print('All done. Plotting...')

    # Plot
    f.plotit(sroEst, sroResEst, legLabels, gt=p.eps)

    stop = 1


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------

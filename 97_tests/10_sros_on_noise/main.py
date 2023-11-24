# Purpose of script:
# Check whether one can apply SROS on noise signals.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from resampy import resample
import matplotlib.pyplot as plt

FS = 16000
N = 1000
SRO = 20000  # [PPM]

def main():
    """Main function (called by default when running script)."""
    # Generate noise signal
    noise = np.random.randn(N)
    
    sro = SRO / 1e6

    # Apply SRO by resampling
    fs = FS * (1 + sro)
    noise_sro = resample(
        noise, sr_orig=FS, sr_new=fs,
    )

    # Compute time-domain correlation
    corr = np.correlate(noise, noise_sro, mode='full')
    corrBasis = np.correlate(noise, noise, mode='full')
    corrRandom = np.correlate(noise, np.random.randn(N), mode='full')
    
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    # Set position of figure
    fig.canvas.manager.window.move(0,0)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    axes.plot(corrBasis, label='Correlation with itself')
    axes.plot(corr, '--', label='Correlation')
    axes.plot(corrRandom, label='Correlation with random noise')
    axes.grid()
    axes.set_title(f'Correlation between noise and noise with SRO={SRO} PPM')
    axes.set_xlabel('Time lag [samples]')
    axes.set_ylabel('Correlation')
    axes.legend()
    fig.tight_layout()
    plt.show(block=False)

    stop = 1

if __name__ == '__main__':
    sys.exit(main())
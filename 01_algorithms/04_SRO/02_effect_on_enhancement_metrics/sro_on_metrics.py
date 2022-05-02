from email.mime import base
import sys
import fcns
import numpy as np
import resampy
import scipy.signal as sig
import matplotlib.pyplot as plt

params = fcns.Parameters(
    signalPath='02_data/00_raw_signals/01_speech/speech1.wav',
    srosToTest=np.arange(0,110,step=10),    # [ppm]
    metricsToCompute=['fwSNRseg', 'STOI', 'PESQ'],
    baseFs=44000
)

def main():

    # Get signal
    mysig, fs = fcns.get_signal(params.signalPath)
    # Get signal to base sampling frequency
    mysig = resampy.core.resample(mysig, fs, params.baseFs)       # SRO as resampling


    # Initialize arrays
    metricsForPlot = dict()
    for ii in range(len(params.metricsToCompute)):
        metricsForPlot.update({params.metricsToCompute[ii]: []})

    # Apply SROs
    sigs = np.zeros((len(mysig), len(params.srosToTest)))
    for ii in range(len(params.srosToTest)):
        print(f'Computing speech enhancement metrics for SRO #{ii+1}/{len(params.srosToTest)}...')
        
        fsSRO = params.baseFs * (1 + params.srosToTest[ii] * 1e-6)     # SRO definition
        sigWithSRO = resampy.core.resample(mysig, params.baseFs, fsSRO)       # SRO as resampling
        sigWithSRO = sig.resample(mysig, int(np.floor(len(mysig) * fsSRO / params.baseFs)))
        sigWithSRO = sigWithSRO[:len(mysig)]
        sigs[:, ii] = sigWithSRO


        # Compute metrics
        metrics = fcns.compute_metrics(mysig, sigWithSRO, params.baseFs, params.metricsToCompute)
        # metrics = fcns.compute_metrics(mysig, sigWithSRO, fsSRO, params.metricsToCompute)

        # Store metrics
        for metricName in metrics.keys():
            if metrics[metricName] is None:     # <-- can happen, e.g., if PESQ is not calculate because fs \neq 16 or 8 kHz
                metricsForPlot.pop(metricName, None)
            else:
                metricsForPlot[metricName].append(metrics[metricName])

    print('All done. Plotting...')

    # Visualize results
    fcns.plot_metrics(metricsForPlot, params.srosToTest)

    return None



# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
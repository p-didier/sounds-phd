from cProfile import label
import sys
import fcns
import numpy as np
import resampy
import scipy.signal as sig
import matplotlib.pyplot as plt

params = fcns.Parameters(
    signalPath='02_data/00_raw_signals/01_speech/speech1.wav',
    srosToTest=np.arange(0,110,step=10),    # [ppm]
    metricsToCompute=['STOI', 'fwSNRseg', 'PESQ'],
    baseFs=16000
)

def main():

    # Get signal
    mysig, fs = fcns.get_signal(params.signalPath)
    # Get signal to base sampling frequency
    mysig = resampy.core.resample(mysig, fs, params.baseFs)       # SRO as resampling


    # Initialize arrays
        # case1: original signal vs. SRO-affected copy
        # case2: original signal vs. (original signal + SRO-affected copy)
    metricsForPlot_case1 = dict()
    metricsForPlot_case2 = dict()
    for ii in range(len(params.metricsToCompute)):
        metricsForPlot_case1.update({params.metricsToCompute[ii]: []})
        metricsForPlot_case2.update({params.metricsToCompute[ii]: []})

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
            # case1: original signal vs. SRO-affected copy
            # case2: original signal vs. (original signal + SRO-affected copy)
        metrics_case1 = fcns.compute_metrics(mysig, sigWithSRO, params.baseFs, params.metricsToCompute)
        metrics_case2 = fcns.compute_metrics(mysig, sigWithSRO + mysig, params.baseFs, params.metricsToCompute)
        # metrics = fcns.compute_metrics(mysig, sigWithSRO, fsSRO, params.metricsToCompute)

        # Store metrics
        for metricName in metrics_case1.keys():
            if metrics_case1[metricName] is None:     # <-- can happen, e.g., if PESQ is not calculate because fs \neq 16 or 8 kHz
                metricsForPlot_case1.pop(metricName, None)
                metricsForPlot_case2.pop(metricName, None)
            else:
                metricsForPlot_case1[metricName].append(metrics_case1[metricName])
                metricsForPlot_case2[metricName].append(metrics_case2[metricName])

    print('All done. Plotting...')

    # Visualize results
    plt.style.use('95_style_sheets/plot1.mplstyle')
    metricNames = list(metricsForPlot_case1.keys())
    fig, axes = plt.subplots(1,len(metricNames))
    fig.set_size_inches(12.5, 4)
    fcns.plot_metrics(axes, metricsForPlot_case1, params.srosToTest, color='C0')
    fcns.plot_metrics(axes, metricsForPlot_case2, params.srosToTest, color='C1')
    axes[0].legend(handles=[axes[0].lines[0], axes[0].lines[-1]], labels=['$x$ vs. $x_\mathrm{sro}$',
        '$x$ vs. $(x_\mathrm{sro} + x)$'],loc='lower left')
    if 1:
        fig.savefig(f'01_algorithms/04_SRO/02_effect_on_enhancement_metrics/figs/out.png')
        fig.savefig(f'01_algorithms/04_SRO/02_effect_on_enhancement_metrics/figs/out.pdf')

    return None



# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
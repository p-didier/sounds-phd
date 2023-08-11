# Purpose of script:
# Plot the results of the test battery.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
from test_battery import TestOutput, TestParameters

# Get current file folder
FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER_NAME = f'{FILE_FOLDER}/results'
RESULTS_FILE_NAME = 'rank1model_fs8kHz.npz'

# Global variables
FIGSIZE = (12, 4)  # Figure size
TMAX = 30  # [s] Maximum duration of the simulated data
TO_COMPUTE = [
    'gevdmwf_batch',  # GEVD-MWF (batch)
    'gevdmwf_online',  # GEVD-MWF (online) 
    'gevddanse_sim_online',  # GEVD-DANSE (online), simultaneous node-updating
]
EXPORT_FIGURES = True  # Whether to export figures to PDF and PNG files
EXPORT_PATH = f'{FILE_FOLDER}/figs/battery_test/20230809_tests'  # Path to export figures to
TAUS = [2., 4., 8.]  # [s] Time constants for exp. avg. in online filters

def main():
    """Main function (called by default when running script)."""
    # Load compressed results file
    npz = np.load(
        f'{RESULTS_FOLDER_NAME}/{RESULTS_FILE_NAME}',
        allow_pickle=True
    )
    results = npz['allOutputs']

    # Signal durations tested
    durations = np.logspace(np.log10(1), np.log10(TMAX), 30)
    
    # Plot results for Type-1/2 tests
    figs = plot_results(
        [out for out in results if out.name == 'test1' or out.name == 'test2'],
        xAxisBatch=durations,
        taus=TAUS,
    )
    if EXPORT_FIGURES:
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        for fig in figs:
            fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.pdf', bbox_inches='tight')
            fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.png', bbox_inches='tight', dpi=300)

    # Plot results for Type-3 (full Monte-Carlo) tests
    fig = plot_results_mc_tests(
        [out for out in results if 'SEED' in out.description],
        xAxisBatch=durations,
        indivRunsFig=False,
        taus=TAUS,
    )
    if EXPORT_FIGURES:
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.pdf', bbox_inches='tight')
        fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.png', bbox_inches='tight', dpi=300)

    # Show figures
    plt.show(block=False)

    print('ALL DONE.')
    stop = 1


def generate_plots(baseColor, filterType, data, axes, xAxisBatch, taus):
    """Generate plots."""

    lineStyles = ['-', '--', '-.', ':']

    if 'online' in filterType:
        xAxisOnline = np.linspace(
            start=0,
            stop=np.amax(xAxisBatch),
            num=data.shape[1]
        )
        for idxTau in range(len(taus)):
            axes.fill_between(
                xAxisOnline,
                np.amin(data[:, :, idxTau], axis=0),
                np.amax(data[:, :, idxTau], axis=0),
                color=baseColor,
                alpha=0.15
            )
            idxLineStyles = idxTau % len(lineStyles)
            axes.loglog(
                xAxisOnline,
                np.mean(
                    data[:, :, idxTau],
                    axis=0
                ),
                f'{baseColor}{lineStyles[idxLineStyles]}',
                label=f'{filterType} ($\\tau=${taus[idxTau]} s)'
            )
    else:
        xAxisOnline = None
        axes.fill_between(
            xAxisBatch,
            np.amin(data, axis=0),
            np.amax(data, axis=0),
            color=baseColor,
            alpha=0.15
        )
        axes.loglog(
            xAxisBatch,
            np.mean(
                data,
                axis=0
            ),
            f'{baseColor}o-',
            label=filterType
        )
    return xAxisOnline


def plot_results(resAll: list[TestOutput], xAxisBatch, taus) -> list[plt.Figure]:
    """Plot results for Type-1 and Type-2 test."""
    
    figs = []
    for res in resAll:
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(FIGSIZE)
        for ii, filterType in enumerate(TO_COMPUTE):
            xAxisOnline = generate_plots(
                f'C{ii}',
                filterType,
                res.results[filterType],
                axes,
                xAxisBatch,
                taus
            )
        axes.legend()
        axes.grid(True, which='both')
        axes.set_xlabel('Duration [s]')
        axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
        # Adapt y-axis limits to the data
        ymin, ymax = compute_yaxis_limits(res.results, xAxisOnline, xAxisBatch)
        axes.set_ylim([ymin, ymax])
        axes.set_title(f'$K={res.parameters.K}$, $M={res.parameters.M}$, $M_k={res.parameters.Mk}$ ({res.results[filterType].shape[0]} MC runs)')
        fig.set_label(res.name)
        figs.append(fig)

    return figs


def plot_results_mc_tests(
        res: list[TestOutput],
        xAxisBatch,
        indivRunsFig=False,
        taus=None
    ):
    """Plot results for Type-3 (full Monte-Carlo) tests."""

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(FIGSIZE)
    if indivRunsFig:
        for testOutput in res:
            if 'SEED' in testOutput.description:
                for ii, filterType in enumerate(TO_COMPUTE):
                    xAxisOnline = generate_plots(
                        f'C{ii}',
                        filterType,
                        testOutput.results[filterType],
                        axes,
                        xAxisBatch,
                        taus
                    )
    else:  # average over all runs
        # Build data to plot
        dataToPlot = dict([(filterType, None) for filterType in TO_COMPUTE])
        for ii, filterType in enumerate(TO_COMPUTE):
            # Combine all MC runs along one axis
            comb = res[0].results[filterType]
            for jj in range(len(res) - 1):
                comb = np.concatenate(
                    (
                        comb,
                        res[jj + 1].results[filterType]
                    ),
                    axis=0
                )
            dataToPlot[filterType] = comb
        
        # Plot
        for ii, filterType in enumerate(TO_COMPUTE):
            xAxisOnline = generate_plots(
                f'C{ii}',
                filterType,
                dataToPlot[filterType],
                axes,
                xAxisBatch,
                taus
            )
        axes.set_title(f'Average over {dataToPlot[filterType].shape[0]} MC runs (random $\{{M_k\}}_{{k\\in\\mathcal{{K}}}}$)')

    axes.legend()
    axes.grid(True, which='both')
    axes.set_xlabel('Duration [s]')
    axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
    # Adapt y-axis limits to the data
    ymin, ymax = compute_yaxis_limits(dataToPlot, xAxisOnline, xAxisBatch)
    axes.set_ylim([ymin, ymax])
    fig.set_label('test3')

    return fig


def compute_yaxis_limits(
        data: dict[str, np.ndarray],
        xAxisOnline=None,
        xAxisBatch=None
    ):
    """Compute y-axis limits for plotting."""
    ymin, ymax = np.inf, -np.inf
    for filterType in TO_COMPUTE:
        if 'online' in filterType:
            idxStart = np.argmin(np.abs(xAxisOnline - xAxisBatch[0]))
            ymin = min(
                ymin,
                np.amin(np.mean(data[filterType][:, idxStart:, :], axis=0))
            )
            ymax = max(
                ymax,
                np.amax(np.mean(data[filterType][:, idxStart:, :], axis=0))
            )
        else:
            ymin = min(ymin, np.amin(data[filterType]))
            ymax = max(ymax, np.amax(data[filterType]))
    # Add some margin
    ymin *= 0.9
    ymax *= 1.1
    return ymin, ymax


def combine_dims(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`, 
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
    return np.reshape(a, s[:i] + [combined] + s[i+n+1:])


if __name__ == '__main__':
    sys.exit(main())
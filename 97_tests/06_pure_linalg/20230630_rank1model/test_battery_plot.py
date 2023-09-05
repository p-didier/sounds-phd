# Purpose of script:
# Plot the results of the test battery.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
import functools
import numpy as np
from test_battery import *
import matplotlib.pyplot as plt

# Get current file folder
FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER_NAME = f'{FILE_FOLDER}/results'
RESULTS_FILE_NAME = '20230905_functionalRank1/speech/betaExt1.npz'
RESULTS_FILE_NAME = '20230905_functionalRank1/_quick/quickie1.npz'

# Global variables5
FIGSIZE = (12, 4)  # Figure size
TMAX = 20  # [s] Maximum duration of the simulated data
TAUS = [2., 8.]  # [s] Time constants for exp. avg. in online filters
EXPORT_FIGURES = True  # Whether to export figures to PDF and PNG files
EXPORT_PATH = f'{FILE_FOLDER}\\figs\\battery_test\\{RESULTS_FILE_NAME[:-4]}'  # Path to export figures to
# Booleans
# SHOW_RESULTS_RANGE = True  # whether to show the range of y-axis values as a shaded area
SHOW_RESULTS_RANGE = False  # whether to show the range of y-axis values as a shaded area
# SHOW_INDIV_RUN_LINES = True  # whether to show individual MC runs as light lines
SHOW_INDIV_RUN_LINES = False  # whether to show individual MC runs as light lines

# VAD parameters
FS = 8000  # [Hz] Sampling frequency
EFFECTIVE_FRAME_SIZE = 512  # [samples] Effective frame size
INTERRUPTION_DURATION = 1  # [s] Duration of the interruption
INTERRUPTION_PERIOD = 2  # [s] Period of the interruption

def main():
    """Main function (called by default when running script)."""
    # Load compressed results file
    npz = np.load(
        f'{RESULTS_FOLDER_NAME}/{RESULTS_FILE_NAME}',
        allow_pickle=True
    )
    results = npz['allOutputs']

    # Signal durations tested
    # durations = np.logspace(np.log10(1), np.log10(TMAX), 30)
    durations = np.linspace(1, TMAX, 30)
    
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

    # Get name of current file
    print(f'ALL DONE ({os.path.basename(__file__)}).')
    stop = 1


def generate_plots(
        baseColor,
        filterType,
        data: np.ndarray,
        axes: plt.Axes,
        xAxisBatch,
        taus,
    ):
    """Generate plots."""

    lineStyles = ['-', '--', '-.', ':']

    if ('online' in filterType or 'wola' in filterType) and\
        'batch' not in filterType:
        xAxisOnline = np.linspace(
            start=0,
            stop=np.amax(xAxisBatch),
            num=data.shape[1]
        )
        for idxTau in range(len(taus)):
            if SHOW_RESULTS_RANGE:
                axes.fill_between(
                    xAxisOnline,
                    np.amin(data[:, :, idxTau], axis=0),
                    np.amax(data[:, :, idxTau], axis=0),
                    color=baseColor,
                    alpha=0.15
                )
            if SHOW_INDIV_RUN_LINES:
                for ii in range(data.shape[0]):
                    axes.semilogy(
                        xAxisOnline,
                        data[ii, :, idxTau],
                        f'{baseColor}{lineStyles[idxTau]}',
                        linewidth=0.5,
                        alpha=0.1
                    )
            idxLineStyles = idxTau % len(lineStyles)
            axes.semilogy(
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
        if SHOW_RESULTS_RANGE:
            axes.fill_between(
                xAxisBatch,
                np.amin(data, axis=0),
                np.amax(data, axis=0),
                color=baseColor,
                alpha=0.15
            )
        axes.semilogy(
            xAxisBatch,
            np.mean(
                data,
                axis=0
            ),
            f'{baseColor}o-',
            label=filterType
        )
    return xAxisOnline


def add_vad_line(
        axes: plt.Axes,
        durations: np.ndarray,
        vad: np.ndarray,
    ):
    """Add VAD line to plot."""
    # Generate VAD
    # dur = np.amax(durations)
    # nInterruptions = int(dur / interruptionPeriod)
    # vad = np.ones((nSamples, 1))
    # for k in range(nInterruptions):
    #     idxStart = int(
    #         ((k + 1) * interruptionPeriod - interruptionDuration) *\
    #             fsTarget
    #     )
    #     idxEnd = int(idxStart + interruptionDuration * fsTarget)
    #     vad[idxStart:idxEnd, 0] = 0
    
    # Add VAD line to plot
    axesRight = axes.twinx()
    axesRight.plot(
        np.linspace(0, np.amax(durations), len(vad)),
        vad,
        'k--',
        linewidth=0.5,
        zorder=-1
    )
    axesRight.set_ylabel('VAD (-)')
    axesRight.set_yticks([0, 1])
    axesRight.set_zorder(-1)  # put the VAD behind the other plots
    axes.set_frame_on(False)  # remove the frame of the main axes
    # Set `axes` grid only to horizontal lines
    axes.grid(axis='y', which='both')
    # Add grid to `axesRight` only to vertical lines
    axesRight.set_xticks(axes.get_xticks())
    axesRight.grid(axis='x', which='major')
    # Ensure the grid is behind the other plots
    axesRight.set_axisbelow(True)


def plot_results(resAll: list[TestOutput], xAxisBatch, taus) -> list[plt.Figure]:
    """Plot results for Type-1 and Type-2 test."""

    # Check if batch-mode is included in the test outputs (this will impact
    # the y-axis limits of the plots)
    labels = list(resAll[0].results.keys())
    flagBatch = any(['batch' in label for label in labels])
    
    figs = []
    for res in resAll:
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(FIGSIZE)
        for ii, filterType in enumerate(labels):
            xAxisOnline = generate_plots(
                f'C{ii}',
                filterType,
                res.results[filterType],
                axes,
                xAxisBatch,
                taus
            )
        # axes.legend(loc='upper right')
        # Place legend outside plot
        axes.legend(
            loc='upper left',
            # fontsize='small',
            bbox_to_anchor=(1.05, 1)
        )
        axes.set_xlabel('Duration [s]')
        if flagBatch:
            axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
            # Adapt y-axis limits to the data
            ymin, ymax = compute_yaxis_limits(res.results, xAxisOnline, xAxisBatch)
        else:
            axes.set_xlim([0, np.amax(xAxisOnline)])
            # Adapt y-axis limits to the data
            ymin, ymax = compute_yaxis_limits(res.results, xAxisOnline)
        axes.set_ylim([ymin, ymax])
        axes.set_title(f'$K={res.parameters.K}$, $M={res.parameters.M}$, $M_k={res.parameters.Mk}$ ({res.results[filterType].shape[0]} MC runs)')
        if res.vad is not None:
            add_vad_line(
                axes,
                durations=xAxisBatch,
                vad=res.vad
            )
        else:
            axes.grid(which='both')
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

    labels = list(res[0].results.keys())
    flagBatch = any(['batch' in label for label in labels])

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(FIGSIZE)
    if indivRunsFig:
        for testOutput in res:
            if 'SEED' in testOutput.description:
                for ii, filterType in enumerate(labels):
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
        dataToPlot = dict([(filterType, None) for filterType in labels])
        for ii, filterType in enumerate(labels):
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
        for ii, filterType in enumerate(labels):
            xAxisOnline = generate_plots(
                f'C{ii}',
                filterType,
                dataToPlot[filterType],
                axes,
                xAxisBatch,
                taus
            )
        axes.set_title(f'Average over {dataToPlot[filterType].shape[0]} MC runs (random $\{{M_k\}}_{{k\\in\\mathcal{{K}}}}$)')

    # axes.legend(loc='upper right')
    # Place legend outside plot
    axes.legend(
        loc='upper left',
        # fontsize='small',
        bbox_to_anchor=(1.05, 1)
    )
    axes.set_xlabel('Duration [s]')
    if flagBatch:
        axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
        # Adapt y-axis limits to the data
        ymin, ymax = compute_yaxis_limits(dataToPlot, xAxisOnline, xAxisBatch)
    else:
        axes.set_xlim([0, np.amax(xAxisOnline)])
        # Adapt y-axis limits to the data
        ymin, ymax = compute_yaxis_limits(dataToPlot, xAxisOnline)
    axes.set_ylim([ymin, ymax])
    if res[0].vad is not None:
        add_vad_line(
            axes,
            durations=xAxisBatch,
            vad=res[0].vad
        )
    else:
        axes.grid(which='both')
    fig.set_label('test3')

    return fig


def compute_yaxis_limits(
        data: dict[str, np.ndarray],
        xAxisOnline=None,
        xAxisBatch=None
    ):
    """Compute y-axis limits for plotting."""
    ymin, ymax = np.inf, -np.inf
    for filterType in list(data.keys()):
        if ('online' in filterType or 'wola' in filterType) and\
            'batch' not in filterType and\
            xAxisBatch is not None:
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
            ymin = min(ymin, np.amin(np.mean(data[filterType], axis=0)))
            ymax = max(ymax, np.amax(np.mean(data[filterType], axis=0)))
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
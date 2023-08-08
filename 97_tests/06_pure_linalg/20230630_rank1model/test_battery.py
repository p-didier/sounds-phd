# Purpose of script:
# Run a battery of tests for the rank-1 model.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Creation date: 07.08.2023

import os
import sys
import functools
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from script import main as run_test


# Global variables
GLOBAL_SEED = 0  # Global seed for random number generator
TMAX = 30  # [s] Maximum duration of the simulated data
FS = 16000  # [Hz] Sampling frequency
N_MC = 5  # Number of Monte Carlo repetitions
N_MC_2 = 5  # Number of Monte Carlo repetitions for SC3
MAX_NUM_SENSORS_PER_NODE = 5  # Maximum number of sensors per node
TO_COMPUTE = [
    'gevdmwf_batch',  # GEVD-MWF (batch)
    'gevdmwf_online',  # GEVD-MWF (online) 
    'gevddanse_sim_online',  # GEVD-DANSE (online), simultaneous node-updating
]
EXPORT_FIGURES = True  # Whether to export figures to PDF and PNG files
EXPORT_PATH = '97_tests/06_pure_linalg/20230630_rank1model/figs/battery_test/20230808_tests'  # Path to export figures to

@dataclass
class TestParameters:
    """Class for parameters of a single test."""
    seed: int  # Seed for random number generator
    K: int  # Number of nodes
    M: int = None # Number of sensors in total (if None, is randomized (at least equal to `K`))
    Mk: list[int] = None  # Number of sensors per node (if None, is randomized)

    def __post_init__(self):
        """Post-initialization checks and randomization."""
        if self.Mk is not None:
            assert len(self.Mk) == self.K, 'Number of nodes is not equal to length of Mk.'
        else:
            self.Mk = [None] * self.K
            if self.M is None:
                for k in range(self.K):
                    self.Mk[k] = np.random.randint(1, MAX_NUM_SENSORS_PER_NODE + 1)
            else:
                # Distribute `self.M` sensors over `self.K` nodes
                for k in range(self.K):
                    if k == self.K - 1:
                        self.Mk[k] = self.M - sum(self.Mk[:-1])
                    else:
                        self.Mk[k] = np.random.randint(
                            1,
                            self.M - sum(self.Mk[:k]) - (self.K - k - 1) + 1
                        )
        if self.M is None:
            self.M = sum(self.Mk)
        else:
            assert self.M >= self.K, 'Number of sensors is smaller than number of nodes.'

@dataclass
class Test:
    """Class for a single test."""
    name: str
    description: str
    parameters: Union[TestParameters, list[TestParameters]]


@dataclass
class TestBattery:
    """Class for a battery of tests."""
    name: str
    description: str
    tests: list[Test]


@dataclass
class TestOutput:
    """Class for the output of a single test."""
    name: str
    description: str
    parameters: TestParameters
    results: dict[str, dict[str, np.ndarray]]

def main():
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(GLOBAL_SEED)
    # Prepare seeds for Monte Carlo repetitions, without replacement
    batteryOfSeeds = np.random.choice(range(9999), N_MC_2, replace=False)
    
    # Create battery of tests
    battery = TestBattery(
        name='Rank-1 model',
        description='Battery of tests for the rank-1 model.',
        tests=[
            Test(
                name='Test 1',
                description='Sanity check with only single-sensor nodes.',
                parameters=TestParameters(
                    seed=0,
                    K=5,
                    M=5
                )
            ),
            Test(
                name='Test 2',
                description='Sanity check with one single-sensor node and one multi-sensor node.',
                parameters=TestParameters(
                    seed=0,
                    K=2,
                    M=5,
                    Mk=[1, 4]  # 1 single-sensor node, 1 multi-sensor node
                )
            ),
            Test(
                name='Test 3',
                description='Battery of tests with random K and Mk values.',
                parameters=[
                    TestParameters(
                        seed=batteryOfSeeds[n],
                        K=3,
                    ) for n in range(N_MC_2)  # Repeat `N_MC` times
                ]
            )
        ]
    )

    # Signal durations to test (for batch mode -- online mode only considers
    # the largest duration)
    durations = np.logspace(np.log10(1), np.log10(TMAX), 30)

    # Prepare output
    print(f'Running battery of tests "{battery.name}"...')
    print(f'Description: {battery.description}')
    print(f'Number of tests: {len(battery.tests)}')
    allOutputs = []
    
    # Run battery of tests
    for test in battery.tests:
        commonKwargs = {
            'toCompute': TO_COMPUTE,
            'durations': durations,
            'fs': FS,
            'exportFigures': False,
            'nMC': N_MC,
            'verbose': False
        }
        if isinstance(test.parameters, list):
            print(f'- Running battery of tests "{test.name}"...')
            for ii, parameters in enumerate(test.parameters):
                print(f'-- Running test {ii + 1} of {len(test.parameters)} (seed: {parameters.seed}, M={parameters.M})...')
                # Run test
                res = run_test(
                    M=parameters.M,
                    K=parameters.K,
                    seed=parameters.seed,
                    Mk=parameters.Mk,
                    **commonKwargs
                )
                # Save output
                allOutputs.append(
                    TestOutput(
                        name=f'{test.name}_{ii + 1}',
                        description=f'SEED: {parameters.seed} - {test.description}',
                        parameters=parameters,
                        results=res
                    )
                )
        else:
            print(f'- Running test "{test.name}"...')

            # Run test
            res = run_test(
                M=test.parameters.M,
                K=test.parameters.K,
                seed=test.parameters.seed,
                Mk=test.parameters.Mk,
                **commonKwargs
            )
            # Save output
            allOutputs.append(
                TestOutput(
                    name=test.name,
                    description=test.description,
                    parameters=test.parameters,
                    results=res
                )
            )

    # Plot results for Type-1/2 tests
    figs = plot_results(
        [out for out in allOutputs if out.name == 'Test 1' or out.name == 'Test 2'],
        xAxisBatch=durations,
    )
    if EXPORT_FIGURES:
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        for fig in figs:
            fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.pdf', bbox_inches='tight')
            fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.png', bbox_inches='tight', dpi=300)
    # Plot results for Type-3 (full Monte-Carlo) tests
    fig = plot_results_mc_tests(
        [out for out in allOutputs if 'SEED' in out.description],
        xAxisBatch=durations,
        indivRunsFig=False
    )
    if EXPORT_FIGURES:
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.pdf', bbox_inches='tight')
        fig.savefig(f'{EXPORT_PATH}/{fig.get_label()}.png', bbox_inches='tight', dpi=300)

    # Show figures
    plt.show(block=False)

    # Wait and close figures
    print('Press any key to close figures and exit...')
    plt.waitforbuttonpress()
    plt.close('all')

    stop = 1


def generate_plots(baseColor, filterType, data, axes, xAxisBatch):
    """Generate plots."""

    lineStyles = ['-', '--', '-.', ':']

    if 'online' in filterType:
        xAxisOnline = np.linspace(
            start=0,
            stop=np.amax(xAxisBatch),
            num=data.shape[1]
        )
        for idxTau in range(data.shape[-1]):
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
                label=filterType
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


def plot_results(resAll: list[TestOutput], xAxisBatch):
    """Plot results for Type-1 and Type-2 test."""
    
    figs = []
    for res in resAll:
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        for ii, filterType in enumerate(TO_COMPUTE):
            xAxisOnline = generate_plots(
                f'C{ii}',
                filterType,
                res.results[filterType],
                axes,
                xAxisBatch
            )
        axes.legend()
        axes.grid(True, which='both')
        axes.set_xlabel('Duration [s]')
        axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
        # Adapt y-axis limits to the data
        ymin, ymax = compute_yaxis_limits(res.results, xAxisOnline, xAxisBatch)
        axes.set_ylim([ymin, ymax])
        axes.set_title(f'$K={res.parameters.K}$, $M={res.parameters.M}$, $M_k={res.parameters.Mk}$ ({N_MC_2} MC runs)')
        fig.set_label(res.name)
        figs.append(fig)

    return figs


def plot_results_mc_tests(
        res: list[TestOutput],
        xAxisBatch,
        indivRunsFig=False
    ):
    """Plot results for Type-3 (full Monte-Carlo) tests."""

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    if indivRunsFig:
        for testOutput in res:
            if 'SEED' in testOutput.description:
                for ii, filterType in enumerate(TO_COMPUTE):
                    xAxisOnline = generate_plots(
                        f'C{ii}',
                        filterType,
                        testOutput.results[filterType],
                        axes,
                        xAxisBatch
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
                xAxisBatch
            )
        axes.set_title(f'Average over {N_MC} $\\times$ {N_MC_2} MC runs (random $\{{M_k\}}_{{k\\in\\mathcal{{K}}}}$)')

    axes.legend()
    axes.grid(True, which='both')
    axes.set_xlabel('Duration [s]')
    axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
    # Adapt y-axis limits to the data
    ymin, ymax = compute_yaxis_limits(dataToPlot, xAxisOnline, xAxisBatch)
    axes.set_ylim([ymin, ymax])
    fig.set_label('Test 3')

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
# Purpose of script:
# Run a battery of tests for the rank-1 model.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Creation date: 07.08.2023

import sys
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from script import main as run_test


# Global variables
GLOBAL_SEED = 0  # Global seed for random number generator
TMAX = 10  # [s] Maximum duration of the simulated data
FS = 16000  # [Hz] Sampling frequency
N_MC = 5  # Number of Monte Carlo repetitions
N_MC_2 = 5  # Number of Monte Carlo repetitions for SC3
MAX_NUM_SENSORS_PER_NODE = 5  # Maximum number of sensors per node
TO_COMPUTE = [
    'gevdmwf_batch',  # GEVD-MWF (batch)
    'gevdmwf_online',  # GEVD-MWF (online) 
    'gevddanse_sim_online',  # GEVD-DANSE (online), simultaneous node-updating
]

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
            # Test(
            #     name='Test 1',
            #     description='Sanity check with only single-sensor nodes.',
            #     parameters=TestParameters(
            #         seed=0,
            #         K=5,
            #         M=5
            #     )
            # ),
            # Test(
            #     name='Test 2',
            #     description='Sanity check with one single-sensor node and one multi-sensor node.',
            #     parameters=TestParameters(
            #         seed=0,
            #         K=2,
            #         M=5,
            #         Mk=[1, 4]  # 1 single-sensor node, 1 multi-sensor node
            #     )
            # ),
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

    # Plot results
    plot_results(allOutputs, xAxisBatch=durations)

    stop = 1


def plot_results(res: list[TestOutput], xAxisBatch, indivRunsFig=False):

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    if indivRunsFig:
        for testOutput in res:
        
            if 'SEED' in testOutput.description:
                for ii, filterType in enumerate(TO_COMPUTE):
                    
                    baseColor = f'C{ii}'

                    if 'online' in filterType:
                        xAxisOnline = np.linspace(
                            start=0,
                            stop=np.amax(xAxisBatch),
                            num=testOutput.results[filterType].shape[1]
                        )
                        for idxTau in range(testOutput.results[filterType].shape[-1]):
                            axes.fill_between(
                                xAxisOnline,
                                np.amin(testOutput.results[filterType][:, :, idxTau], axis=0),
                                np.amax(testOutput.results[filterType][:, :, idxTau], axis=0),
                                color=baseColor,
                                alpha=0.15
                            )
                            axes.loglog(
                                xAxisOnline,
                                np.mean(
                                    testOutput.results[filterType][:, :, idxTau],
                                    axis=0
                                ),
                                f'{baseColor}-',
                                label=filterType
                            )
                    else:
                        axes.fill_between(
                            xAxisBatch,
                            np.amin(testOutput.results[filterType], axis=0),
                            np.amax(testOutput.results[filterType], axis=0),
                            color=baseColor,
                            alpha=0.15
                        )
                        axes.loglog(
                            xAxisBatch,
                            np.mean(
                                testOutput.results[filterType],
                                axis=0
                            ),
                            f'{baseColor}o-',
                            label=filterType
                        )
    else:
        pass  # TODO: average over all runs
    axes.legend()
    axes.grid(True)
    axes.set_xlabel('Duration [s]')
    axes.set_xlim([np.amin(xAxisBatch), np.amax(xAxisBatch)])
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
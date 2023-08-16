# Purpose of script:
# Run a battery of tests for the rank-1 model.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Creation date: 07.08.2023

import os
import sys
import numpy as np
from typing import Union
from dataclasses import dataclass
from script import main as run_test
from script import ScriptParameters


# Get current file folder
FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Global variables
GLOBAL_SEED = 0  # Global seed for random number generator
TMAX = 15  # [s] Maximum duration of the simulated data
FS = 8000  # [Hz] Sampling frequency
N_MC = 10  # Number of Monte Carlo repetitions
N_MC_2 = 10  # Number of Monte Carlo repetitions for SC3
MAX_NUM_SENSORS_PER_NODE = 5  # Maximum number of sensors per node
TO_COMPUTE = [
    # 'gevdmwf_batch',  # GEVD-MWF (batch)
    # 'gevdmwf_online',  # GEVD-MWF (online) 
    # 'gevddanse_sim_online',  # GEVD-DANSE (online), simultaneous node-updating
    'gevdmwf_wola',  # WOLA-based GEVD-MWF (online)
    'gevddanse_sim_wola',  # WOLA-based GEVD-DANSE (online), simultaneous node-updating
    # 'mwf_wola',  # WOLA-based GEVD-MWF (online)
    # 'danse_sim_wola',  # WOLA-based GEVD-DANSE (online), simultaneous node-updating
]
TAUS = [2., 4., 8.]  # [s] Time constants for exp. avg. in online filters
TAUS = [4.]  # [s] Time constants for exp. avg. in online filters

# Export parameters
BATTERY_NAME = 'baseline_betaExt0p9_GEVD_15s'
EXPORT_FOLDER = 'results\\wola\\fs8kHz_correct2'
# EXPORT_FOLDER = 'results\\online\\fs8kHz'

@dataclass
class TestParameters:
    """Class for parameters of a single test."""
    seed: int = 0  # Seed for random number generator
    K: int = 1  # Number of nodes
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
    name: str = ''
    description: str = ''
    parameters: TestParameters = TestParameters()
    results: dict[str, dict[str, np.ndarray]] = None


def main():
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(GLOBAL_SEED)
    # Prepare seeds for Monte Carlo repetitions, without replacement
    batteryOfSeeds = np.arange(N_MC_2)
    # batteryOfSeeds = np.random.choice(range(9999), N_MC_2, replace=False)
    
    # Create battery of tests
    battery = TestBattery(
        name=BATTERY_NAME,
        description='Battery of tests for the rank-1 model.',
        tests=[
            Test(
                name='test1',
                description='Sanity check with only single-sensor nodes.',
                parameters=TestParameters(
                    seed=0,
                    K=5,
                    M=5
                )
            ),
            Test(
                name='test2',
                description='Sanity check with one single-sensor node and one multi-sensor node.',
                parameters=TestParameters(
                    seed=0,
                    K=2,
                    M=5,
                    Mk=[1, 4]  # 1 single-sensor node, 1 multi-sensor node
                )
            ),
            Test(
                name='test3',
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


    # Prepare output
    print(f'Running battery of tests "{battery.name}"...')
    print(f'Description: {battery.description}')
    print(f'Number of tests: {len(battery.tests)}')
    allOutputs = []
    
    # Run battery of tests
    durations = np.logspace(np.log10(1), np.log10(TMAX), 30)  # Signal
    # durations to test (for batch mode -- online mode only considers
    # the largest duration)
    commonKwargs = {
        'toCompute': TO_COMPUTE,
        'durations': durations,
        'fs': FS,
        'exportFigures': False,
        'nMC': N_MC,
        'verbose': False,
        'taus': TAUS,
    }
    for test in battery.tests:
        if isinstance(test.parameters, list):
            print(f'- Running battery of tests "{test.name}"...')
            for ii, parameters in enumerate(test.parameters):
                print(f'-- Running "{test.name}" test {ii + 1} of {len(test.parameters)} (seed: {parameters.seed}, M={parameters.M})...')
                # Run test
                res = run_test(
                ScriptParameters(
                    nSensors=parameters.M,
                    nNodes=parameters.K,
                    seed=parameters.seed,
                    Mk=parameters.Mk,
                    **commonKwargs
                ))
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
            res = run_test(ScriptParameters(
                nSensors=test.parameters.M,
                nNodes=test.parameters.K,
                seed=test.parameters.seed,
                Mk=test.parameters.Mk,
                **commonKwargs
            ))
            # Save output
            allOutputs.append(
                TestOutput(
                    name=test.name,
                    description=test.description,
                    parameters=test.parameters,
                    results=res
                )
            )

    # Export results
    fullExportFolder = f'{FILE_FOLDER}/{EXPORT_FOLDER}'
    if not os.path.exists(fullExportFolder):
        os.makedirs(fullExportFolder)
    np.savez_compressed(
        f'{fullExportFolder}/{battery.name}.npz',
        allOutputs=allOutputs
    )

    print('ALL DONE.')


if __name__ == '__main__':
    sys.exit(main())
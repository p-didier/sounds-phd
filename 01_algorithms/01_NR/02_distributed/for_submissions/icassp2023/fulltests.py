import sys
from pathlib import Path, PurePath
from dataclasses import dataclass, field
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_testing import *

@dataclass
class TestsParams:
    pathToASC: str = '' # path to acoustic scenario to consider
    computeCentralised: bool = False    # if True, compute centralised estimate too for baseline reference
    deltaSROs: list[int] = field(default_factory=list)  # list of SRO magnitudes to consider.
                                                        # --> 2 simulations per element: w/ and w/out comp.
    exportBasePath: str = ''    # path to exported files

# -------------------------------------------
# ------------- TEST PARAMETERS -------------
# -------------------------------------------
p = TestsParams(
    pathToASC='02_data/01_acoustic_scenarios/for_submissions/icassp2023/J4Mk[1_3_2_5]_Ns1_Nn2/AS18_RT150ms',
    computeCentralised=False,
    deltaSROs=[0],
    exportBasePath='01_algorithms/01_NR/02_distributed/res/for_submissions/icassp2023',
)
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------


def main():

    # Derive the number of simulations to conduct
    nTest = 2 * len(p.deltaSROs)
    if p.computeCentralised:
        nTest += 1

    # Derive the number of nodes
    nNodes = int(Path(p.pathToASC).parent.stem[1])

    for ii in range(nTest):
        if p.computeCentralised and ii == nTest - 1:
            params = dict([
                ('listedSROs', [0] * nNodes), ('comp', False), ('bc', 'wholeChunk'),\
                ('computeCentrEstimate', True), ('ASC', p.pathToASC)   # centralised case
            ])
        else:
            lsros = [((-1)**jj) * ((jj + 1) // 2) * p.deltaSROs[ii // 2] for jj in range(nNodes)]
            if ii % 2 == 0:
                params = dict([
                    ('listedSROs', lsros), ('comp', True), ('bc', 'samplePerSample'),\
                    ('computeCentrEstimate', False), ('ASC', p.pathToASC)   # do compensate SROs
                ])
            else:
                params = dict([
                    ('listedSROs', lsros), ('comp', False), ('bc', 'wholeChunk'),\
                    ('computeCentrEstimate', False), ('ASC', p.pathToASC)  # don't compensate SROs
                ])

        run_simul(params, p.exportBasePath)


def run_simul(params, exportBasePath):

    # Generate test parameters
    danseTestingParams = sro_testing.DanseTestingParameters(
        # General hyperparameters
        writeOver=False,    # if True, the script will overwrite existing data when filenames conflict
        #
        ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
        signalsPath=f'{Path(__file__).parent}/validations/signals',
        #
        # vvvvvvvvvvvvvv
        specificAcousticScenario=[f'{pathToRoot}/{params["ASC"]}'],
        #
        fs=16000,
        specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
        specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
        sigDur=15,
        baseSNR=5,
        #
        SROsParams=TestSROs(
            type='specific',
            # vvvvvvvvvvvvvv
            listedSROs=params['listedSROs'],
        ),
        #
        timeBtwExternalFiltUpdates=3.,
        #
        # vvvvvvvvvvvvvv
        broadcastScheme=params['bc'],
        performGEVD=1,
        #
        computeLocalEstimate=True,
        # vvvvvvvvvvvvvv
        computeCentrEstimate=params['computeCentrEstimate'],
        #
        asynchronicity=SamplingRateOffsets(
            plotResult=True,
            # vvvvvvvvvvvvvv
            compensateSROs=params['comp'],
            estimateSROs='CohDrift',
            cohDriftMethod=CohDriftSROEstimationParameters(
                loop='open'
            )
        ),
        printouts=PrintoutsParameters(
            progressPrintingInterval=0.5
        )
    )

    # Run test
    sro_testing.go(danseTestingParams, exportBasePath)


if __name__=='__main__':
    sys(exit(main()))
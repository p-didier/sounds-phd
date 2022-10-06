"""Main script for testing implementations of DANSE algorithm."""

from random import gauss
import sys

from danse_utilities.testing import sro_testing
from danse_utilities.classes import SamplingRateOffsets, CohDriftSROEstimationParameters, PrintoutsParameters
from pathlib import Path, PurePath
import numpy as np

from testing.sro_testing import TestSROs
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')

# ------------------------------- PARAMETERS ----------------------------------
# Path where to export all test results (each experiment will be a subfolder of `exportBasePath`)
exportBasePath = f'{Path(__file__).parent}/res/for_submissions/icassp2023'

# Build testing parameters object
danseTestingParams = sro_testing.DanseTestingParameters(
    # General hyperparameters
    writeOver=True,    # if True, the script will overwrite existing data when filenames conflict
    #
    ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
    signalsPath=f'{Path(__file__).parent}/validations/signals',
    #
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/for_submissions/icassp2023/J4Mk[6_3_8_3]_Ns1_Nn2/AS1_RT150ms'],  # overrides use of `danseTestingParams.ascBasePath`
    specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/for_submissions/icassp2023/J4Mk[1_3_2_5]_Ns1_Nn2/AS2_RT150ms'],  # overrides use of `danseTestingParams.ascBasePath`
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/for_submissions/icassp2023/J2Mk[1_1]_Ns1_Nn1/AS2_anechoic'],  # overrides use of `danseTestingParams.ascBasePath`
    #
    # fs=8000,
    fs=16000,
    specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    sigDur=15,
    baseSNR=5,
    #
    # possibleSROs=[int(ii) for ii in np.linspace(10, 100, num=10)],
    # possibleSROs=[int(ii) for ii in np.linspace(0, 100, num=11)],
    # possibleSROs=[100],
    SROsParams=TestSROs(
        # type='g',           # 'list' pick from list | 'g' pick from normal dist.
        # type='list',      # 'list' pick from list | 'g' pick from normal dist.
        type='specific',      # 'list' pick from list | 'g' pick from normal dist.
        # listedSROs=[0, 20, -20, 50],    # small SROs, single run
        listedSROs=[0, 0, 0, 0],    # small SROs, single run
        # listedSROs=[0, 50, -50, 100],   # medium SROs, single run
        # listedSROs=[0, 200, -200, 400], # large SROs, single run
        gaussianParams=[7.5, 2],    # <-- [7.5, 2]: "small SROs" case
        # gaussianParams=[75, 20],  # <-- [75, 20]: "medium SROs" case
        # gaussianParams=[275, 50], # <-- [275, 50]: "large SROs" case
        numGaussianDraws=10  # number of SRO scenarios to consider
    ),
    #
    timeBtwExternalFiltUpdates=3.,
    #
    # broadcastScheme='samplePerSample',
    broadcastScheme='wholeChunk',
    performGEVD=1,
    #
    computeLocalEstimate=True,
    computeCentrEstimate=True,
    #
    asynchronicity=SamplingRateOffsets(
        plotResult=True,
        # compensateSROs=True,
        compensateSROs=False,
        estimateSROs='CohDrift',
        # estimateSROs='Oracle',
        cohDriftMethod=CohDriftSROEstimationParameters(
            loop='open'
        )
    ),
    printouts=PrintoutsParameters(
        progressPrintingInterval=0.5
    )
)


def main():
    """Main wrapper for DANSE testing."""
    sro_testing.go(danseTestingParams, exportBasePath)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
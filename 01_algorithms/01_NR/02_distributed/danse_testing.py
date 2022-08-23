"""Main script for testing implementations of DANSE algorithm."""

import sys

from danse_utilities.testing import sro_testing
from danse_utilities.classes import SamplingRateOffsets, CohDriftSROEstimationParameters, PrintoutsParameters
from pathlib import Path, PurePath
import numpy as np
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')

# ------------------------------- PARAMETERS ----------------------------------
# Path where to export all test results (each experiment will be a subfolder of `exportBasePath`)
exportBasePath = f'{Path(__file__).parent}/res/testing_SROs/automated/20220822_SROestComp_wFlags'

# Build testing parameters object
danseTestingParams = sro_testing.DanseTestingParameters(
    # General hyperparameters
    writeOver=True,    # if True, the script will write over existing data if filenames conflict
    #
    ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
    signalsPath=f'{Path(__file__).parent}/validations/signals',
    #
    specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS2_anechoic'],  # overrides use of `danseTestingParams.ascBasePath`
    #
    fs=8000,
    specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    sigDur=15,
    baseSNR=5,
    #
    possibleSROs=[int(ii) for ii in np.linspace(10, 100, num=10)],
    # possibleSROs=[int(ii) for ii in np.linspace(0, 100, num=11)],
    # possibleSROs=[100],
    #
    timeBtwExternalFiltUpdates=3.,
    #
    # broadcastScheme='samplePerSample',
    broadcastScheme='wholeChunk',
    performGEVD=1,
    #
    computeLocalEstimate=True,
    #
    asynchronicity=SamplingRateOffsets(
        plotResult=True,
        compensateSROs=True,
        # compensateSROs=False,
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
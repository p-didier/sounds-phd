"""Main script for testing implementations of DANSE algorithm."""

import sys
from danse_utilities.testing import sro_testing
from danse_utilities.classes import SamplingRateOffsets, DWACDParameters
from pathlib import Path, PurePath
import numpy as np
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')

# ------------------------------- PARAMETERS ----------------------------------
exportBasePath = f'{Path(__file__).parent}/res/testing_SROs/automated'

# Build testing parameters object
danseTestingParams = sro_testing.DanseTestingParameters(
    ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
    signalsPath=f'{Path(__file__).parent}/validations/signals',
    #
    specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS8_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS6_RT400ms'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS4_RT200ms'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J3Mk[2_3_4]_Ns1_Nn1/AS5_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3_1]_Ns1_Nn1/AS1_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3_1]_Ns1_Nn1/AS2_RT500ms'],
    #
    fs=8000,    
    # fs=16000,    
    specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    sigDur=40,
    baseSNR=5,
    # possibleSROs=[int(ii) for ii in np.linspace(0, 100, num=10)],
    possibleSROs=[int(ii) for ii in np.linspace(0, 1000, num=10)],
    nodeUpdating='simultaneous',
    timeBtwExternalFiltUpdates=3,
    # timeBtwExternalFiltUpdates=np.Inf,
    # broadcastLength=8,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
    broadcastLength=2**9,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
    broadcastDomain='f',
    #
    asynchronicity=SamplingRateOffsets(
        compensateSROs=True,
        # compensateSROs=False,
        # estimateSROs='DWACD',
        estimateSROs='Residuals',
        dwacd=DWACDParameters(
            seg_shift=2**11,
        )
    ),
)


def main():
    """Main wrapper for DANSE testing."""
    sro_testing.go(danseTestingParams, exportBasePath)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
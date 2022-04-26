"""Main script for testing implementations of DANSE algorithm."""

import sys
from danse_utilities.testing import sro_testing
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
    specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1, 1]_Ns1_Nn1/AS2_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J3Mk[2, 3, 4]_Ns1_Nn1/AS5_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3, 1]_Ns1_Nn1/AS1_anechoic'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3, 1]_Ns1_Nn1/AS2_RT500ms'],
    #
    # fs=8000,    
    fs=16000,    
    specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    sigDur=20,
    baseSNR=5,
    # possibleSROs=[0, 100, 200, 400, 800, 1600, 3200],
    # possibleSROs=[0, 20, 40, 60, 80, 100],
    possibleSROs=[int(ii) for ii in np.linspace(0, 100, num=20)],
    # possibleSROs=[0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    # possibleSROs=[int(ii) for ii in np.logspace(0, np.log10(32000), num=20)],
    # possibleSROs=[32000],
    nodeUpdating='simultaneous',
    # timeBtwExternalFiltUpdates=1,
    timeBtwExternalFiltUpdates=np.Inf,
    # broadcastLength=8,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
    broadcastLength=512,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
    #
    compensateSROs=True,
    estimateSROs=True,
)


def main():
    """Main wrapper"""
    sro_testing.go(danseTestingParams, exportBasePath)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
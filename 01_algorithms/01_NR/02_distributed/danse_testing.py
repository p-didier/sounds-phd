"""Main script for testing implementations of DANSE algorithm."""

import sys
from danse_utilities.testing import sro_testing
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')

# ------------------------------- PARAMETERS ----------------------------------
testType = 'basic_performance_srofree'
testType = 'sro_effect'
# testType = 'sro_compensation'

exportBasePath = f'{Path(__file__).parent}/res/testing_SROs/automated'

# Build testing parameters object
danseTestingParams = sro_testing.DanseTestingParameters(
    ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
    signalsPath=f'{Path(__file__).parent}/validations/signals',
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3, 3]_Ns1_Nn1_anechoic/AS1'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[2, 2]_Ns1_Nn1_anechoic/AS1'],
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[2, 1]_Ns1_Nn1_anechoic/AS1'],
    specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[3, 1]_Ns1_Nn1_anechoic/AS1'],
    # specificAcousticScenario=f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1_anechoic/AS3',
    # specificAcousticScenario=[f'{pathToRoot}/02_data/01_acoustic_scenarios/tests/J2Mk[1, 1]_Ns1_Nn1_anechoic/AS1'],
    specificDesiredSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    specificNoiseSignalFiles=[f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    sigDur=20,
    baseSNR=5,
    # possibleSROs=[0, 100, 200, 400, 800, 1600, 3200],
    # possibleSROs=[0, 20, 40, 60, 80, 100],
    # possibleSROs=[0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    possibleSROs=[0, 10000, 20000, 40000, 80000, 160000, 320000],
    nodeUpdating='simultaneous',
    broadcastLength=8,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
)
# -----------------------------------------------------------------------------

def main(type, danseTestingParams, exportBasePath):
    """Test type selection -- redirection towards testing functions"""

    if type == 'basic_performance_srofree':     # basic (GEVD-)DANSE performance, synchronized network, no SROs
        danseTestingParams.possibleSROs = [0]
    elif type == 'sro_effect':
        sro_testing.go(danseTestingParams, exportBasePath)
    elif type == 'sro_compensation':
        pass

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(testType, danseTestingParams, exportBasePath))
# ------------------------------------------------------------------------------------
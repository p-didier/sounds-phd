from pathlib import Path, PurePath
import sys

# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}\\01_algorithms\\01_NR\\02_distributed')
from danse_utilities import events_manager as evm
from danse_utilities.setup import generate_signals
from danse_utilities.classes import ProgramSettings


# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'

mySettings = ProgramSettings(
    samplingFrequency=8000,
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1, 1]_Ns1_Nn1/AS2_anechoic',
    #
    desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    #
    signalDuration=20,
    baseSNR=5,
    DFTsize=2**10,            # DANSE iteration processing chunk size [samples]
    Ns=0.5,           # overlap between DANSE iteration processing chunks [/100%]
    broadcastLength=2**9,       # broadcast chunk size `L` [samples]
    #
    SROsppm=[50000, 0]
    )


def main():

    mySignals, asc = generate_signals(mySettings)

    events, _ = evm.get_events_matrix(mySignals.timeStampsSROs,
                            mySettings.stftWinLength,
                            mySettings.stftEffectiveFrameLen,
                            mySettings.broadcastLength,
                            asc.nodeLinks,
                            mySignals.fs)
                            
    evm.visualize_events(events,
                        tmax=3,    # [s]
                        suptitle=f'SROs: {mySettings.SROsppm} (ppm)')

    stop = 1



# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
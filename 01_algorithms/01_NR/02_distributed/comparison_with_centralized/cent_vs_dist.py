import sys
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np
#
from utils_comp import batch
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
if not any("_third_parties" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_third_parties')
if not any("01_algorithms/01_NR/02_distributed" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
# Custom packages imports
from danse_utilities.classes import ProgramSettings, get_stft
from danse_utilities.setup import generate_signals, danse


# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'
# Set experiment settings
mySettings = ProgramSettings(
    samplingFrequency=16000,
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1, 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J1Mk[4]_Ns1_Nn1/AS1_anechoic',
    desiredSignalFile=[f'{signalsPath}/01_speech/speech1.wav'],
    noiseSignalFile=[f'{signalsPath}/02_noise/whitenoise_signal_1.wav'],
    #
    signalDuration=10,
    baseSNR=0,
    chunkSize=2**10,            # DANSE iteration processing chunk size [samples]
    chunkOverlap=0.5,           # Overlap between DANSE iteration processing chunks [/100%]
    SROsppm=0,
    #
    broadcastLength=2**9,
    expAvg50PercentTime=2.,             # Time in the past [s] at which the value is weighted by 50% via exponential averaging
    danseUpdating='simultaneous',       # node-updating scheme
    referenceSensor=0,
    computeLocalEstimate=True,
    performGEVD=1,
    minTimeBtwFiltUpdates=1,            # [s] minimum time between 2 consecutive filter update at a node 
    )

# SIMULATIONMODE_MWF = 'online'
SIMULATIONMODE_MWF = 'batch'
    


def main(settings):

    print(mySettings)

    print(f'Simulating in {SIMULATIONMODE_MWF} mode.')

    print('\nGenerating simulation signals...')
    # Generate base signals (and extract acoustic scenario)
    mySignals, asc = generate_signals(settings)

    if 0:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(mySignals.sensorSignals)
        ax.grid()
        plt.tight_layout()	
        plt.show()


    # Centralized solution - GEVD-MWF
    if SIMULATIONMODE_MWF == 'batch':
        print('Computing centralized solution...')
        DfiltGEVD, _ = batch.gevd_mwf_batch(mySignals, asc, settings)

    elif SIMULATIONMODE_MWF == 'online':
        raise ValueError('NOT YET IMPLEMENTED')

    # Distributed solution - DANSE
    print('Computing distributed solution (DANSE)...')
    mySignals.desiredSigEst, mySignals.desiredSigEstLocal = danse(mySignals, asc, settings)

    DfiltDANSE, f, t = get_stft(mySignals.desiredSigEst, mySignals.fs, settings)

    # Get colorbar limits
    climHigh = np.amax(np.concatenate((20*np.log10(np.abs(DfiltGEVD)), 20*np.log10(np.abs(DfiltDANSE[:,:,0]))), axis=0))
    climLow = np.amin(np.concatenate((20*np.log10(np.abs(DfiltGEVD)), 20*np.log10(np.abs(DfiltDANSE[:,:,0]))), axis=0))

    # Comparison
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(311)
    mapp = plt.imshow(20*np.log10(np.abs(DfiltGEVD)), extent=[t[0], t[-1], f[-1,0], f[0,0]])
    ax.invert_yaxis()
    ax.set_aspect('auto')
    plt.colorbar(mapp)
    plt.title(f'After batch-GEVD-MWF (rank {settings.GEVDrank})')
    ax = fig.add_subplot(312)
    mapp = plt.imshow(20*np.log10(np.abs(DfiltDANSE[:,:,0])), extent=[t[0], t[-1], f[-1,0], f[0,0]])
    ax.invert_yaxis()
    ax.set_aspect('auto')
    plt.colorbar(mapp)
    plt.title(f'After GEVD-DANSE (rank {settings.GEVDrank})')
    ax = fig.add_subplot(313)
    mapp = plt.imshow(np.abs(20*np.log10(np.abs(DfiltDANSE[:,:,0])) - 20*np.log10(np.abs(DfiltGEVD))), extent=[t[0], t[-1], f[-1,0], f[0,0]])
    ax.invert_yaxis()
    ax.set_aspect('auto')
    plt.colorbar(mapp)
    plt.title(f'Absolute magnitude difference [dB]')
    plt.tight_layout()	
    plt.show()

    stop = 1
        
    
# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(mySettings))
# ------------------------------------------------------------------------------------
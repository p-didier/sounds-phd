import sys
from pathlib import Path, PurePath
from threading import local
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
from danse_utilities.classes import ProgramSettings, get_stft, PrintoutsParameters
from danse_utilities.setup import generate_signals, danse


# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'
# Set experiment settings
mySettings = ProgramSettings(
    samplingFrequency=16000,
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[2, 3, 4]_Ns1_Nn1/AS4_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3, 1]_Ns1_Nn1/AS1_anechoic',
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1, 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[2, 3, 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[1, 2, 3]_Ns1_Nn1/AS3_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[2, 3, 4]_Ns1_Nn1/AS3_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[4, 5, 6, 5, 4]_Ns1_Nn1/AS1_anechoic',
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
    selfnoiseSNR=-50,
    #
    broadcastLength=2**9,
    # broadcastLength=2**5,
    expAvg50PercentTime=2.,             # [s] time in the past at which the value is weighted by 50% via exponential averaging
    # expAvg50PercentTime=.1,             # [s] time in the past at which the value is weighted by 50% via exponential averaging
    danseUpdating='simultaneous',       # node-updating scheme
    referenceSensor=0,
    computeLocalEstimate=True,
    performGEVD=1,
    # timeBtwExternalFiltUpdates=3,       # [s] minimum time between 2 consecutive filter update at a node 
    # timeBtwExternalFiltUpdates=0,       # [s] minimum time between 2 consecutive filter update at a node 
    timeBtwExternalFiltUpdates=np.Inf,       # [s] minimum time between 2 consecutive filter update at a node
    #
    broadcastDomain='t',
    # broadcastDomain='f',
    #
    # vvv Printouts parameters vvv
    printouts=PrintoutsParameters(events_parser=True,
                                externalFilterUpdates=False),
    )

# SIMULATIONMODE_MWF = 'online'
SIMULATIONMODE_MWF = 'batch'


def main(settings):

    print(mySettings)

    print('\nGenerating simulation signals...')
    # Generate base signals (and extract acoustic scenario)
    mySignals, asc = generate_signals(settings)

    # Centralized solution - GEVD-MWF
    print(f'Simulating centralized solution in {SIMULATIONMODE_MWF} mode.')
    if SIMULATIONMODE_MWF == 'batch':
        print('Computing centralized solution...')
        DfiltGEVD, _ = batch.gevd_mwf_batch(mySignals, asc, settings)

    elif SIMULATIONMODE_MWF == 'online':
        raise ValueError('NOT YET IMPLEMENTED')


    # Distributed solution - DANSE
    print('Computing distributed solution (DANSE)...')
    mySignals.desiredSigEst, mySignals.desiredSigEstLocal = danse(mySignals, asc, settings)

    # PLOT
    DfiltDANSE, f, t = get_stft(mySignals.desiredSigEst, mySignals.fs, settings)
    DfiltDANSElocal, f, t = get_stft(mySignals.desiredSigEstLocal, mySignals.fs, settings)
    if f.ndim == 1:
        f = f[:, np.newaxis]


    for ii in range(asc.numNodes):
        localData = mySignals.sensorSignals[:, asc.sensorToNodeTags == ii+1]
        if localData.ndim == 2:
            localData = localData[:, 0]
        Ystft, _, _ = get_stft(localData, mySignals.fs, settings)
        # Get colorbar limits
        climHigh = np.amax(np.concatenate((20*np.log10(np.abs(DfiltGEVD)), 20*np.log10(np.abs(DfiltDANSE[:,:,0])), 20*np.log10(np.abs(Ystft[:,:,0]))), axis=0))
        climLow = np.amin(np.concatenate((20*np.log10(np.abs(DfiltGEVD)), 20*np.log10(np.abs(DfiltDANSE[:,:,0])), 20*np.log10(np.abs(Ystft[:,:,0]))), axis=0))
        if climLow < -200:
            climLow = -200

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(221)
        mapp = plt.imshow(20*np.log10(np.abs(Ystft)), extent=[t[0], t[-1], f[-1,0], f[0,0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'Original microphone signal')
        #
        ax = fig.add_subplot(222)
        mapp = plt.imshow(20*np.log10(np.abs(DfiltGEVD)), extent=[t[0], t[-1], f[-1,0], f[0,0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'After batch-GEVD-MWF (rank {settings.GEVDrank})')
        #
        ax = fig.add_subplot(223)
        mapp = plt.imshow(20*np.log10(np.abs(DfiltDANSElocal[:,:,ii])), extent=[t[0], t[-1], f[-1,0], f[0,0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'After GEVD-DANSE [local estimate]')
        #
        ax = fig.add_subplot(224)
        mapp = plt.imshow(20*np.log10(np.abs(DfiltDANSE[:,:,ii])), extent=[t[0], t[-1], f[-1,0], f[0,0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'After GEVD-DANSE (rank {settings.GEVDrank})')
        #
        plt.suptitle(f'Node {ii+1} ({asc.numSensorPerNode[ii]} sensors)')
        plt.tight_layout()	
    plt.show()

    stop = 1
        
    
# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(mySettings))
# ------------------------------------------------------------------------------------
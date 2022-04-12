# Using the "danse_env" virtual environment
# %%
# ---------------- Imports
import time
t00 = time.perf_counter()
from pathlib import Path, PurePath
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background
print(f'Global packages loaded ({round(time.perf_counter() - t00, 2)}s)')
t0 = time.perf_counter()
from danse_utilities.classes import ProgramSettings, Results, PrintoutsParameters
from danse_utilities.setup import run_experiment
print(f'DANSE packages loaded ({round(time.perf_counter() - t0, 2)}s)')
t0 = time.perf_counter()
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
if not any("_third_parties" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_third_parties')
from playsounds.playsounds import playwavfile
from general.osutils import wipe_folder
print(f'Custom (non-DANSE) packages loaded ({round(time.perf_counter() - t0, 2)}s)')
print(f'Total packages loading time: {round(time.perf_counter() - t00, 2)}s\n')
# ------------------------

# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'
# Set experiment settings
mySettings = ProgramSettings(
    samplingFrequency=8000,
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3, 3]_Nss1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J1Mk[4]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[2, 3, 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1/AS6_allNodesInSamePosition_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3, 1]_Ns1_Nn1/AS1_anechoic',
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1, 1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1/AS10_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3, 1]_Ns1_Nn1/AS2_allNodesInSamePosition_anechoic',
    #
    # desiredSignalFile=[f'{signalsPath}/03_test_signals/tone100Hz.wav'],
    desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    #
    signalDuration=30,
    baseSNR=5,
    chunkSize=2**10,            # DANSE iteration processing chunk size [samples]
    chunkOverlap=0.5,           # overlap between DANSE iteration processing chunks [/100%]
    # broadcastLength=2**9,       # broadcast chunk size `L` [samples]
    broadcastLength=16,       # broadcast chunk size `L` [samples]
    #
    # vvv SROs parameters vvv
    # SROsppm=[0, 10000, 20000],               # SRO
    # SROsppm=[0, 4000, 6000],
    # SROsppm=[0, 400, 600],
    # SROsppm=[0, 2000, 12000, 22000, 32000],
    SROsppm=[0, 100],
    # SROsppm=0,
    compensateSROs=True,                # if True, compensate SROs
    # compensateSROs=False,                # if True, compensate SROs
    estimateSROs=False,                 # if True, estimate SROs; elif `compensateSROs == True`: use oracle knowledge of SROs for compensation
    #
    # vvv STOs parameters vvv
    # STOinducedDelays=[0, 0.1],         # [s]
    # compensateSTOs=False,                # if True, compensate STOs
    compensateSTOs=True,                # if True, compensate STOs
    #
    expAvg50PercentTime=2.,             # [s] time in the past at which the value is weighted by 50% via exponential averaging
    danseUpdating='simultaneous',       # node-updating scheme
    referenceSensor=0,
    computeLocalEstimate=True,
    performGEVD=1,
    # bypassFilterUpdates=True,
    timeBtwExternalFiltUpdates=1,       # [s] time between 2 consecutive external filter update (for broadcasting) at a node
    # 
    # vvv Printouts parameters vvv
    printouts=PrintoutsParameters(events_parser=True)
    )

# Subfolder for export
subfolder = f'testing_SROs/single_tests/{mySettings.danseUpdating}/{Path(mySettings.acousticScenarioPath).parent.name}'
# experimentName = f'SROcompTesting/SROs{mySettings.SROsppm}' # experiment reference label
# experimentName = f'testing_SROs/single_tests/{mySettings.danseUpdating}_{[int(sro) for sro in mySettings.SROsppm]}ppm' # experiment reference label
experimentName = f'{[int(sro) for sro in mySettings.SROsppm]}ppm_{int(mySettings.signalDuration)}s' # experiment reference label
if mySettings.compensateSROs:
    experimentName += '_comp'
else:
    experimentName += '_nocomp'

exportPath = f'{Path(__file__).parent}/res/{subfolder}/{experimentName}'
lightExport = True          # <-- set to True to not export whole signals in pkl.gz archives
# ------------------------

def main(mySettings: ProgramSettings, exportPath, showPlots=1, lightExport=True):
    """Main function for DANSE runs.

    Parameters
    ----------
    mySettings : ProgramSettings object
        Experiment settings.    
    exportPath : str
        Path to export directory, containing label of experiment.
    showPlots : bool
        If True, shows plots in Python interpreter window.
    lightExport : bool
        If True, export a lighter version (not all results, just the minimum).
    """
    # Check if experiment has already been run
    runExpFlag = False
    if Path(exportPath).is_dir():
        val = input(f'Already existing experiment "{PurePath(exportPath).name}" in folder "{PurePath(Path(exportPath).parent).name}".\nRe-run experiment anyway? [Y/N]  ')
        if val == 'y' or val == 'Y':
            runExpFlag = True
            print(f'\nRe-running experiment "{PurePath(exportPath).name}" ...\n')
            print(mySettings)
        else:
            print('\nNot re-running experiment.\n')
    else:
        runExpFlag = True
        print(mySettings)
    
    if runExpFlag:
        # ================ Run experiment ================
        results = run_experiment(mySettings)
        # ================================================
        if not mySettings.bypassFilterUpdates:
            # Export
            if not Path(exportPath).is_dir():
                Path(exportPath).mkdir(parents=True)
                print(f'Created output directory "{exportPath}".')
            else:
                wipe_folder(exportPath)     # wipe export folder    
            results.save(exportPath, lightExport)        # save results
            mySettings.save(exportPath)     # save settings
            # Post-process
            get_figures_and_sound(results, exportPath, mySettings, showPlots, listen=False)
        else:
            print('FILTER UPDATES WERE BYPASSED. NO EXPORT, NO RESULTS VISUALIZATION.')

    return None


def get_figures_and_sound(results: Results, pathToResults, settings: ProgramSettings, showPlots=False, listen=False, listeningMaxDuration=5.):
    """From exported pkl.gz file names, import simulation results,
    plots them nicely, exports plots, exports relevent sounds,
    and, if asked, plays back the sounds. 
    
    Parameters
    ----------
    pathToResults : str
        Path to pkl.gz results files.
    settings : ProgramSettings object
        Experiment settings. 
    showPlots : bool
        If True, shows plots in Python interpreter window.
    listen : bool
        If True, plays back some signals in interpreter system.
    listeningMaxDuration : float
        Maximal playback duration (only used if <listen> is True).
    """

    # Export as WAV
    wavFilenames = results.signals.export_wav(pathToResults)

    # Plot scenario
    fig = results.acousticScenario.plot()
    myPath = Path(settings.acousticScenarioPath)
    fig.suptitle(f'{myPath.parent.name}_{myPath.name}')
    fig.tight_layout()
    plt.savefig(f'{pathToResults}/acousScenario.png')
    if showPlots:
        plt.draw()

    # Plot performance
    fig = results.plot_enhancement_metrics()
    expAvgTau = np.log(0.5) * settings.stftEffectiveFrameLen / (np.log(settings.expAvgBeta) * results.signals.fs[0])
    fig.suptitle(f'Speech enhancement metrics ($\\beta={np.round(settings.expAvgBeta, 4)} \Leftrightarrow \\tau_{{50\%, 0\\mathrm{{ppm}}}} = {np.round(expAvgTau, 2)}$ s)')
    fig.tight_layout()
    plt.savefig(f'{pathToResults}/enhMetrics.png')
    if showPlots:
        plt.draw()

    # Plot best performance node (in terms of STOI)
    maxSTOI = -9999999
    minSTOI = 9999999
    for idxNode in range(results.acousticScenario.numNodes):
        currSTOI = results.enhancementEval.stoi[f'Node{idxNode + 1}'].after
        if currSTOI >= maxSTOI:
            bestNode, maxSTOI = idxNode, currSTOI
        if currSTOI <= minSTOI:
            worstNode, minSTOI = idxNode, currSTOI


    print(f'Best node (STOI = {round(maxSTOI, 2)})')
    stoiImpLocalVsGlobal = None
    if settings.computeLocalEstimate:
        stoiBest = results.enhancementEval.stoi[f'Node{bestNode + 1}']
        stoiImpLocalVsGlobal = stoiBest.after - stoiBest.afterLocal
    results.signals.plot_signals(bestNode, settings, stoiImpLocalVsGlobal)
    plt.savefig(f'{pathToResults}/bestPerfNode.png')
    if showPlots:
        plt.draw()
    
    print(f'Worst node (STOI = {round(minSTOI, 2)})')
    stoiImpLocalVsGlobal = None
    if settings.computeLocalEstimate:
        stoiBest = results.enhancementEval.stoi[f'Node{worstNode + 1}']
        stoiImpLocalVsGlobal = stoiBest.after - stoiBest.afterLocal
    results.signals.plot_signals(worstNode, settings, stoiImpLocalVsGlobal)
    plt.savefig(f'{pathToResults}/worstPerfNode.png')
    if showPlots:
        plt.draw()

    results.signals.plot_enhanced_stft(bestNode, worstNode, results.enhancementEval)
    plt.savefig(f'{pathToResults}/bestAndWorstSTFTs.png')
    if showPlots:
        plt.draw()

    if showPlots:
        plt.show()

    if listen:
        # LISTEN
        playwavfile(wavFilenames['Noisy'][bestNode], listeningMaxDuration)
        playwavfile(wavFilenames['Desired'][bestNode], listeningMaxDuration)
        playwavfile(wavFilenames['Enhanced'][bestNode], listeningMaxDuration)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(mySettings, exportPath, lightExport))
# ------------------------------------------------------------------------------------
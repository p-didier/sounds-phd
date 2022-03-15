# Using the "danse_env" virtual environment
# %%
# ---------------- Imports
import time
t00 = time.perf_counter()
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background
print(f'Global packages loaded ({round(time.perf_counter() - t00, 2)}s)')
t0 = time.perf_counter()
from danse_utilities.classes import ProgramSettings
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
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3, 3]_Ns1_Nn1_anechoic/AS1',
    # acousticScenarioPath=f'{ascBasePath}/tests/J1Mk[4]_Ns1_Nn1_anechoic/AS1',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1_anechoic/AS1',
    desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    signalDuration=10,
    baseSNR=0,
    chunkSize=2**10,            # DANSE iteration processing chunk size [samples]
    chunkOverlap=0.5,           # Overlap between DANSE iteration processing chunks [/100%]
    # SROsppm=[0, 10000, 20000],               # SRO
    # SROsppm=[0, 4000, 6000],               # SRO
    # SROsppm=[0, 400, 600],               # SRO
    # SROsppm=[0, 32000],                  # SRO
    SROsppm=0,                  # SRO
    compensateSROs=True,                # if True, estimate + compensate SRO dynamically
    broadcastLength=8,                  # number of (compressed) samples to be broadcasted at a time to other nodes -- only used if `danseUpdating == "simultaneous"`
    danseUpdating='simultaneous',       # node-updating scheme
    referenceSensor=0,
    computeLocalEstimate=True,
    performGEVD=False
    )
# experimentName = f'SROcompTesting/SROs{mySettings.SROsppm}' # experiment reference label
# experimentName = f'testing_SROs/single_tests/{mySettings.danseUpdating}_{[int(sro) for sro in mySettings.SROsppm]}ppm' # experiment reference label
experimentName = f'{mySettings.danseUpdating}_{[int(sro) for sro in mySettings.SROsppm]}ppm' # experiment reference label
if mySettings.compensateSROs:
    experimentName += '_comp'
else:
    experimentName += '_nocomp'

exportPath = f'{Path(__file__).parent}/res/single_sensor_nodes/{experimentName}'
lightExport = True          # <-- set to True to not export whole signals
# ------------------------

def main(mySettings, exportPath, showPlots=1, lightExport=True):
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
        # Export
        wipe_folder(exportPath)     # wipe export folder    
        results.save(exportPath, lightExport)        # save results
        mySettings.save(exportPath)     # save settings

        # Post-process
        get_figures_and_sound(results, exportPath, mySettings, showPlots, listen=False)

    return None


def get_figures_and_sound(results, pathToResults, settings, showPlots=False, listen=False, listeningMaxDuration=5.):
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
    fig.suptitle(f'Speech enhancement metrics ($\\beta={settings.expAvgBeta}$)')
    fig.tight_layout()
    plt.savefig(f'{pathToResults}/enhMetrics.png')
    if showPlots:
        plt.draw()

    # Plot best performance node (in terms of STOI)
    maxSTOI = 0
    minSTOI = 1
    for idxNode in range(results.acousticScenario.numNodes):
        currSTOIs = results.enhancementEval.stoi[f'Node{idxNode + 1}']
        for idxSensor, stoi in enumerate(currSTOIs):
            if stoi >= maxSTOI:
                bestNode, bestSensor = idxNode, idxSensor
                maxSTOI = stoi
            if stoi <= minSTOI:
                worstNode, worstSensor = idxNode, idxSensor
                minSTOI = stoi
    print(f'Best node (STOI = {round(maxSTOI * 100, 2)}%)')
    results.signals.plot_signals(bestNode, bestSensor, settings)
    plt.savefig(f'{pathToResults}/bestPerfNode.png')
    if showPlots:
        plt.draw()
    print(f'Worst node (STOI = {round(minSTOI * 100, 2)}%)')
    results.signals.plot_signals(worstNode, worstSensor, settings)
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
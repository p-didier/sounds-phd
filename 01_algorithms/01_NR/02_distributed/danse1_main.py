# Using the "danse_env" virtual environment
# %%
from turtle import Turtle
from danse_utilities.classes import ProgramSettings, Results
from danse_utilities.setup import run_experiment
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background
#
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from playsounds.playsounds import playwavfile
# ------------------------

def main():
    """Main function for DANSE runs"""
    # General parameters
    ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
    signalsPath = f'{pathToRoot}/02_data/00_raw_signals'

    # Set experiment settings
    mySettings = ProgramSettings(
        # acousticScenarioPath=f'{ascBasePath}/J3Mk[1, 2, 3]_Ns1_Nn1/AS0_anechoic',
        acousticScenarioPath=f'{ascBasePath}/J2Mk[1 1]_Ns1_Nn1/AS0_anechoic',
        desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
        noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
        signalDuration=10,
        baseSNR=-10,
        plotAcousticScenario=False,
        timeBtwConsecUpdates=0.3,       # time btw. consecutive DANSE filter updates
        VADwinLength=40e-3,             # VAD window length [s]
        VADenergyFactor=4000,           # VAD factor (threshold = max(energy signal)/VADenergyFactor)
        expAvgBeta=0.98,
        minNumAutocorrUpdates=10,
        initialWeightsAmplitude=1,
        performGEVD=True,               # set to True for GEVD-DANSE
        SROsppm=[0, 100],               # SRO
        compensateSROs=True,            # if True, estimate + compensate SRO dynamically
        )
    print(mySettings)

    experimentName = f'SROcompTesting/SROs{mySettings.SROsppm}' # experiment name
    exportPath = f'{Path(__file__).parent}/res/{experimentName}'

    # Check if experiment has already been run
    runExpFlag = False
    if Path(exportPath).is_dir():
        val = input(f'Already existing experiment "{PurePath(exportPath).name}" in folder "{PurePath(Path(exportPath).parent).name}".\nRe-run experiment anyway? [Y/N]  ')
        if val == 'y' or val == 'Y':
            runExpFlag = True
            print(f'\nRe-running experiment "{PurePath(exportPath).name}" ...\n')
        else:
            print('\nNot re-running experiment.\n')
    else:
        runExpFlag = True
    
    if runExpFlag:
        # Run experiment
        results = run_experiment(mySettings)
        # Export
        results.save(exportPath)        # save results
        mySettings.save(exportPath)     # save settings

    # Post-process
    get_figures_and_sound(exportPath, mySettings, showPlots=1, listen=False)


def get_figures_and_sound(pathToResults, settings, showPlots=False, listen=False, listeningMaxDuration=5.):
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

    # Import results
    results = Results().load(pathToResults)
    # Export as WAV
    wavFilenames = results.signals.export_wav(pathToResults)

    # Plot scenario
    fig = results.acousticScenario.plot()
    myPath = settings.acousticScenarioPath
    fig.suptitle(myPath[myPath.rfind('/', 0, myPath.rfind('/')) + 1:-4])
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
    sys.exit(main())
# ------------------------------------------------------------------------------------
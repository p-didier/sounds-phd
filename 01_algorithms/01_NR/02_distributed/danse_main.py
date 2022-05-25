# `danse_env` virtual environment
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
from danse_utilities.classes import ProgramSettings, Results, PrintoutsParameters, SamplingRateOffsets, DWACDParameters, FiltShiftSROEstimationParameters
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
from metrics.eval_enhancement import DynamicMetricsParameters
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
    # samplingFrequency=16000,
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3_3]_Nss1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J1Mk[4]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J3Mk[2_3_4]_Ns1_Nn1/AS5_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1 1 1 1 1]_Ns1_Nn1/AS6_allNodesInSamePosition_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3_1]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1_1]_Ns1_Nn1/AS1_anechoic',
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1_1]_Ns1_Nn1/AS2_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1_1]_Ns1_noiseless/AS2_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[2_2]_Ns1_Nn1/AS1_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1_1]_Ns1_Nn1/AS3_RT500ms',
    # acousticScenarioPath=f'{ascBasePath}/tests/J5Mk[1_1_1_1_1]_Ns1_Nn1/AS10_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[3_1]_Ns1_Nn1/AS2_allNodesInSamePosition_anechoic',
    #
    # desiredSignalFile=[f'{signalsPath}/03_test_signals/tone100Hz.wav'],
    # desiredSignalFile=[f'U:\\py\\sounds-phd\\01_algorithms\\03_signal_gen\\02_noise_maker\\02_sine_combinations\\sounds\\mySineCombination1.wav'],
    desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    #
    signalDuration=20,
    baseSNR=5,
    # baseSNR=-90,
    #
    stftFrameOvlp=0.5,
    stftWinLength=2**10,
    chunkOverlap=0.5,           # overlap between DANSE iteration processing chunks [/100%]
    chunkSize=2**10,            # DANSE iteration processing chunk size [samples]
    broadcastLength=2**9,       # broadcast chunk size `L` [samples]
    # broadcastDomain='t',
    broadcastDomain='f',
    # selfnoiseSNR=-np.Inf,
    # broadcastLength=8,       # broadcast chunk size `L` [samples]
    #
    # vvv SROs parameters vvv
    asynchronicity=SamplingRateOffsets(
        SROsppm=[0, 0],
        # SROsppm=[0, 75],
        # SROsppm=[0, 50],
        # compensateSROs=True,
        compensateSROs=False,
        estimateSROs='Oracle',    # <-- Oracle SRO knowledge, no estimation error
        # estimateSROs='FiltShift',   # <-- Filter shift method (inspired by Thüne & Enzner + Nokia work)
        # estimateSROs='DWACD',     # <-- Dynamic WACD by Gburrek et al.
        dwacd=DWACDParameters(
            seg_shift=2**11,
        ),
        filtShiftsMethod=FiltShiftSROEstimationParameters(
            nFiltUpdatePerSeg=1,
            # estimationMethod='gs',      # golden section search
            # estimationMethod='mean',    # mean method
            estimationMethod='ls',      # least-squares
            startAfterNupdates=30,
        )
    ),
    # bypassFilterUpdates=True,
    #
    expAvg50PercentTime=2.,             # [s] time in the past at which the value is weighted by 50% via exponential averaging
    # expAvg50PercentTime=1.,             # [s] time in the past at which the value is weighted by 50% via exponential averaging
    danseUpdating='simultaneous',       # node-updating scheme
    # danseUpdating='sequential',       # node-updating scheme
    referenceSensor=0,                  # index of reference sensor at each node (same for every node)
    computeLocalEstimate=True,          # if True, also compute and store the local estimate (as if there was no cooperation between nodes)
    performGEVD=1,                      # if True, perform GEVD-DANSE
    # timeBtwExternalFiltUpdates=np.Inf,       # [s] time between 2 consecutive external filter update (for broadcasting) at a node
    timeBtwExternalFiltUpdates=3,       # [s] time between 2 consecutive external filter update (for broadcasting) at a node
    # timeBtwExternalFiltUpdates=0,       # [s] time between 2 consecutive external filter update (for broadcasting) at a node
    # 
    # vvv Printouts parameters vvv
    printouts=PrintoutsParameters(events_parser=True,
                                    externalFilterUpdates=True,),
    #
    dynamicMetricsParams=DynamicMetricsParameters(chunkDuration=0.5,   # [s]         # dynamic speech enhancement metrics computation parameters
                                    chunkOverlap=0.5,   # [/100%]
                                    dynamicfwSNRseg=True,
                                    dynamicSNR=True)
    )

# Subfolder for export
subfolder = f'testing_SROs/single_tests/{mySettings.danseUpdating}/{Path(mySettings.acousticScenarioPath).parent.name}'
# experimentName = f'SROcompTesting/SROs{mySettings.SROsppm}' # experiment reference label
# experimentName = f'testing_SROs/single_tests/{mySettings.danseUpdating}_{[int(sro) for sro in mySettings.SROsppm]}ppm' # experiment reference label
experimentName = f'{Path(mySettings.acousticScenarioPath).name}_{[int(sro) for sro in mySettings.asynchronicity.SROsppm]}ppm_{int(mySettings.signalDuration)}s' # experiment reference label
if (np.array(mySettings.asynchronicity.SROsppm) != 0).any():
    if mySettings.asynchronicity.compensateSROs:
        experimentName += f'_comp{mySettings.asynchronicity.estimateSROs}'
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
            # Post-process - Speech enhancement metrics, figures, sounds
            flagFigsAndSound = True
            if not np.array(['speech' in str(s) for s in mySettings.desiredSignalFile]).any():   # check that we are dealing with speech
                inp = input(f"""No "speech" sub-string detected in `desiredSignalFile` paths.
                        --> Possibly not using an actual speech target signal.
                        --> Compute intelligibility metrics (+ figs/wavs export) anyway? [y]/n  """)
                if inp not in ['y', 'Y']:
                    print('Not computing intelligibility metrics (+ figs/wavs export).')
                    flagFigsAndSound = False
            if flagFigsAndSound:
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
    fig1, fig2 = results.plot_enhancement_metrics(plotLocal=False)
    expAvgTau = np.log(0.5) * settings.stftEffectiveFrameLen / (np.log(settings.expAvgBeta) * results.signals.fs[0])
    fig1.suptitle(f"""Speech enhancement metrics ($\\beta={np.round(settings.expAvgBeta, 4)}
\Leftrightarrow \\tau_{{50\%, 0\\mathrm{{ppm}}}} = {np.round(expAvgTau, 2)}$ s)""".replace('\n',' '))   # long-string trick https://stackoverflow.com/a/24331604
    fig1.tight_layout()
    fig1.savefig(f'{pathToResults}/enhMetrics.png')
    if showPlots:
        plt.draw()
    if fig2 is not None:
        # Dynamic metrics
        fig2.suptitle(f"""Dynamic metrics [{settings.dynamicMetricsParams.chunkDuration}s
chunks, {int(settings.dynamicMetricsParams.chunkOverlap * 100)}% overlap] ($\\beta={np.round(settings.expAvgBeta, 4)}
\Leftrightarrow \\tau_{{50\%, 0\\mathrm{{ppm}}}} = {np.round(expAvgTau, 2)}$ s)""".replace('\n',' '))   # long-string trick https://stackoverflow.com/a/24331604
        fig2.tight_layout()
        fig2.savefig(f'{pathToResults}/enhDynamicMetrics.png')
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

    plt.close("all")    # close all figures

    if listen:
        # LISTEN
        playwavfile(wavFilenames['Noisy'][bestNode], listeningMaxDuration)
        playwavfile(wavFilenames['Desired'][bestNode], listeningMaxDuration)
        playwavfile(wavFilenames['Enhanced'][bestNode], listeningMaxDuration)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(mySettings, exportPath, lightExport))
# ------------------------------------------------------------------------------------
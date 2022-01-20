# Using the "danse_env" virtual environment

# %%
from danse_utilities.classes import AcousticScenario, ProgramSettings, Results
from danse_utilities.setup import run_experiment
from pathlib import Path, PurePath
import sys, os
import simpleaudio as sa
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

ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'

experimentName = 'firsttry.csv' # set experiment name
# Set experiment settings
mySettings = ProgramSettings(
    acousticScenarioPath=f'{ascBasePath}/J3Mk[1, 2, 3]_Ns1_Nn1/AS0_anechoic',
    # acousticScenarioPath=f'{ascBasePath}/J4Mk[10 10 10 10]_Ns1_Nn1/AS0_anechoic',
    desiredSignalFile=[f'{signalsPath}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{signalsPath}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    signalDuration=10,
    baseSNR=0,
    plotAcousticScenario=False,
    timeBtwConsecUpdates=0.3,       # time btw. consecutive DANSE filter updates
    VADwinLength=40e-3,             # VAD window length [s]
    VADenergyFactor=4000,           # VAD factor (threshold = max(energy signal)/VADenergyFactor)
    expAvgBeta=0.98,
    minNumAutocorrUpdates=10,
    initialWeightsAmplitude=1
    )
# mySettings.save(experimentName) # save settings
print(mySettings)

#%%
# Run experiment
results = run_experiment(mySettings)
exportPath = f'{Path(__file__).parent}/res/testrun'
results.save(exportPath)

#%%

# Import results
exportPath = f'{Path(__file__).parent}/res/testrun'
importedResults = Results().load(exportPath)
# Export as WAV
wavFilenames = importedResults.signals.export_wav(exportPath)

#%%
exportFigs = True

# Plot scenario
fig = importedResults.acousticScenario.plot()
myPath = mySettings.acousticScenarioPath
fig.suptitle(myPath[myPath.rfind('/', 0, myPath.rfind('/')) + 1:-4])
fig.tight_layout()
if exportFigs:
    plt.savefig(f'{exportPath}/acousScenario.png')
plt.show()
plt.close()

# Plot performance
fig = importedResults.plot_enhancement_metrics()
fig.suptitle(f'Speech enhancement metrics ($\\beta={mySettings.expAvgBeta}$)')
if exportFigs:
    plt.savefig(f'{exportPath}/enhMetrics.png')
plt.show()
plt.close()

# Plot best performance node (in terms of STOI)
maxSTOI = 0
minSTOI = 1
for idxNode in range(importedResults.acousticScenario.numNodes):
    currSTOIs = importedResults.enhancementEval.stoi[f'Node{idxNode + 1}']
    for idxSensor, stoi in enumerate(currSTOIs):
        if stoi >= maxSTOI:
            bestNode, bestSensor = idxNode, idxSensor
            maxSTOI = stoi
        if stoi <= minSTOI:
            worseNode, worseSensor = idxNode, idxSensor
            minSTOI = stoi
print(f'Best node (STOI = {round(maxSTOI * 100, 2)}%)')
importedResults.signals.plot_signals(bestNode, bestSensor, mySettings.expAvgBeta)
if exportFigs:
    plt.savefig(f'{exportPath}/bestPerfNode.png')
plt.show()
plt.close()
print(f'Worst node (STOI = {round(minSTOI * 100, 2)}%)')
importedResults.signals.plot_signals(worseNode, worseSensor, mySettings.expAvgBeta)
if exportFigs:
    plt.savefig(f'{exportPath}/worstPerfNode.png')
plt.show()
plt.close()

# %% LISTEN

playwavfile(wavFilenames['Noisy'][bestNode])
playwavfile(wavFilenames['Desired'][bestNode])
playwavfile(wavFilenames['Enhanced'][bestNode])

# %%

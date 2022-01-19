# Using the "danse_env" virtual environment

# %%
from danse_utilities.classes import AcousticScenario, ProgramSettings, Results
from danse_utilities.setup import run_experiment
from pathlib import Path
import matplotlib.pyplot as plt


ASBASEPATH = '/users/sista/pdidier/py/sounds-phd/02_data/01_acoustic_scenarios'
SIGNALSPATH = '/users/sista/pdidier/py/sounds-phd/02_data/00_raw_signals'

experimentName = 'firsttry.csv' # set experiment name
# Set experiment settings
mySettings = ProgramSettings(
    acousticScenarioPath=f'{ASBASEPATH}/J3Mk[1, 2, 3]_Ns1_Nn1/AS0_anechoic',
    desiredSignalFile=[f'{SIGNALSPATH}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    signalDuration=5,
    baseSNR=10,
    plotAcousticScenario=True,
    VADwinLength=40e-3,             # VAD window length [s]
    VADenergyFactor=4000,           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    )
# mySettings.save(experimentName) # save settings
print(mySettings)

#%%
# Run experiment
results = run_experiment(mySettings)

# Export results
exportPath = f'{Path(__file__).parent}/res/testrun'
results.save(exportPath)


# %%


exportPath = f'{Path(__file__).parent}/res/testrun'
# Import results
importedResults = Results().load(exportPath)
#%%
# Plot scenario
fig = importedResults.acousticScenario.plot()
myPath = mySettings.acousticScenarioPath
fig.suptitle(myPath[myPath.rfind('/', 0, myPath.rfind('/')) + 1:-4])
plt.tight_layout()
plt.savefig(f'{exportPath}/acousScenario.png')

# Plot performance
importedResults.plot_enhancement_metrics()
plt.savefig(f'{exportPath}/enhMetrics.png')

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
print(f'Best node (STOI = {maxSTOI * 100}%)')
importedResults.signals.plot_signals(bestNode, bestSensor)
plt.savefig(f'{exportPath}/bestPerfNode.png')
print(f'Worst node (STOI = {minSTOI * 100}%)')
importedResults.signals.plot_signals(worseNode, worseSensor)
plt.savefig(f'{exportPath}/worstPerfNode.png')


# %%

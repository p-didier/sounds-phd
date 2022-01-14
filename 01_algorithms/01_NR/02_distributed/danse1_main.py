# Using the "danse_env" virtual environment

# %%
from danse_utilities.classes import *
from danse_utilities.scripts import *

ASBASEPATH = os.path.join(os.path.expanduser('~'), 'py/sounds-phd/02_data/01_acoustic_scenarios/')
SIGNALSPATH = os.path.join(os.path.expanduser('~'), 'py/sounds-phd/02_data/00_raw_signals/')

experimentName = 'firsttry.csv' # set experiment name
# Set experiment settings
mySettings = ProgramSettings( 
    acousticScenarioPath = f'{ASBASEPATH}J2Mk1_Ns1_Nn1/AS0_anechoic',
    desiredSignalFile = [f'{SIGNALSPATH}01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile = [f'{SIGNALSPATH}02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    signalDuration=5,
    baseSNR = 10,
    plotAcousticScenario=True,
    )
# mySettings.save(experimentName) # save settings

print(mySettings)

# Run experiment
results = runExperiment(mySettings)
# results.save(f"out_{experimentName}")

# fig, ax = plt.subplots()
# ax.plot(results)
# ax.grid()

stop = 1

# %%

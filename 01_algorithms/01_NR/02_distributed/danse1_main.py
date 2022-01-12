# Using the "danse_env" virtual environment

# %%
from danse_utilities.classes import *
from danse_utilities.scripts import *

experimentName = 'firsttry.csv' # set experiment name

# Set experiment settings
mySettings = ProgramSettings(
    samplingFrequency=16e3, 
    signalDuration=3
    )
mySettings.save(experimentName) # save settings

# Run experiment
results = runExperiment(mySettings)
results.save(f"out_{experimentName}")

# %%

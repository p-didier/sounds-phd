# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Validation script for the implementation of the sequential GEVD-DANSE algorithm
# gevddanse_valid.py original creation date: 01/27/2022 - 10:43
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
import sys, os
import danse_main
from pathlib import Path, PurePath
from danse_utilities.classes import ProgramSettings
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
# ------------------------

# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios/validations'
signalsPath = f'{Path(__file__).parent}/validations/signals'
exportBasePath = f'{Path(__file__).parent}/validations'

# List all acoustic scenarios folder paths
acsSubDirs = list(Path(ascBasePath).iterdir())
acousticScenarios = []
for ii in range(len(acsSubDirs)):
    acss = list(Path(acsSubDirs[ii]).iterdir())
    for jj in range(len(acss)):
        acousticScenarios.append(acss[jj])
# List all sound files paths
speechFiles = [f for f in Path(f'{signalsPath}/speech').glob('**/*') if f.is_file()]
speechFile = speechFiles[0]     # select one speech signal
noiseFiles = [f for f in Path(f'{signalsPath}/noise').glob('**/*') if f.is_file()]

# Build experiments list
experiments = []
for i1, asc in enumerate(acousticScenarios):
    for i2, noise in enumerate(noiseFiles):
        sets = ProgramSettings(
                acousticScenarioPath=asc,       # Loop 1
                desiredSignalFile=speechFile,
                noiseSignalFile=noise,          # Loop 2
                signalDuration=10,
                baseSNR=-10,
                plotAcousticScenario=False,
                VADwinLength=40e-3,
                VADenergyFactor=4000,
                expAvgBeta=0.98,
                minNumAutocorrUpdates=10,
                performGEVD=1,
                SROsppm=[0],
                )
        exportPath = f'{exportBasePath}/danse_res/{acousticScenarios[i1].parent.name}_{acousticScenarios[i1].name}_{os.path.splitext(noise.name)[0]}'     # experiment export path
        experiments.append(dict([('sets', sets), ('path', exportPath)]))

print(f'About to run {len(experiments)} DANSE simulations...')


def main(experiments):
    """Main wrapper for running validation experiment of DANSE implementation.
    
    Parameters
    ----------
    experiments : list of dictionaries
        Each element of <experiments> is a dictionary containing
        a ProgramSettings object ("sets" key) and an export
        path string ("path" key).
    """
    for idxExp in range(len(experiments)):

        if not Path(experiments[idxExp]['path']).is_dir():
            danse_main.main(experiments[idxExp]['sets'], experiments[idxExp]['path'], showPlots=0)
        else:
            print(f'...NOT RUNNING "{Path(experiments[idxExp]["path"]).name}"...')


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(experiments))
# ------------------------------------------------------------------------------------
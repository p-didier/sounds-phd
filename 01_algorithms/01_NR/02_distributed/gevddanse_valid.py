# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Validation script for the implementation of the sequential GEVD-DANSE algorithm
# gevddanse_valid.py original creation date: 01/27/2022 - 10:43
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
import sys
import danse_main
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
# ------------------------

# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{Path(__file__).parent}/validations/signals'

# Subfolder where acoustic scenarios are stored
ascSubdir = 'validations'

# List all acoustic scenarios folders
acousticScenarios = list(Path(f'{ascBasePath}/{ascSubdir}').iterdir())
# List all sound files
speechFiles = [f for f in Path(f'{signalsPath}/speech').glob('**/*') if f.is_file()]
noiseFiles = [f for f in Path(f'{signalsPath}/noise').glob('**/*') if f.is_file()]
# List all exponential averaging constant values
betaValues = []

#%%

def main():
    
    # Build experiments object
    experiments = []

    for idxExp in range(len(experiments)):

        danse_main.main(experiments[idxExp].settings)


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Acoustic scenario generation script for generating many scenarios automatically.
# File original creation date: 01/27/2022 - 11:19
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%%
import ASgen
import sys
from utils.classes import ASCProgramSettings
import itertools
import numpy as np
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
# ------------------

seed = 12345
# Create random generator
rng = np.random.default_rng(seed)

# ---------- PARAMETERS ----------
# Acoustic scenarios layouts
ascPoor = dict([('J', 2), ('Mk', [3, 2])])
possibleCombs = [list(p) for p in itertools.product([2, 3, 5], repeat=3)]
ascMedium = dict([('J', 3), ('Mk', rng.choice(possibleCombs))])
possibleCombs = [list(p) for p in itertools.product([2, 3, 5], repeat=6)]
ascGood = dict([('J', 6), ('Mk', rng.choice(possibleCombs))])
# Reverberation times 
rtPoor = 0.6    # quite reverberant
rtMedium = 0.2  # a little reverberant
rtGood = 0      # anechoic
# Number of good-medium-poor scenarios to generate
nAS = 1
# Export folder
basepath = f'{pathToRoot}/02_data/01_acoustic_scenarios/validations'

# ---------- PROCESSING ----------
ascs = [ascPoor, ascMedium, ascGood]
rts = [rtPoor, rtMedium, rtGood]
experiments = []
for idxRT in range(len(rts)):
    for idxASC in range(len(ascs)):
        exp = ASCProgramSettings(
            roomDimBounds=[3,7],
            numScenarios=nAS,
            numNodes=ascs[idxASC]['J'],
            numSensorPerNode=ascs[idxASC]['Mk'],
            arrayGeometry='radius',
            revTime=rts[idxRT]
        )
        d = dict([('sets', exp), ('path', basepath)])
        experiments.append(d)
# ---------------------------------


def main(experiments):
    """Wrapper for automated acoustic scenario generation.
    
    Parameters
    ----------
    experiments : list of dictionaries
        Each element of <experiments> is a dictionary containing
        an ASCProgramSettings object ("sets" key) and an export
        path string ("path" key).
    """

    for idxExp in range(len(experiments)):
        print(f'ACOUSTIC SCENARIOS AUTOMATED GENERATION {idxExp + 1}/{len(experiments)}...\n')
        ASgen.main(experiments[idxExp]['sets'], experiments[idxExp]['path'])
    print('DONE.')

    return None

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(experiments))
# ------------------------------------------------------------------------------------

# %%
''
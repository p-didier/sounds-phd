
#%%
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import numpy as np
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_utilities.classes import Results, ProgramSettings
sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes')

# resultsBaseFolder = f'{Path(__file__).parent}/automated/10s_signals_2nodes_whitenoise_3sensorseach'
# resultsBaseFolder = f'{Path(__file__).parent}/automated/10s_signals_2nodes_whitenoise_2sensorseach'
# resultsBaseFolder = f'{Path(__file__).parent}/automated/10s_signals_2nodes_whitenoise_1sensoreach'
resultsBaseFolder = f'{Path(__file__).parent}/automated/J2Mk[3, 1]_Ns1_Nn1/RT500ms/Leq512'
exportFileName = f'{resultsBaseFolder}/postProcessed'  # + ".png" & ".pdf"

resSubDirs = list(Path(resultsBaseFolder).iterdir())
resSubDirs = [f for f in resSubDirs if f.name[-1] == ']']

# Find number of nodes from number of SRO values
idxStart = resSubDirs[0].name.rfind('[')
nNodes = len([int(s) for s in resSubDirs[0].name[idxStart+1:-1].split(', ') if s.isdigit()])

# Set default flags
flagDelta_stoi = False
flagDelta_fwSNRseg = False

stois = np.zeros((nNodes, len(resSubDirs)))
fwSNRsegs = np.zeros((nNodes, len(resSubDirs)))
sros = np.zeros((nNodes, len(resSubDirs)))
idxbenchmark = np.nan
for ii in range(len(resSubDirs)):
    resObject = Results().load(resSubDirs[ii], silent=True)
    if resObject.acousticScenario.numNodes != nNodes:
        raise ValueError('Mismatch between expected vs. actual number of nodes in WASN.')
    params = ProgramSettings().load(resSubDirs[ii], silent=True)
    for idx, val in enumerate(resObject.enhancementEval.stoi.values()):
        if not isinstance(val, float):
            val = val[-1]   # plot before/after enhancement improvement
            flagDelta_stoi = True
        stois[idx, ii] = val
    for idx, val in enumerate(resObject.enhancementEval.fwSNRseg.values()):
        if not isinstance(val, float):
            val = val[-1]   # plot before/after enhancement improvement
            flagDelta_fwSNRseg = True
        fwSNRsegs[idx, ii] = val
    sros[:, ii] = params.SROsppm
    if all(v == 0 for v in params.SROsppm):
        idxbenchmark = ii

# Pre-plot
if nNodes == 2:
    sros[sros == 0] = np.nan    # ensures that the 0ppm nodes are not counted inside the mean
    if not np.isnan(idxbenchmark):
        sros = np.insert(sros, idxbenchmark, np.zeros(sros.shape[0]), axis=1)   # don't forget the [0,0..,0] SROs 
    srosMean = np.nanmean(sros, axis=0)
    srosMean = srosMean[~np.isnan(srosMean)]    # get rid of the extra column remaining due to benchmark (0 SROs) case
else:
    srosMean = np.mean(sros, axis=0)
idxSorting = np.argsort(srosMean)
srosMean = srosMean[idxSorting]
stois = stois[:, idxSorting]
fwSNRsegs = fwSNRsegs[:, idxSorting]

# PLOT
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(121)
for idxNode in range(nNodes):
    ax.semilogx(srosMean, stois[idxNode, :], f'C{idxNode}-o', label=f'Node {idxNode+1}')
ax.grid()
# ax.set_ylim([0, 1])
ax.set_xlabel('Absolute SRO with neighbor node [ppm]')
if flagDelta_stoi:
    ax.set_ylabel('$\\Delta$STOI')
else:
    ax.set_ylabel('STOI')
#
ax = fig.add_subplot(122)
for idxNode in range(nNodes):
    ax.semilogx(srosMean, fwSNRsegs[idxNode, :] , f'C{idxNode}-o', label=f'Node {idxNode+1}')
ax.grid()
# ax.set_ylim([-5, 3])
ax.set_xlabel('Absolute SRO with neighbor node [ppm]')
if flagDelta_fwSNRseg:
    ax.set_ylabel('$\\Delta$fwSNRseg [dB]')
else:
    ax.set_ylabel('fwSNRseg [dB]')
plt.tight_layout()
plt.legend()
# fig.savefig(exportFileName + ".png")
# fig.savefig(exportFileName + ".pdf")
plt.show()

stop = 1

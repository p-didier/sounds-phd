
#%%
from msilib import typemask
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
from matplotlib import markers
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
# resultsBaseFolder = f'{Path(__file__).parent}/automated/J2Mk[3, 1]_Ns1_Nn1/anechoic_Leq8'
resultsBaseFolder = [f'{Path(__file__).parent}/automated/J2Mk[3, 1]_Ns1_Nn1/anechoic_Leq512',\
                        f'{Path(__file__).parent}/automated/J2Mk[3, 1]_Ns1_Nn1/anechoic_Leq8']
TYPEMETRIC = 'improvement'
TYPEMETRIC = 'afterEnhancement'


def main():

    
    # Global results storing dictionary
    res = dict([('stoi', []), ('fwSNRseg', []), ('sro', []), ('ref', [])])

    if isinstance(resultsBaseFolder, list):

        exportFileName = f'{Path(resultsBaseFolder[0]).parent}/metricAsFcnOfSRO_{TYPEMETRIC}'  # + ".png" & ".pdf"
        
        for ii in range(len(resultsBaseFolder)):
            resSubDirs = list(Path(resultsBaseFolder[ii]).iterdir())
            resSubDirs = [f for f in resSubDirs if f.name[-1] == ']']
            # Find number of nodes from number of SRO values
            idxStart = resSubDirs[0].name.rfind('[')
            nNodesCurr = len([int(s) for s in resSubDirs[0].name[idxStart+1:-1].split(', ') if s.isdigit()])
            # Check coherence in number of nodes
            if ii == 0:
                nNodes = nNodesCurr
            elif nNodesCurr != nNodes:
                raise ValueError('The number of nodes in each subfolder is not matching.')
                
            stois, fwSNRsegs, sros, flagDelta_stoi, flagDelta_fwSNRseg = extract_metrics(nNodes, resSubDirs, typeMetric=TYPEMETRIC)

            res['stoi'].append(stois)
            res['fwSNRseg'].append(fwSNRsegs)
            res['sro'].append(sros)
            res['ref'].append(PurePath(resultsBaseFolder[ii]).name)

    elif isinstance(resultsBaseFolder, str):

        exportFileName = f'{resultsBaseFolder}/metricAsFcnOfSRO_{TYPEMETRIC}'  # + ".png" & ".pdf"

        resSubDirs = list(Path(resultsBaseFolder).iterdir())
        resSubDirs = [f for f in resSubDirs if f.name[-1] == ']']
        # Find number of nodes from number of SRO values
        idxStart = resSubDirs[0].name.rfind('[')
        nNodes = len([int(s) for s in resSubDirs[0].name[idxStart+1:-1].split(', ') if s.isdigit()])
        #
        stois, fwSNRsegs, sros, flagDelta_stoi, flagDelta_fwSNRseg = extract_metrics(nNodes, resSubDirs, typeMetric=TYPEMETRIC)
        res['stoi'].append(stois)
        res['fwSNRseg'].append(fwSNRsegs)
        res['sro'].append(sros)
        res['ref'].append(PurePath(resultsBaseFolder).name)


    # PLOT
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)
    for idxNode in range(nNodes):
        for ii in range(len(res['stoi'])):
            ax.semilogx(res['sro'][ii], res['stoi'][ii][idxNode, :],\
                        f'C{idxNode * len(res["stoi"]) + ii}-o',\
                        label=f'Node {idxNode+1} -- {res["ref"][ii]}', markersize=2)
    ax.grid()
    ax.set_xlabel('Absolute SRO with neighbor node [ppm]')
    if flagDelta_stoi:
        ax.set_ylabel('$\\Delta$STOI')
    else:
        ax.set_ylabel('STOI')
    plt.legend()
    #
    ax = fig.add_subplot(122)
    for idxNode in range(nNodes):
        for ii in range(len(res['fwSNRseg'])):
            ax.semilogx(res['sro'][ii], res['fwSNRseg'][ii][idxNode, :],\
                        f'C{idxNode * len(res["fwSNRseg"]) + ii}-o',\
                        label=f'Node {idxNode+1} -- {res["ref"][ii]}', markersize=2)
    ax.grid()
    ax.set_xlabel('Absolute SRO with neighbor node [ppm]')
    if flagDelta_fwSNRseg:
        ax.set_ylabel('$\\Delta$fwSNRseg [dB]')
    else:
        ax.set_ylabel('fwSNRseg [dB]')
    plt.tight_layout()
    fig.savefig(exportFileName + ".png")
    fig.savefig(exportFileName + ".pdf")
    plt.show()

    stop = 1


def extract_metrics(nNodes, resSubDirs, typeMetric='improvement'):

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
                if typeMetric == 'improvement':
                    val = val[-1]   # plot before/after enhancement improvement
                    flagDelta_stoi = True
                elif typeMetric == 'afterEnhancement':
                    val = val[1]   # plot before/after enhancement improvement
            stois[idx, ii] = val
        for idx, val in enumerate(resObject.enhancementEval.fwSNRseg.values()):
            if not isinstance(val, float):
                if typeMetric == 'improvement':
                    val = val[-1]   # plot before/after enhancement improvement
                    flagDelta_fwSNRseg = True
                elif typeMetric == 'afterEnhancement':
                    val = val[1]   # plot before/after enhancement improvement
            fwSNRsegs[idx, ii] = val
        sros[:, ii] = params.SROsppm
        if all(v == 0 for v in params.SROsppm):
            idxbenchmark = ii

    # Sorting by increasing SROs
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

    return stois, fwSNRsegs, srosMean, flagDelta_stoi, flagDelta_fwSNRseg


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
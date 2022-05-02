
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

# resultsBaseFolder = [f'{Path(__file__).parent}/automated/J2Mk[1, 1]_Ns1_Nn1/{ii}' for ii in ['Leq512', 'Leq8']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/J2Mk[1, 1]_Ns1_Nn1/{ii}' for ii in ['Leq512']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/J3Mk[2, 3, 4]_Ns1_Nn1/{ii}' for ii in ['Leq8']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/J2Mk[1, 1]_Ns1_Nn1']
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/with_compression/freq_domain_compression/J2Mk[1, 1]_Ns1_Nn1/{ii}' for ii in ['no_compensation', 'oracle_compensation']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/with_compression/freq_domain_compression/J2Mk[1, 1]_Ns1_Nn1/{ii}' for ii in ['no_compensation']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/J2Mk[1, 1]_Ns1_Nn1/with_perfectly_delayed_initial_updates']
resultsBaseFolder = [f'{Path(__file__).parent}/automated/20220502_impactOfReverb/J2Mk[1_1]_Ns1_Nn1/{ii}' for ii in ['anechoic_nocomp', 'anechoic_comp']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/20220502_impactOfReverb/J2Mk[1_1]_Ns1_Nn1/{ii}' for ii in ['RT200ms_nocomp', 'RT200ms_comp']]
# resultsBaseFolder = [f'{Path(__file__).parent}/automated/20220502_impactOfReverb/J2Mk[1_1]_Ns1_Nn1/{ii}' for ii in ['RT400ms_nocomp', 'RT400ms_comp']]
resultsBaseFolder = [f'{Path(__file__).parent}/automated/20220502_impactOfReverb/J2Mk[1_1]_Ns1_Nn1/{ii}' for ii in \
    ['anechoic_nocomp', 'anechoic_comp',\
    'RT200ms_nocomp', 'RT200ms_comp',\
        'RT400ms_nocomp', 'RT400ms_comp']]
# givenFormats, givenLegend = None, None
givenFormats = ['C0o-','C0o:','C1o-','C1o:','C2o-','C2o:']
givenLegend = ['Anechoic', 'Anechoic - Oracle compensation',\
            '$T_{{60}}=0.2$s', '$T_{{60}}=0.2$s - Oracle compensation',\
            '$T_{{60}}=0.4$s', '$T_{{60}}=0.4$s - Oracle compensation']

TYPEMETRIC = 'improvement'
TYPEMETRIC = 'afterEnhancement'
SINGLENODEPLOT = True

def main():
    
    # Global results storing dictionary
    res = dict([('stoi', []), ('fwSNRseg', []), ('sro', []), ('ref', [])])

    if isinstance(resultsBaseFolder, list):

        exportFileName = f'{Path(resultsBaseFolder[0]).parent}/metrics_{TYPEMETRIC}'  # + ".png" & ".pdf"
        
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

        exportFileName = f'{resultsBaseFolder}/metrics_{TYPEMETRIC}'  # + ".png" & ".pdf"

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

    # Plot
    plotit(nNodes,
            res,
            flagDelta_stoi, flagDelta_fwSNRseg,
            exportFileName,
            givenFormats, givenLegend,
            ylims=[0, 6.5])


def plotit(nNodes, res, flagDelta_stoi, flagDelta_fwSNRseg, exportFileName, givenFormats=None, givenLegend=None, ylims=None):
    """Plotting function."""
    
    colors = [f'C{i}' for i in range(nNodes)]
    styles = ['-', '--', '-.', ':']
    markers = ['o', 'x', '+', 'd']
    if len(styles) < len(res['stoi']) and len(givenFormats) < len(res['stoi']):
        raise ValueError('Plotting issue: not enough linestyles given for the number of subfolders results to plot.')

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)
    _subplot(ax, nNodes, res['sro'], res['stoi'], res['ref'], colors, styles, markers, flagDelta_stoi, 'STOI', givenFormats)
    ax.set_ylim([0, 1])
    if givenLegend is not None and len(givenLegend) == len(res['stoi']):
        plt.legend(labels=givenLegend)
    else:
        plt.legend()
    #
    ax = fig.add_subplot(122)
    _subplot(ax, nNodes, res['sro'], res['fwSNRseg'], res['ref'], colors, styles, markers, flagDelta_fwSNRseg, 'fwSNRseg', givenFormats)
    if ylims is not None:
        ax.set_ylim(ylims)
    plt.tight_layout()
    fig.savefig(exportFileName + ".png")
    fig.savefig(exportFileName + ".pdf")
    plt.show()


def _subplot(ax, nNodes, sros, metric, refs, colors, styles, markers, flagDelta, ylab, givenFormats=None):
    """Subplot helper function for conciseness."""

    for idxNode in range(nNodes):
        if SINGLENODEPLOT and idxNode > 0:
            continue
        for ii in range(len(metric)):
            if givenFormats is not None:
                fmt = givenFormats[ii]
            else:
                if not SINGLENODEPLOT:
                    fmt = f'{colors[idxNode]}{styles[ii]}{markers[ii]}'
                else:
                    fmt = f'{colors[ii]}{styles[0]}{markers[0]}'
            ax.plot(sros[ii], metric[ii][idxNode, :],\
                        fmt,label=f'Node {idxNode+1} -- {refs[ii]}', markersize=5)
    ax.grid()
    ax.set_xlabel('Absolute SRO with neighbor node [ppm]')
    if flagDelta:
        ax.set_ylabel(f'$\\Delta${ylab}')
    else:
        ax.set_ylabel(ylab)

    return None


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
            if typeMetric == 'improvement':
                stois[idx, ii] = val.diff  # plot before/after enhancement improvement
                flagDelta_stoi = True
            elif typeMetric == 'afterEnhancement':
                stois[idx, ii] = val.after   # plot before/after enhancement improvement
        for idx, val in enumerate(resObject.enhancementEval.fwSNRseg.values()):
            if typeMetric == 'improvement':
                fwSNRsegs[idx, ii] = val.diff  # plot before/after enhancement improvement
                flagDelta_fwSNRseg = True
            elif typeMetric == 'afterEnhancement':
                fwSNRsegs[idx, ii] = val.after   # plot before/after enhancement improvement
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
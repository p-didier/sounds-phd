from dataclasses import dataclass, field
import json
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import dataclass_wizard as dcw

PATHTORESULTS = '01_algorithms/01_NR/02_distributed/res/testing_SROs/automated/20220822_SROestComp_wFlags/J2Mk[1_1]_Ns1_Nn1'
RESULTSFILENAME = 'Results.json'
SETTINGSFILENAME = 'ProgramSettings.json'
EXPORTFIGURE = 1

@dataclass
class Options:
    metricsToPlot: list[str]    # speech enhancement metrics to be plotted
    valuesToPlot: str     # type of values to be plotted (e.g., "before" for before enhancement, "after", "diff", "diffLocal", etc.)
    compOptions: list[str] = field(default_factory=list)
    singleNode: list = field(default_factory=list)
    nodeIndices: np.ndarray = np.array([])

def main():

    opts = Options(
        # metricsToPlot=['stoi', 'pesq', 'fwSNRseg'],
        metricsToPlot=['stoi', 'fwSNRseg'],
        valuesToPlot='diff',
        # valuesToPlot=['after'],
        compOptions=['Comp.', 'No comp.'],
        singleNode=[True, 0]    # [yes/no, node index [python]]
        # singleNode=[True, 1]    # [yes/no, node index [python]]
        # singleNode=[False, None]    # [yes/no, node index [python]]
        )

    sros, data = process_data(opts)

    fig = plot_data(sros, data, opts)

    if EXPORTFIGURE:
        fig.savefig(f'{Path(PATHTORESULTS).parent}/figs/metricsFigure.png')
        fig.savefig(f'{Path(PATHTORESULTS).parent}/figs/metricsFigure.pdf')
        plt.show()

    return None


def process_data(opts: Options):

    # Find subfolders
    dirs = os.listdir(PATHTORESULTS)
    dirs = [ii for ii in dirs if os.path.isdir(PATHTORESULTS + '/' + ii)]

    # Find number of nodes
    idx = PATHTORESULTS.rfind('/')
    opts.nodeIndices = np.arange(int(PATHTORESULTS[idx+2]))
    if opts.singleNode[0]:
        opts.nodeIndices = [opts.singleNode[1]]

    # Init dictionary
    res = dict.fromkeys(dirs, None)
    for dir in dirs:
        subdirs = os.listdir(PATHTORESULTS + '/' + dir)

        # Find all SRO values from subdirectory names
        sros = []
        for subdir in subdirs:
            idxStart = subdir.find(',')
            idxFinish = subdir.find(']')
            sroCurr = float(subdir[idxStart + 2:idxFinish])
            if sroCurr not in sros:
                sros.append(sroCurr)
        sros.sort()

        # Build full nested dictionary
        res[dir] = dict.fromkeys(opts.compOptions, None)
        for aa in range(len(opts.compOptions)):
            res[dir][opts.compOptions[aa]] = dict.fromkeys([f'N{ii + 1}' for ii in opts.nodeIndices], None)
            for ii in opts.nodeIndices:
                res[dir][opts.compOptions[aa]][f'N{ii + 1}'] = dict.fromkeys(opts.metricsToPlot, None)
                for jj in range(len(opts.metricsToPlot)):
                    res[dir][opts.compOptions[aa]][f'N{ii + 1}'][opts.metricsToPlot[jj]] = np.full(len(sros), fill_value=None)

        for _, subdir in enumerate(subdirs):
            # Load results file
            pathToFile = f'{PATHTORESULTS}/{dir}/{subdir}/{RESULTSFILENAME}'
            with open(pathToFile) as f:
                d = json.load(f)

            # Load settings file
            pathToFile = f'{PATHTORESULTS}/{dir}/{subdir}/{SETTINGSFILENAME}'
            with open(pathToFile) as f:
                sets = json.load(f)
            if sets['asynchronicity']['compensateSROs']:
                currCompOption = 'Comp.'
            else:
                currCompOption = 'No comp.'

            srosCurr = [int(s) for s in sets['asynchronicity']['sROsppm'][1:-1].split(' ') if s.isdigit()]
            currSro = np.amax([int(ii) for ii in srosCurr])
            idxSrosList = sros.index(currSro)

            for ii in opts.nodeIndices:
                for jj in range(len(opts.metricsToPlot)):
                    res[dir][currCompOption][f'N{ii + 1}'][opts.metricsToPlot[jj]][idxSrosList] = \
                        d['enhancementEval'][opts.metricsToPlot[jj]][f'Node{ii + 1}'][opts.valuesToPlot]

                    # Edge case: SRO = 0 PPM -- not computed: simply copy "Comp." value to "No comp." entry
                    if currCompOption == 'No comp.' and currSro == 0:
                        res[dir]['Comp.'][f'N{ii + 1}'][opts.metricsToPlot[jj]][idxSrosList] = \
                            d['enhancementEval'][opts.metricsToPlot[jj]][f'Node{ii + 1}'][opts.valuesToPlot]

    return sros, res


def plot_data(sros, data: dict, opts: Options):

    nSubplots = len(opts.metricsToPlot)

    dirs = list(data.keys())

    fig, axes = plt.subplots(1, nSubplots)
    fig.set_size_inches(8.5, 4.5)
    for ii in range(nSubplots):
        for jj in range(len(dirs)): 
            for aa in range(len(opts.compOptions)):
                for idxk, k in enumerate(opts.nodeIndices):
                    toPlot = data[dirs[jj]][opts.compOptions[aa]][f'N{k + 1}'][opts.metricsToPlot[ii]]
                    
                    lab = f'{dirs[jj]} - N{k + 1} - {opts.compOptions[aa]}'
                    if 'No' in opts.compOptions[aa]:
                        style = f'C{jj}.--'
                    else:
                        style = f'C{jj}.-'
                    axes[ii].plot(sros, toPlot, style, label=lab, alpha=1/(idxk + 1))
        axes[ii].grid()
        axes[ii].set_xlabel('SRO $\\varepsilon_{{kq}}$ [ppm]')
        if opts.metricsToPlot[ii] == 'stoi':
            tistr = 'eSTOI'
        elif opts.metricsToPlot[ii] == 'pesq':
            tistr = 'PESQ'
        elif opts.metricsToPlot[ii] == 'fwSNRseg':
            tistr = 'fwSNRseg'
        if opts.valuesToPlot == 'diff':
            tistr += ' - $\\Delta$ before/after enhancement'
        elif opts.valuesToPlot == 'after':
            tistr += ' - metric after enhancement'
        elif opts.valuesToPlot == 'before':
            tistr += ' - metric before enhancement'
        axes[ii].set_title(tistr)
        if (toPlot < 0).any():
            axes[ii].hlines(y=0, xmin=0, xmax=np.amax(sros), colors='k', linestyles='--')
    plt.tight_layout()
    # axes[0].legend(loc='lower left')
    axes[0].legend(loc='best')

    if not EXPORTFIGURE:
        plt.show()

    return fig


if __name__ == '__main__':
    sys.exit(main())

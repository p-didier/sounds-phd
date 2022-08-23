from dataclasses import dataclass, field
import json
import sys, os
from textwrap import fill
import numpy as np
import matplotlib.pyplot as plt
# import dataclass_wizard as dcw

PATHTORESULTS = '01_algorithms/01_NR/02_distributed/res/testing_SROs/automated/20220822_SROestComp_wFlags/J2Mk[1_1]_Ns1_Nn1'
RESULTSFILENAME = 'Results.json'
SETTINGSFILENAME = 'ProgramSettings.json'

# @dataclass
# class PlottingOptions:
#     titles: dict = dict(stoi='STOI', pesq='PESQ')

@dataclass
class Options:
    metricsToPlot: list[str]    # speech enhancement metrics to be plotted
    valuesToPlot: list[str]     # type of values to be plotted (e.g., "before" for before enhancement, "after", "diff", "diffLocal", etc.)
    compOptions: list[str] = field(default_factory=list)
    singleNode: bool = True

def main():

    opts = Options(
        metricsToPlot=['stoi', 'pesq'],
        # valuesToPlot=['diff'],
        valuesToPlot=['after'],
        compOptions=['Comp.', 'No comp.'],
        singleNode=True
        # Plotting options vvv
        
        )

    sros, data = process_data(opts)

    plot_data(sros, data, opts)

    return None


def process_data(opts: Options):

    # Find subfolders
    dirs = os.listdir(PATHTORESULTS)

    # Find number of nodes
    idx = PATHTORESULTS.rfind('/')
    nNodes = int(PATHTORESULTS[idx+2])
    if opts.singleNode:
        nNodes = 1

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
            res[dir][opts.compOptions[aa]] = dict.fromkeys([f'N{ii + 1}' for ii in range(nNodes)], None)
            for ii in range(nNodes):
                res[dir][opts.compOptions[aa]][f'N{ii + 1}'] = dict.fromkeys(opts.metricsToPlot, None)
                for jj in range(len(opts.metricsToPlot)):
                    res[dir][opts.compOptions[aa]][f'N{ii + 1}'][opts.metricsToPlot[jj]] = \
                        dict.fromkeys(opts.valuesToPlot, np.full(len(sros), fill_value=None))

        for idxSubdir, subdir in enumerate(subdirs):
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

            for ii in range(nNodes):
                for jj in range(len(opts.metricsToPlot)):
                    for kk in range(len(opts.valuesToPlot)):
                        res[dir][currCompOption][f'N{ii + 1}'][opts.metricsToPlot[jj]][opts.valuesToPlot[kk]][idxSrosList] = \
                            d['enhancementEval'][opts.metricsToPlot[jj]][f'Node{ii + 1}'][opts.valuesToPlot[kk]]

                        # Edge case: SRO = 0 PPM -- not computed: simply copy "Comp." value to "No comp." entry
                        if currCompOption == 'No comp.' and currSro == 0:
                            res[dir]['Comp.'][f'N{ii + 1}'][opts.metricsToPlot[jj]][opts.valuesToPlot[kk]][idxSrosList] = \
                                d['enhancementEval'][opts.metricsToPlot[jj]][f'Node{ii + 1}'][opts.valuesToPlot[kk]]


    return sros, res


def plot_data(sros, data: dict, opts: Options):

    nSubplots = len(opts.metricsToPlot)

    dirs = list(data.keys())

    fig, axes = plt.subplots(1, nSubplots)
    fig.set_size_inches(6.5, 2.5)
    for ii in range(nSubplots):
        for jj in range(len(dirs)): 
            for aa in range(len(opts.compOptions)):
                for k in range(len(data[dirs[jj]][opts.compOptions[aa]])):
                    for kk in range(len(opts.valuesToPlot)):
                        toPlot = data[dirs[jj]][opts.compOptions[aa]][f'N{k + 1}'][opts.metricsToPlot[ii]][opts.valuesToPlot[kk]]
                        lab = f'{dirs[jj]} - N{k + 1} - {opts.compOptions[aa]}'
                        if 'No' in opts.compOptions[aa]:
                            style = f'C{jj}.--'
                        else:
                            style = f'C{jj}.-'
                        axes[ii].plot(sros, toPlot, style, label=lab)
        axes[ii].grid()
        axes[ii].legend(loc='lower left')
        axes[ii].set_xlabel('SRO $\\varepsilon_{{kq}}$ [ppm]')
        if opts.metricsToPlot[ii] == 'stoi':
            axes[ii].set_ylim([0, 1])
            axes[ii].set_title(f'eSTOI -- {opts.valuesToPlot[kk]}')
        if opts.metricsToPlot[ii] == 'pesq':
            axes[ii].set_title(f'PESQ -- {opts.valuesToPlot[kk]}')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())


import os
import sys
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path, PurePath

from pyparsing import col
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
sys.path.append(f'{pathToRoot}/_general_fcts')
from danse_utilities.classes import Results, ProgramSettings
import danse_utilities
from plotting.general import lighten_color

@dataclass
class PostProcParams:
    pathToResults : str = ''
    plottype : str = 'group_per_node'     # plot type:
                                # - 'group_per_node': bar chart, grouped per node
                                # - 'group_per_node_vertical': same as previous, but vertical orientation
    savefigure : bool = False   # if True, export figure as PNG and PDF format
    savePath : str = ''         # path to folder where to save file (only used if `savefigure == True`)

# Set post-processing parameters
myParams = PostProcParams(
    pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2',
    # plottype='group_per_node',
    plottype='group_per_node_vertical',
    savefigure=False,
    savePath=Path(__file__).parent.parent
)

def main():
    
    res = run(myParams)

    fig = plot(res, myParams.plottype)

    if myParams.savefigure:
        fig.savefig(f'{myParams.savePath}/myfig.png')
        fig.savefig(f'{myParams.savePath}/myfig.pdf')
    plt.show()
    


def run(params: PostProcParams):

    # Fetch data
    dirs = os.listdir(params.pathToResults)
    # Find number of subdirs
    subdirs = os.listdir(f'{params.pathToResults}/{dirs[0]}/comp')
    nRuns = len(subdirs)

    # Get number of nodes
    nNodes = int(Path(params.pathToResults).stem[1])

    # Results arrays dimensions [type of SROs, comp/nocomp, acoustic scenario, nodes]
    dims = (len(dirs), 2, nRuns, nNodes)
    # Initialize arrays
    stoi = np.zeros(dims)
    stoiLocal = np.zeros(dims)
    stoiOriginal = np.zeros(dims)
    snr = np.zeros(dims)
    snrLocal = np.zeros(dims)
    snrOriginal = np.zeros(dims)
    fwSNRseg = np.zeros(dims)
    fwSNRsegLocal = np.zeros(dims)
    fwSNRsegOriginal = np.zeros(dims)

    # Extract results
    for ii in range(len(dirs)):
        subdirs = os.listdir(f'{params.pathToResults}/{dirs[ii]}')
        for jj in range(len(subdirs)):
            subsubdirs = os.listdir(f'{params.pathToResults}/{dirs[ii]}/{subdirs[jj]}')
            for kk in range(len(subsubdirs)):
                currDirPath = f'{params.pathToResults}/{dirs[ii]}/{subdirs[jj]}/{subsubdirs[kk]}'

                r = Results()
                r = r.load(currDirPath)

                # Get useful values
                for nn in range(nNodes):
                    stoi[ii, jj, kk, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].after
                    stoiLocal[ii, jj, kk, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].afterLocal
                    stoiOriginal[ii, jj, kk, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].before
                    snr[ii, jj, kk, nn] = r.enhancementEval.snr[f'Node{nn+1}'].after
                    snrLocal[ii, jj, kk, nn] = r.enhancementEval.snr[f'Node{nn+1}'].afterLocal
                    snrOriginal[ii, jj, kk, nn] = r.enhancementEval.snr[f'Node{nn+1}'].before
                    fwSNRseg[ii, jj, kk, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].after
                    fwSNRsegLocal[ii, jj, kk, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].afterLocal
                    fwSNRsegOriginal[ii, jj, kk, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].before

    res = dict([
        ('stoi', stoi), ('stoiLocal', stoiLocal), ('stoiOriginal', stoiOriginal),\
        ('snr', snr), ('snrLocal', snrLocal), ('snrOriginal', snrOriginal),\
        ('fwSNRseg', fwSNRseg), ('fwSNRsegLocal', fwSNRsegLocal), ('fwSNRsegOriginal', fwSNRsegOriginal)
    ])

    return res


def plot(res, plottype):

    if plottype == 'group_per_node':
        fig = plot_grouppedpernode(res)
    elif plottype == 'group_per_node_vertical':
        fig = plot_grouppedpernode_vert(res)
    
    return fig


def plot_grouppedpernode(res):

    ylimsSTOI = [0, 1]
    categories = ['$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{l}}$',\
        '$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{m}}$',\
        '$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{s}}$']
    w = 1/4  # width parameter

    # Booleans
    showErrorBars = False

    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(6.5, 4)
    subplot_fcn(axes[0], res['stoi'], res['stoiOriginal'], w, showErrorBars, ylimsSTOI, categories)
    axes[0].set_ylabel('$\Delta$eSTOI')
    subplot_fcn(axes[1], res['fwSNRseg'], res['fwSNRsegOriginal'], w, showErrorBars, None, categories)
    axes[1].set_ylabel('$\Delta$fwSNRseg')
    plt.tight_layout()

    return fig


def plot_grouppedpernode_vert(res):

    ylimsSTOI = [0, 1]
    categories = ['$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{l}}$',\
        '$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{m}}$',\
        '$\\varepsilon\\in\\mathcal{{E}}_\\mathrm{{s}}$']
    w = 1/4  # width parameter

    # Booleans
    showErrorBars = True
    plotSecondMetric = False

    if plotSecondMetric:
        fig, axes = plt.subplots(res['stoi'].shape[0], 2)
        fig.set_size_inches(8, 6.5)
    else:
        fig, axes = plt.subplots(res['stoi'].shape[0], 1)
        fig.set_size_inches(6, 5.5)
    for ii in range(res['stoi'].shape[0]):
        
        if plotSecondMetric:
            subplot_fcn_2(axes[ii, 0], res['stoi'][ii, :, :, :], res['stoiOriginal'][ii, :, :, :],
                w, showErrorBars, ylimsSTOI, f'C{ii}')
            subplot_fcn_2(axes[ii, 1], res['fwSNRseg'][ii, :, :, :], res['fwSNRsegOriginal'][ii, :, :, :],
                w, showErrorBars, [0, 8], f'C{ii}', showLegend=True)
            if ii == 0:
                axes[ii, 0].set_title('eSTOI')
                axes[ii, 1].set_title('fwSNRseg [dB]')
            if ii == res['stoi'].shape[0] - 1:
                axes[ii, 0].set_xlabel('Node index $k$')
                axes[ii, 1].set_xlabel('Node index $k$')
            # SRO domains texts
            yplacement = np.amax(axes[ii, 1].get_ylim()) * 0.85
            axes[ii, 1].text(x=np.amax(axes[ii, 1].get_xlim()) + 1.25, y=yplacement,
                s=categories[ii], bbox=dict(boxstyle='round', facecolor=f'C{ii}', alpha=1), color='w',
                fontsize=12)
        else:
            subplot_fcn_2(axes[ii], res['stoi'][ii, :, :, :], res['stoiOriginal'][ii, :, :, :],
                w, showErrorBars, ylimsSTOI, f'C{ii}', showLegend=True)
            if ii == 0:
                axes[ii].set_title('eSTOI')
            if ii == res['stoi'].shape[0] - 1:
                axes[ii].set_xlabel('Node index $k$')
            # SRO domains texts
            yplacement = np.amax(axes[ii].get_ylim()) * 0.85
            axes[ii].text(x=np.amax(axes[ii].get_xlim()) + 1.15, y=yplacement,
                s=categories[ii], bbox=dict(boxstyle='round', facecolor=f'C{ii}', alpha=1), color='w',
                fontsize=12)
            
    plt.tight_layout()

    return fig


def subplot_fcn_2(ax, res, resBeforeEnhancement, w, showErrorBars, ylims, mycolor, showLegend=False):

    if showErrorBars:
        ecolor = 'k'
    else:
        ecolor = 'none'

    nNodes = res.shape[-1]

    ax.grid()
    ax.set_axisbelow(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    # With compensation
    currStoiData = res[0, :, :]
    handle1 = ax.bar(np.arange(nNodes) + w, np.mean(currStoiData, axis=0),
        width=w, yerr=np.std(currStoiData, axis=0), align='center', alpha=1, ecolor=ecolor, capsize=2,
        color=lighten_color(mycolor, 1), edgecolor='k')
    # Without compensation
    currStoiData = res[1, :, :]
    handle2 = ax.bar(np.arange(nNodes), np.mean(currStoiData, axis=0),
        width=w, yerr=np.std(currStoiData, axis=0), align='center', alpha=1, ecolor=ecolor, capsize=2,
        color=lighten_color(mycolor, 0.66), edgecolor='k')
    # Without enhancement
    currStoiData = resBeforeEnhancement[0, :, :]
    handle3 = ax.bar(np.arange(nNodes) - w, np.mean(currStoiData, axis=0),
        width=w, align='center', alpha=1,
        color=lighten_color(mycolor, 0.33), edgecolor='k')
    ax.set_xticks(np.arange(nNodes))
    xtlabs = np.array([f'{ii+1}' for ii in range(nNodes)])
    ax.set_xticklabels(xtlabs)
    if showLegend:
        ax.legend(
            handles=[handle1, handle2, handle3],
            labels=['DANSE + comp.', 'DANSE', 'Original'],
            bbox_to_anchor=(1.04, 0),
            loc="lower left"
        )


def subplot_fcn(ax, res, resBeforeEnhancement, w, showErrorBars, ylims, categories):

    if showErrorBars:
        ecolor = 'k'
    else:
        ecolor = 'none'

    nNodes = res.shape[-1]

    ax.grid()
    ax.set_axisbelow(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    for jj in range(res.shape[0]):
        # With compensation
        currStoiData = res[jj, 0, :, :]
        ax.bar(np.arange(nNodes) + jj*nNodes + w, np.mean(currStoiData, axis=0),
            width=w, yerr=np.std(currStoiData, axis=0), align='center', alpha=1, ecolor=ecolor, capsize=2,
            color=lighten_color(f'C{jj}', 1), edgecolor='k')
        # Without compensation
        currStoiData = res[jj, 1, :, :]
        ax.bar(np.arange(nNodes) + jj*nNodes, np.mean(currStoiData, axis=0),
            width=w, yerr=np.std(currStoiData, axis=0), align='center', alpha=1, ecolor=ecolor, capsize=2,
            color=lighten_color(f'C{jj}', 0.66), edgecolor='k')
        # Without enhancement
        currStoiData = resBeforeEnhancement[jj, 0, :, :]
        ax.bar(np.arange(nNodes) + jj*nNodes - w, np.mean(currStoiData, axis=0),
            width=w, yerr=np.std(currStoiData, axis=0), align='center', alpha=1, ecolor=ecolor, capsize=2,
            color=lighten_color(f'C{jj}', 0.33), edgecolor='k')
    ax.vlines(x=(np.arange(res.shape[0]-1)+1)*nNodes-0.5, ymin=np.amin(ax.get_ylim()), ymax=np.amax(ax.get_ylim()), colors='k', linestyles=':')
    ax.set_xticks(np.arange(res.shape[0]*nNodes))
    xtlabs = np.array([f'{ii+1}' for ii in range(nNodes)] * res.shape[0])
    ax.set_xticklabels(xtlabs)
    # Add useful texts
    ax.text(x=np.amax(ax.get_xlim()),
        y=np.amin(ax.get_ylim()) - 0.2 * (np.amax(ax.get_ylim()) - np.amin(ax.get_ylim())),
        s='Node #')
    # SRO domains
    yplacement = np.amax(ax.get_ylim()) * 0.88
    for ii in range(res.shape[0]):
        ax.text(x=ii*nNodes-w, y=yplacement, s=categories[ii], bbox=dict(boxstyle='round', facecolor=f'C{ii}', alpha=1), color='w')

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
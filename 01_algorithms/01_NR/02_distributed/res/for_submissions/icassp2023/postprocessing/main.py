
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path, PurePath

# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
sys.path.append(f'{pathToRoot}/_general_fcts')
from danse_utilities.classes import Results
from plotting.general import lighten_color

@dataclass
class PostProcParams:
    pathToResults : str = ''
    plottype : str = 'group_per_node'     # plot type:
                                # - 'group_per_node': bar chart, grouped per node
                                # - 'group_per_node_vertical': same as previous, but vertical orientation
    savefigure : bool = False   # if True, export figure as PNG and PDF format
    savePath : str = ''         # path to folder where to save file (only used if `savefigure == True`)
    includeCentralisedPerf : bool = False   # if True, include the centralised performance in the graph
    includeLocalPerf : bool = False   # if True, include the local performance in the graph
    firstMetric: str = 'eSTOI'       # first metric to plot ('eSTOI', 'SNR', or 'fwSNRseg')
    secondMetric: str = 'fwSNRseg'  # second metric to plot ('eSTOI', 'SNR', or 'fwSNRseg')

# Set post-processing parameters
p = PostProcParams(
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_diffwn',  # w/ diffuse white noise
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_wn',  # w/ white noise
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2',  # ?
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_ssn',   # w/ SSN
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_diffbab',   # w/ diffuse babble noise
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_bab',   # w/ babble noise
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_10cmspacing',
    # pathToResults=f'{Path(__file__).parent.parent}/J4Mk[1_3_2_5]_Ns1_Nn2_SpS',
    # pathToResults=f'{Path(__file__).parent.parent}/_archive/J4Mk[1_3_2_5]_Ns1_Nn2__week40_AS2',  # ?
    #
    # vvv test on 17.01.2023 (see journal 2023 week03) vvv
    # pathToResults=f'{Path(__file__).parent.parent}/test_postReviews/test_FSDs_20230117/J4_withExtraPSF',  # with extra phase shift factor (same as done for ICASSP paper)
    pathToResults=f'{Path(__file__).parent.parent}/test_postReviews/test_FSDs_20230117/J4_withoutExtraPSF',  # withOUT extra phase shift factor (same as done for ICASSP paper)
    # ^^^ test on 17.01.2023 (see journal 2023 week03) ^^^
    #
    # plottype='group_per_node',
    plottype='group_per_node_vertical',
    savePath=Path(__file__).parent.parent,
    includeCentralisedPerf=True,
    includeLocalPerf=True,
    firstMetric='eSTOI',
    # firstMetric='SNR',
    # firstMetric='fwSNRseg',
    secondMetric='fwSNRseg',
    #
    savefigure=False,
)

def main():
    
    res = run(p)

    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 12})

    fig = plot(res, p)

    if p.savefigure:
        fname = f'myfig__{Path(p.pathToResults).stem}'
        if p.includeLocalPerf:
            fname += '_incLocal'
        fig.savefig(f'{p.savePath}/{fname}.png')
        fig.savefig(f'{p.savePath}/{fname}.pdf')
    plt.show()


def run(params: PostProcParams):

    # Fetch data
    dirs = os.listdir(params.pathToResults)
    if params.includeCentralisedPerf:
        # Check whether there is a centralised estimate folder
        if 'centralised' not in dirs:
            print('Not centralised data found. Not plotting it.')
            centralisedDir = None
        else:
            centralisedDir = f'{params.pathToResults}/centralised'
    dirs = [ii for ii in dirs if ii != 'centralised']  # do not include the centralised estimation yet
    # Invert order of directories to process SROs from small to large
    dirs = np.flip(dirs)

    # Get number of nodes
    nNodes = int(Path(params.pathToResults).stem[1])

    # Results arrays dimensions [type of SROs, comp/nocomp, acoustic scenario, nodes]
    dims = (len(dirs), 2, nNodes)
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

            if 'nocomp' in subdirs[jj]:
                idx = 0
            else:
                idx = 1
            
            currDirPath = f'{params.pathToResults}/{dirs[ii]}/{subdirs[jj]}'

            r = Results()
            r = r.load(currDirPath)

            # Get useful values
            for nn in range(nNodes):
                stoi[ii, idx, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].after
                stoiLocal[ii, idx, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].afterLocal
                stoiOriginal[ii, idx, nn] = r.enhancementEval.stoi[f'Node{nn+1}'].before
                snr[ii, idx, nn] = r.enhancementEval.snr[f'Node{nn+1}'].after
                snrLocal[ii, idx, nn] = r.enhancementEval.snr[f'Node{nn+1}'].afterLocal
                snrOriginal[ii, idx, nn] = r.enhancementEval.snr[f'Node{nn+1}'].before
                fwSNRseg[ii, idx, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].after
                fwSNRsegLocal[ii, idx, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].afterLocal
                fwSNRsegOriginal[ii, idx, nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].before

    # Get centralised data (if asked)
    stoiCentr = np.full(nNodes, fill_value=None)
    snrCentr = np.full(nNodes, fill_value=None)
    fwSNRsegCentr = np.full(nNodes, fill_value=None)
    if params.includeCentralisedPerf:
        if centralisedDir is not None:
            r = Results()
            r = r.load(centralisedDir)
            for nn in range(nNodes):
                stoiCentr[nn] = r.enhancementEval.stoi[f'Node{nn+1}'].after
                snrCentr[nn] = r.enhancementEval.snr[f'Node{nn+1}'].after
                fwSNRsegCentr[nn] = r.enhancementEval.fwSNRseg[f'Node{nn+1}'].after

    res = dict([
        ('eSTOI', stoi), ('eSTOILocal', stoiLocal), ('eSTOIOriginal', stoiOriginal), ('eSTOICentr', stoiCentr),\
        ('SNR', snr), ('SNRLocal', snrLocal), ('SNROriginal', snrOriginal), ('SNRCentr', snrCentr),\
        ('fwSNRseg', fwSNRseg), ('fwSNRsegLocal', fwSNRsegLocal),\
            ('fwSNRsegOriginal', fwSNRsegOriginal), ('fwSNRsegCentr', fwSNRsegCentr)
    ])

    return res


def plot(res, p: PostProcParams):

    if p.plottype == 'group_per_node':
        fig = plot_grouppedpernode(res, p.firstMetric, p.secondMetric)
    elif p.plottype == 'group_per_node_vertical':
        fig = plot_grouppedpernode_vert(
            res, p.firstMetric, p.secondMetric, p.includeLocalPerf
        )
    
    return fig


def plot_grouppedpernode(res, metric1, metric2):
    # TODO -- add option to show centralised performance
    # vvv HARD-CODED but ok
    categories = [
        '$40\\geq|\\varepsilon|\\geq 20$ PPM',\
        '$100\\geq|\\varepsilon|\\geq 50$ PPM',\
        '$400\\geq|\\varepsilon|\\geq 200$ PPM'
        ]
    w = 1/4  # width parameter

    # Booleans
    showErrorBars = False

    ylims1 = None
    ylims2 = None
    if metric1 == 'eSTOI':
        ylims1 = [0,1]
    if metric2 == 'eSTOI':
        ylims2 = [0,1]
        
    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(6.5, 4)
    subplot_fcn(axes[0], res[metric1], res[f'{metric1}Original'], w, showErrorBars, ylims1, categories)
    axes[0].set_ylabel(metric1)
    subplot_fcn(axes[1], res[metric2], res[f'{metric2}Original'], w, showErrorBars, ylims2, categories)
    axes[1].set_ylabel(metric2)
    plt.tight_layout()

    return fig


def plot_grouppedpernode_vert(res, metric1, metric2, includeLocal=False):

    # vvv HARD-CODED but ok
    categories = [
        '$40\\geq|\\varepsilon|\\geq 20$ PPM',\
        '$100\\geq|\\varepsilon|\\geq 50$ PPM',\
        '$400\\geq|\\varepsilon|\\geq 200$ PPM'
        ]
    # colors = ['C4', 'C2', 'C1']
    colors = [(19/255, 103/255, 159/255), (192/255, 0, 0), (0, 0, 0)]
    w = 1/4  # width parameter
    if includeLocal:
        w = 1/5

    # Booleans
    plotSecondMetric = False
    
    ylims1 = None
    if metric1 == 'eSTOI':
        ylims1 = [0,1]

    if plotSecondMetric:
        fig, axes = plt.subplots(res[metric1].shape[0], 2)
        fig.set_size_inches(8, 6.5)
    else:
        fig, axes = plt.subplots(res[metric1].shape[0], 1)
        fig.set_size_inches(7, 6.5)
    
    for ii in range(res[metric1].shape[0]):
        
        if plotSecondMetric:
            subplot_fcn_2(axes[ii, 0], res[metric1][ii, :, :], res[f'{metric1}Original'][ii, :, :],
                w, ylims1, colors[ii])
            subplot_fcn_2(axes[ii, 1], res[metric2][ii, :, :], res[f'{metric2}Original'][ii, :, :],
                w, [0, 8], colors[ii], showLegend=True)
            if ii == 0:
                axes[ii, 0].set_title(metric1)
                axes[ii, 1].set_title(metric2)
            if ii == res[metric1].shape[0] - 1:
                axes[ii, 0].set_xlabel('Node index $k$')
                axes[ii, 1].set_xlabel('Node index $k$')
            # SRO domains texts
            yplacement = np.amax(axes[ii, 1].get_ylim()) * 0.85
            axes[ii, 1].text(x=np.amax(axes[ii, 1].get_xlim()) + .25, y=yplacement,
                s=categories[ii], bbox=dict(boxstyle='round', facecolor=colors[ii], alpha=1), color='w',
                fontsize=12)
            # Centralised performance
            if res[f'{metric1}Centr'] is not None:
                # Plot that too
                axes[ii, 0].hlines(y=res[f'{metric1}Centr'], xmin=np.amin(axes[ii].get_xlim()), xmax=np.amax(axes[ii].get_xlim()),
                    colors='k', linestyles='--')
                axes[ii, 1].hlines(y=res[f'{metric2}Centr'], xmin=np.amin(axes[ii].get_xlim()), xmax=np.amax(axes[ii].get_xlim()),
                    colors='k', linestyles='--')
        else:
            localData = None
            if includeLocal:
                localData = res[f'{metric1}Local'][ii, :, :]
            subplot_fcn_2(
                axes[ii],
                res[metric1][ii, :, :],
                res[f'{metric1}Original'][ii, :, :],
                localData,
                w,
                ylims1,
                colors[ii],
                showLegend=True,
                centralised=res[f'{metric1}Centr']
            )
            if ii == 0:
                axes[ii].set_title(metric1)
            if ii == res[metric1].shape[0] - 1:
                axes[ii].set_xlabel('Node index $k$')
            # SRO domains texts
            yplacement = np.amax(axes[ii].get_ylim()) * 0.9
            axes[ii].text(x=np.amax(axes[ii].get_xlim()) + .25, y=yplacement,
                s=categories[ii], bbox=dict(boxstyle='round', facecolor=colors[ii], alpha=1), color='w',
                fontsize=12)
    plt.tight_layout()

    return fig


def subplot_fcn_2(
    ax,
    res,
    resBeforeEnhancement,
    resLocal,
    w,
    ylims,
    mycolor,
    showLegend=False,
    centralised=None
    ):
    """Helper subplot function."""

    nNodes = res.shape[-1]

    # ax.grid()
    ax.set_axisbelow(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    if resLocal is not None:
        # With compensation
        handle1 = ax.bar(np.arange(nNodes) + 1.5*w, res[1, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 1), edgecolor='k')
        # Without compensation
        handle2 = ax.bar(np.arange(nNodes) + 0.5*w, res[0, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 0.75), edgecolor='k')
        # Local estimate
        handle3 = ax.bar(np.arange(nNodes) - 0.5*w, resLocal[0, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 0.5), edgecolor='k')
        # Without enhancement
        handle4 = ax.bar(np.arange(nNodes) - 1.5*w, resBeforeEnhancement[0, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 0.25), edgecolor='k')
        handles = [handle1, handle2, handle3, handle4]   # handles for legend
        leglabs = [
            'GEVD-DANSE + SRO comp.',
            'GEVD-DANSE',
            'Local GEVD-MWF',
            'Noisy sensor signal $\dot{y}_{{k,1}}$'
        ]    # labels for legend
    else:
        # With compensation
        handle1 = ax.bar(np.arange(nNodes) + w, res[1, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 1), edgecolor='k')
        # Without compensation
        handle2 = ax.bar(np.arange(nNodes), res[0, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 0.75), edgecolor='k')
        # Without enhancement
        handle4 = ax.bar(np.arange(nNodes) - w, resBeforeEnhancement[0, :],
            width=w, align='center', alpha=1,
            color=lighten_color(mycolor, 0.25), edgecolor='k')
        handles = [handle1, handle2, handle4]   # handles for legend
        leglabs = [
            'GEVD-DANSE + SRO comp.',
            'GEVD-DANSE',
            'Noisy sensor signal $\dot{y}_{{k,1}}$'
        ]    # labels for legend

    # Add grey vertical lines
    ax.vlines(
        x=np.arange(nNodes-1) + 0.5,
        ymin=np.amin(ax.get_ylim()),
        ymax=np.amax(ax.get_ylim()),
        colors='tab:gray',
        linestyles='-'
    )
    # Centralised
    if centralised is not None:
        for k in range(nNodes):
            xlims = [k-w*2, k+w*2]
            if resLocal is not None:
                xlims = [k-w*2.5, k+w*2.5]
            tmp = ax.hlines(
                y=centralised[k],
                xmin=xlims[0],
                xmax=xlims[1],
                colors='k',
                linestyles='--'
            )
        handles.append(tmp)
        leglabs.append('GEVD-MWF (no SRO)')
    ax.set_xticks(np.arange(nNodes))
    xtlabs = np.array([f'{ii+1}' for ii in range(nNodes)])
    ax.set_xticklabels(xtlabs)
    ax.set_xlim([-0.5, nNodes-0.5])  
    if showLegend:
        bboxtoa = (1, 0)
        if resLocal is not None:
            bboxtoa = (1, -.15)
        ax.legend(
            handles=handles,
            labels=leglabs,
            bbox_to_anchor=bboxtoa,
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
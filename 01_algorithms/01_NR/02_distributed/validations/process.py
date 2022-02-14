#%%
import numpy as np
import sys
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_utilities import classes
sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes')
import utilsASC

#%%


def main():
    # Path to results folder
    pathToResults = 'danse_res'

    # List all acoustic scenarios folder paths
    acsSubDirs = list(Path(pathToResults).iterdir())

    # Load and arrange results
    objs = []
    allRTs = []
    allNNodes = []
    for ii, dirr in enumerate(acsSubDirs):
        obj = classes.Results().load(dirr)
        # Reverberation time
        if 'RT' in dirr.name:
            idxStart = dirr.name.find('RT')+2
            idxEnd = dirr.name.find('ms')
            rt = float(dirr.name[idxStart:idxEnd]) / 1e3    # [s]
        else:   # anechoic case
            rt = np.Inf
        if rt not in allRTs:
            allRTs.append(rt)
        # Number of nodes
        nNodes = int(dirr.name[1])
        if nNodes not in allNNodes:
            allNNodes.append(nNodes)
        objs.append(dict([('RT', rt), ('J', nNodes), ('metrics', obj.enhancementEval)]))

    # Plots
    for ii, rt in enumerate(allRTs):
        for jj, n in enumerate(allNNodes):
            objscurr = [o in objs if o['RT'] == rt and o['J'] == n]
            label = f'$J$ = {n} nodes, $T_{{60}}$ = {np.round(rt, 2)} s'
            single_plot(objscurr['metrics'], label)


def single_plot(objs, label):
    """Plotting subfunction
    
    Parameters
    ----------
    objs : [N x 1] list of classes.EnhancementMeasures objects
        Enhancement metrics outcomes over N similar acoustic scenarios.
    label : str
        Suptitle of the figure.
    """
    # Compute distribution of results across repetitions
    dsnrs = np.zeros(len(objs))
    stois = np.zeros(len(objs))
    fwsnrsegs = np.zeros(len(objs))
    for ii in range(len(objs)):
        dsnrs[ii] = np.mean(objs[ii].snr)
        stois[ii] = np.mean(objs[ii].stoi)
        fwsnrsegs[ii] = np.mean(objs[ii].fwSNRseg)

    # Plot
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(131)
    plt.hist(dsnrs, bins=np.arange(np.amin(dsnrs), np.amax(dsnrs)+1, step=1), density=True)
    ax.set(xlabel='$E_{k\in[1..J]}\{\Delta\mathrm{SNR}_k\}$ [dB]',
            ylabel='Probability density (normalized counts)')
    ax.set_ylim((0, 1))
    ax = fig.add_subplot(132)
    plt.hist(fwsnrsegs, bins=np.arange(np.amin(fwsnrsegs), np.amax(fwsnrsegs)+1, step=1), density=True)
    ax.set(xlabel='$E_{k\in[1..J]}\{\mathrm{fwSNRseg}_k\}$ [dB]')
    ax.set_ylim((0, 1))
    ax = fig.add_subplot(133)
    plt.hist(stois, bins=np.arange(np.amin(stois), np.amax(stois)+1, step=1), density=True)
    ax.set(xlabel='$E_{k\in[1..J]}\{\mathrm{STOI}_k\}$',
            ylabel='Probability density (normalized counts)')
    ax.set_ylim((0, 1))
    fig.suptitle(label)

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
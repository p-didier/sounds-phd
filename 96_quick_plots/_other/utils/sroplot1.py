#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.style.use('default')  # <-- for Jupyter: white figures background


def plotsro(exportpath=None, style='rb', showNumSamples=True):
    """
    plotsro -- Plots a visualization of the effect of SROs.
    Analog underlying signal vs. sampling by node `k` vs. sampling by node `q`.

    Parameters
    ----------
    exportpath : str
        Full path to figure file to export. If None, do not export figure.
    style : str
        Figure style. 
            If 'rb': red/blue dots style. Ref: SOUNDSSC meeting #4 slide deck, s.11.
            If 'bw': black & white style.
    showNumSamples : bool
        If True, add two bottom lines explicitly showing the number of samples
        captures by node `k` and `q`, respectively. 
    """
    n = 15  # number of sample points for reference node (`k`)
    n2 = n + 2  # number of sample points for SRO-affected node (`q`)
    x = np.arange(n)
    y = np.linspace(start=0, stop=n-1, num=n2)

    f = 0.75/5  # sinusoid frequency
    t = np.linspace(start=0, stop=n-1, num=1000)   # "analog" time axis
    u = 1.5 * np.sin(2 * np.pi * f * t)  # "analog" signal

    # Create sample indices
    idx_x = np.zeros(len(x), dtype=int)
    for ii in range(len(x)):
        idx_x[ii] = np.argmin(np.abs(t - x[ii]))
    idx_y = np.zeros(len(y), dtype=int)
    for ii in range(len(y)):
        idx_y[ii] = np.argmin(np.abs(t - y[ii]))

    if style == 'rb':
        col1 = 'b'
        col2 = 'r'
    elif style == 'bw':
        col1 = 'k'
        col2 = 'r'

    if showNumSamples:
        figHeight = 2
    else:
        figHeight = 1

    delta = 4
    xmaxCut = int(n/2)+2
    x = x[x <= xmaxCut]
    y = y[y <= xmaxCut]
    u = u[t <= np.amax(x)]
    t = t[t <= np.amax(x)]
    idx_x = idx_x[idx_x <= len(u)]
    idx_y = idx_y[idx_y <= len(u)]

    fig = plt.figure(figsize=(6,figHeight))
    ax = fig.add_subplot(111)
    ax.plot(x, np.full_like(x, fill_value=1), f'{col1}o')
    ax.plot(y, np.full_like(y, -1), f'{col2}o')
    # ax.plot(t, u + 3, 'tab:gray')
    ax.plot(t, u + delta, 'k')
    ax.plot(x, u[idx_x] + delta, f'{col1}.')
    ax.plot(y, u[idx_y] + delta, f'{col2}.')
    ax.vlines(x, ymin=np.full_like(x, 1), ymax=u[idx_x] + delta, colors=col1, linestyles=':')
    ax.vlines(y, ymin=np.full_like(y, -1), ymax=u[idx_y] + delta, colors=col2, linestyles=':')
    ax.hlines([-1,1], xmin=-0.5, xmax=int(n), colors='tab:gray')
    yticks = [-1, 1, delta]
    labsyticks = ['Node $q$', 'Node $k$', 'Analog signal']
    if showNumSamples:  # show number of samples at each node
        previousValue = 0
        zo = 999
        for ii in range(len(x)):
            ax.scatter(x[ii], -delta, s=125, facecolors='w', edgecolors='w', zorder=zo)
            ax.text(x[ii], -delta, f'{np.sum(y <= x[ii])}', ha='center', va='center', color=col2, zorder=zo+1)
            if np.sum(y <= x[ii]) > previousValue + 1:
                h = ax.scatter(x[ii], -delta, s=200, facecolors='none', edgecolors=col2, zorder=zo+2)
            previousValue = np.sum(y <= x[ii])
        yticks = [-4] + yticks
        labsyticks = ['# samples\nat node $q$'] + labsyticks
    plt.yticks(yticks, labels=labsyticks)  # y-axis ticks and labels
    if showNumSamples:
        plt.ylim((-delta-2, 6))
        # Add vertical lines
        ax.vlines(x, ymin=np.full_like(x, ax.get_ylim()[0]), ymax=np.full_like(x, 1), colors='tab:gray', zorder=0)
    else:
        plt.ylim((-2, 5))
    plt.xticks(x, labels=x+1)
    plt.xlim((-0.5, xmaxCut+0.5))
    plt.xlabel('Sample index at node $k$')
    # plt.legend(handles=[h], labels=['Full-sample drift'], loc='upper right')
    plt.tight_layout()

    if exportpath is not None:
        # Check that path exists -- if not, create it
        if not Path.exists(Path(exportpath).parent):
            print(f'Export folder\n"{Path(exportpath).parent}"\ndoes not exist -- creating it now.')
            Path.mkdir(Path(exportpath).parent)

        if exportpath[-4:] not in ['.pdf', '.png']:
            # Export in both formats
            fig.savefig(exportpath + '.pdf')
            fig.savefig(exportpath + '.png')
        else:
            fig.savefig(exportpath)
    
    plt.show()


def plotsto(exportpath):
    n = 30
    d = 0.2
    x = np.arange(n)
    y = np.linspace(start=0, stop=n-1, num=n) + d

    f = 0.75/5
    t = np.linspace(start=0, stop=n-1, num=1000)
    u = np.sin(2 * np.pi * f * t)

    idx_x = np.zeros(len(x), dtype=int)
    for ii in range(len(x)):
        idx_x[ii] = np.argmin(np.abs(t - x[ii]))
    idx_y = np.zeros(len(y), dtype=int)
    for ii in range(len(y)):
        idx_y[ii] = np.argmin(np.abs(t - y[ii]))

    fig = plt.figure(figsize=(8,1))
    ax = fig.add_subplot(111)
    plt.plot(x, np.full_like(x, fill_value=1), 'b.:')
    plt.plot(y, np.full_like(y, -1), 'r.:')
    plt.plot(t, u + 3, 'tab:gray')
    plt.plot(x, u[idx_x] + 3, 'b.')
    plt.plot(y, u[idx_y] + 3, 'r.')
    plt.grid(axis='both')
    plt.yticks([-1, 1, 3], labels=['Node $q$', 'Node $k$', 'True signal'])
    plt.ylim((-2, 5))
    plt.xticks(x, labels=[])
    plt.xlim((-0.5, int(n/2)))
    plt.xlabel('Time')
    fig.savefig(exportpath)
    plt.show()

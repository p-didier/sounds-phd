
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import sys, os, re
from scipy.io import wavfile
import scipy.signal as sig
import matplotlib as mpl
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')

# Set colormap
# mpl.rc('image', cmap='magma')
mpl.rc('image', cmap='viridis')


pathToResults = f'{pathToRoot}/01_algorithms/01_NR/02_distributed/res/testing_SROs/automated/with_compression/freq_domain_compression/J2Mk[1, 1]_Ns1_Nn1'

plottingChoices = dict([
    ('noisy', True),
    ('noSROs', True),
    ('withSROs', True),
    ('oracleCompensated', True)
])
sroToPlot = 100     # [ppm]
nodeIdxToPlot = 0   # index of node to plot


def main():

    # Get number of subplots
    vals = plottingChoices.values()
    nSubplots = 0
    keysToPlot = np.full((len(plottingChoices),), fill_value=False)
    for ii, v in enumerate(vals):
        if v:
            nSubplots += 1
            keysToPlot[ii] = True
    clims = [np.Inf, -np.Inf]
    data = []
    for ii in range(len(plottingChoices)):

        if keysToPlot[ii]:
            dataType = list(plottingChoices.keys())[ii]
            x, fs = get_data(dataType, pathToResults, sroToPlot, nodeIdxToPlot)

            # Compute STFT
            f, t, xSTFT = sig.stft(x, fs=fs, window='hann', nperseg=1024, noverlap=512)
            f /= 1e3    # make kHz
            # Set colorbar limits
            if clims[0] > np.amin(todb(xSTFT[xSTFT != 0])):
                clims[0] = np.amin(todb(xSTFT[xSTFT != 0]))
            if clims[1] < np.amax(todb(xSTFT[xSTFT != 0])):
                clims[1] = np.amax(todb(xSTFT[xSTFT != 0]))

            data.append(xSTFT)

    # Adapt clims for better contrast
    climsCompressionFactor = 0.25
    climsnewLow = clims[0] + (clims[1] - clims[0]) * climsCompressionFactor
    climsnewHigh = clims[1] - (clims[1] - clims[0]) * climsCompressionFactor / 2
    clims = [climsnewLow, climsnewHigh]
            

    plotSize = 1.5

    if nSubplots <= 3:
        fig = plt.figure(figsize=(nSubplots * 3 * plotSize, 3 * plotSize))
    else:
        fig = plt.figure(figsize=(nSubplots * 1.5 * plotSize, 4 * plotSize))

    for ii in range(len(plottingChoices)):

        if nSubplots <= 3:
            ax = fig.add_subplot(100 + nSubplots * 10 + ii + 1) 
        else:
            ax = fig.add_subplot(200 + int(np.ceil(nSubplots / 2)) * 10 + ii + 1) 

        
        if keysToPlot[ii]:
            dataType = list(plottingChoices.keys())[ii]

            mapp = ax.imshow(todb(data[ii]), extent=[t[0], t[-1], f[-1], f[0]], vmin=clims[0], vmax=clims[1])
            # ax.grid()
            ax.invert_yaxis()
            ax.set_aspect('auto')
            tistr = dataType
            if dataType == 'withSROs':
                tistr += f' ({sroToPlot} ppm)'
            ax.set_title(tistr)
            ax.set_xlabel('$t$ [s]')
            ax.set_ylabel('$f$ [kHz]')
            plt.colorbar(mapp)
    plt.tight_layout()	
    # Save figure
    fig.savefig(f'{pathToResults}/STFTs.png')
    fig.savefig(f'{pathToResults}/STFTs.pdf')
    plt.show()

    return None


def todb(x):
    """Compute dB value of x"""
    return 20 * np.log10(np.abs(x))


def get_data(dataType, path, sroToPlot, nodeIdxToPlot):
    """Finds correct WAV file in results folder based on user inputs."""

    subdirs = [x[0] for x in os.walk(path)]     # list subdirs

    if dataType in ['noSROs', 'withSROs', 'noisy']:
        if 'no_compensation' in [Path(x).name for x in subdirs]:    
            subdir = f'{path}/no_compensation'
            subdirss = [x[0] for x in os.walk(subdir)]  # list subsubdirs
            datadir = subdirss[1]
            for s in subdirss:
                integersInString = list(map(int, re.findall(r'\d+', Path(s).name[3:])))
                if dataType in ['noSROs', 'noisy']:
                    if len(integersInString) > 0 and (np.array(integersInString) == 0).all():     # find directory containing no-SRO data
                        datadir = s
                elif dataType == 'withSROs':
                    if len(integersInString) > 0 and sroToPlot in np.array(integersInString):     # find directory containing desired SRO
                        datadir = s

    elif dataType == 'oracleCompensated':
        if 'oracle_compensation' in [Path(x).name for x in subdirs]:    
            subdir = f'{path}/oracle_compensation'
            subdirss = [x[0] for x in os.walk(subdir)]  # list subsubdirs
            datadir = subdirss[1]
            for s in subdirss:
                integersInString = list(map(int, re.findall(r'\d+', Path(s).name[3:])))
                if len(integersInString) > 0 and sroToPlot in np.array(integersInString):     # find directory containing desired SRO
                    datadir = s

    if dataType == 'noisy':
        pathToData = f'{datadir}/wav/noisy_N{nodeIdxToPlot+1}_Sref1.wav'
    else:
        pathToData = f'{datadir}/wav/enhanced_N{nodeIdxToPlot+1}.wav'
    fs, data = wavfile.read(pathToData)

    return data, fs


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
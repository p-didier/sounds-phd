
from pathlib import Path, PurePath
import sys, os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
if not any("_general_fcts" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/_general_fcts')
from metrics import eval_enhancement

# Single-sensor nodes DWACD compensation results folder [May 2022]
# targetFolder = f'{Path(__file__).parent.parent}/automated/20220518_SROestimation_withDWACD/J2Mk[1_1]_Ns1_Nn1/anechoic'
# Single-sensor nodes no compensation results folder [May 2022]
targetFolder = f'{Path(__file__).parent.parent}/automated/20220502_impactOfReverb/J2Mk[1_1]_Ns1_Nn1/anechoic_nocomp'
VISUALIZE = True    # if True, plot results in figure


def main():
    
    resSubDirs = list(Path(targetFolder).iterdir())
    resSubDirs = [f for f in resSubDirs if f.name[-1] == ']']   # only keep relevant folders

    # Infer number of nodes in network from the data
    listt = os.listdir(resSubDirs[0] / 'wav')
    nNodes = int(len([i for i in listt if i[-3:] == 'wav' and i[:3] == 'des']))

    # Store STOI
    stoi = np.zeros((len(resSubDirs), nNodes))
    sro = np.zeros(len(resSubDirs))

    for ii in range(len(resSubDirs)):
        print(f'Folder {ii+1}/{len(resSubDirs)}...')
        currDir = resSubDirs[ii] / 'wav'

        # Read SRO from folder name
        idxComma = str(currDir).find(',')
        sro[ii] = int(str(currDir)[idxComma+2:-5])

        for k in range(nNodes):
            
            cleanSignal, fs = sf.read(f'{currDir}/desired_N{k+1}_Sref1.wav')
            enhancedSignal, _ = sf.read(f'{currDir}/enhanced_N{k+1}.wav')
            
            # stoi[ii, k] = eval_enhancement.stoi_fcn(cleanSignal, enhancedSignal, fs, extended=True)
            stoi[ii, k] = eval_enhancement.stoi_fcn(cleanSignal, enhancedSignal, fs, extended=False)

    # Sort
    order = np.argsort(sro)
    sro = sro[order]
    stoi = stoi[order, :]

    if VISUALIZE:
        plotmetrics(sro, stoi)
        stop = 1


def plotmetrics(sro, stoi: np.ndarray):
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    for ii in range(stoi.shape[-1]):
        ax.plot(sro, stoi[:, ii], f'C{ii}.-', label=f'Node {ii+1}')
    ax.grid()
    ax.set_ylim([0,1])
    ax.set_ylabel('STOI')
    ax.set_xlabel('SRO [ppm]')
    plt.tight_layout()
    plt.show()


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------

import pickle, gzip
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# INPUTS
datafolder = Path('U:/py/sounds-phd/01_algorithms/01_NR/02_distributed/dev/quick_exports/20220316_crossCorrMatx_SROeffect')
asc = 'J3Mk[2, 3, 1]'
asc = 'J5Mk[1, 1, 1, 1, 1]'
exportPath = 'U:/py/sounds-phd/01_algorithms/01_NR/02_distributed/dev/quick_tests_scripts/20220316_crossCorrMatx_SROeffect/output'
plotFrequencyIdx = 20


def saveitup(fig, path, name):
    if path[-1] != '/':
        path += '/'

    if not Path(path).is_dir():
        Path(path).mkdir(parents=True)
        print(f'Created directory\n"{path}"\nfor figures export')

    fig.savefig(path + name + '.png')
    fig.savefig(path + name + '.pdf')
    return 0


def main():

    # Find sub-directories
    fullpath = Path.joinpath(datafolder, asc)
    subdirs = [f for f in fullpath.iterdir() if f.is_dir()]
    
    # Derive number of nodes
    files = [f for f in subdirs[0].iterdir() if f.is_file()]
    nNodes = np.amax([int(f.name[7]) for f in files])
    # Derive number of iterations
    allIters = np.unique([int(f.name[10:-7]) for f in files])
    nIters = len(allIters)

    # Init
    bigdictRyy = dict.fromkeys([i.name for i in subdirs], 0)
    bigdictRnn = dict.fromkeys([i.name for i in subdirs], 0)

    for ii in range(len(subdirs)):
        
        files = [f for f in subdirs[ii].iterdir() if f.is_file()]

        # Init giant matrices
        Ryys = []
        Rnns = []

        # Loop over nodes
        for k in range(nNodes):
            print(f'Loading Ryy and Rnn for node {k+1}/{nNodes} in folder "{subdirs[ii].name}"...')

            # Select node-specific files
            filesNodek = [f for f in files if int(f.name[7]) == k + 1]

            # Init current subarrays
            Rshape = pickle.load(gzip.open(str(filesNodek[0]), 'r')).shape
            Ryys_k = np.zeros((Rshape[0], Rshape[1], Rshape[2], nIters), dtype=complex)
            Rnns_k = np.zeros((Rshape[0], Rshape[1], Rshape[2], nIters), dtype=complex)

            for jj in range(len(filesNodek)):

                # Interpret file name
                iternum = int(filesNodek[jj].name[10:-7])
                idxIter = np.where(allIters == iternum)[0][0]

                # Load data file
                p = pickle.load(gzip.open(str(filesNodek[jj]), 'r'))

                if 'Ryy' in filesNodek[jj].name:
                    Ryys_k[:,:,:,idxIter] = p

                elif 'Rnn' in filesNodek[jj].name:
                    Rnns_k[:,:,:,idxIter] = p

            # Build bigger arrays
            Ryys.append(Ryys_k)
            Rnns.append(Rnns_k)

        # Fill in dictionaries
        bigdictRyy[subdirs[ii].name] = Ryys
        bigdictRnn[subdirs[ii].name] = Rnns

    print('All Ryy and Rnn loaded.')

    # Checks
    if plotFrequencyIdx >= bigdictRyy[subdirs[0].name][0].shape[0]:
        raise ValueError('Inexistent frequency index')

    # Generate plot
    for ii in range(len(subdirs)):

        for k in range(nNodes):

            # Ryy
            currDictEntry = bigdictRyy[subdirs[ii].name][k]     # SRO-conditions- and node-specific
            currResRyy = currDictEntry[plotFrequencyIdx, :, :, :]

            # Number of subplots
            nIter = currResRyy.shape[-1]
            nSubplotsX = int(np.floor(np.sqrt(nIter)))
            nSubplotsY = int(np.ceil(np.sqrt(nIter) + 0.5))

            fig, axs = plt.subplots(nSubplotsX, nSubplotsY)
            axs = axs.flatten()
            for jj in range(nIter):
                # Plot normalized matrix
                axs[jj].imshow(np.abs(currResRyy[:,:,jj] / np.amax(currResRyy[:,:,jj])))
                axs[jj].set_title(f'i={allIters[jj]}')
                axs[jj].set_xticks([])
                axs[jj].set_yticks([])
            fig.suptitle(f'$R_{{yy}}$ - {subdirs[ii].name} (Node {k+1}, $\kappa$={plotFrequencyIdx})')
            fig.tight_layout()
            saveitup(fig, exportPath + '/' + asc, f'RyyN{k+1}_{subdirs[ii].name}')

            # Rnn
            currDictEntry = bigdictRnn[subdirs[ii].name][k]     # SRO-conditions- and node-specific
            currResRnn = currDictEntry[plotFrequencyIdx, :, :, :]

            fig, axs = plt.subplots(nSubplotsX, nSubplotsY)
            axs = axs.flatten()
            for jj in range(nIter):
                # Plot normalized matrix
                axs[jj].imshow(np.abs(currResRnn[:,:,jj] / np.amax(currResRnn[:,:,jj])))
                axs[jj].set_title(f'i={allIters[jj]}')
                axs[jj].set_xticks([])
                axs[jj].set_yticks([])
            fig.suptitle(f'$R_{{nn}}$ - {subdirs[ii].name} (Node {k+1}, $\kappa$={plotFrequencyIdx})')
            fig.tight_layout()
            saveitup(fig, exportPath + '/' + asc, f'RnnN{k+1}_{subdirs[ii].name}')


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
import sys
import gzip
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('01_algorithms/01_NR/02_distributed')

DATAFOLDER = '01_algorithms/01_NR/02_distributed/res/for_submissions/icassp2023/test_postReviews/test_convergence_20230124/J4Mk[1_3_2_5]_Ns1_Nn2'
# EXPORT_FILES = True
EXPORT_FILES = False
SHOW_NOCOMP = True
# SHOW_NOCOMP = False

def main():
    
    # Load data
    dataComp, dataNoComp, dataCentr = load_data(path=DATAFOLDER)

    # Plot
    plotit(dataComp, dataNoComp, dataCentr)


def load_data(path):
    """Data loader."""

    dataComp = pickle.load(gzip.open(f'{path}/comp/Results.pkl.gz', 'r'))
    dataNoComp = pickle.load(gzip.open(f'{path}/nocomp/Results.pkl.gz', 'r'))
    dataCentr = pickle.load(gzip.open(f'{path}/centralised/Results.pkl.gz', 'r'))

    return dataComp, dataNoComp, dataCentr


def plotit(dataComp, dataNoComp, dataCentr):

    for k in range(len(dataComp.filtersEvolution.wTilde)):

        Mk = dataComp.acousticScenario.numSensorPerNode[k]
        # Extract relevant filter coefficients
        filtsComp = np.transpose(
            dataComp.filtersEvolution.wTilde[k][:, :, :Mk],
            axes=(1,0,2)
        )
        filtsNoComp = np.transpose(
            dataNoComp.filtersEvolution.wTilde[k][:, :, :Mk],
            axes=(1,0,2)
        )
        filtsCentr = np.transpose(
            dataCentr.filtersEvolution.wTilde[k][:, :, :Mk],
            axes=(1,0,2)
        )
        
        # Compute Deltas
        diffFiltersReal_comp = 20 * np.log10(np.mean(np.abs(
            np.real(filtsComp) - np.real(filtsCentr)
        ), axis=(1,2)))
        diffFiltersImag_comp = 20 * np.log10(np.mean(np.abs(
            np.imag(filtsComp) - np.imag(filtsCentr)
        ), axis=(1,2)))
        diffFiltersReal_nocomp = 20 * np.log10(np.mean(np.abs(
            np.real(filtsNoComp) - np.real(filtsCentr)
        ), axis=(1,2)))
        diffFiltersImag_nocomp = 20 * np.log10(np.mean(np.abs(
            np.imag(filtsNoComp) - np.imag(filtsCentr)
        ), axis=(1,2)))

        # diffFiltersReal_comp = 20 * np.log10(np.mean(np.abs(
        #     np.abs(filtsComp) - np.abs(filtsCentr)
        # ), axis=(1,2)))
        # diffFiltersImag_comp = 20 * np.log10(np.mean(np.abs(
        #     np.angle(filtsComp) - np.angle(filtsCentr)
        # ), axis=(1,2)))
        # diffFiltersReal_nocomp = 20 * np.log10(np.mean(np.abs(
        #     np.abs(filtsNoComp) - np.abs(filtsCentr)
        # ), axis=(1,2)))
        # diffFiltersImag_nocomp = 20 * np.log10(np.mean(np.abs(
        #     np.angle(filtsNoComp) - np.angle(filtsCentr)
        # ), axis=(1,2)))


        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(5.5, 3.5)
        axes.plot(diffFiltersReal_comp, 'k', label=f'$\Delta_\\mathrm{{r}}[i]$ with SRO comp.')
        axes.plot(diffFiltersImag_comp, 'r', label=f'$\Delta_\\mathrm{{i}}[i]$ with SRO comp.')
        if SHOW_NOCOMP:
            axes.plot(diffFiltersReal_nocomp, 'k--', label=f'$\Delta_\\mathrm{{r}}[i]$ without SRO comp.')
            axes.plot(diffFiltersImag_nocomp, 'r--', label=f'$\Delta_\\mathrm{{i}}[i]$ without SRO comp.')
        #
        axes.set_title(f'DANSE convergence towards centr. MWF estimate: node {k+1}')
        nIter = dataComp.filtersEvolution.wTilde[k].shape[1]
        axes.set_xlim([0, nIter])
        axes.set_xlabel('Frame index $i$', loc='left')
        axes.legend()
        axes.grid()
        #
        plt.tight_layout()
        # Export
        if EXPORT_FILES:
            plt.show(block=False)
            fname = f'{Path(__file__).parent}/convPlot_node{k+1}'
            if SHOW_NOCOMP:
                fname += '_noCompIncluded'
            fig.savefig(fname + '.png')
            fig.savefig(fname + '.pdf')
        else:
            plt.show()

    return None



if __name__ == '__main__':
    sys.exit(main())
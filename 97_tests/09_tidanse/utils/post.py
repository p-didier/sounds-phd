from .config import Configuration
import matplotlib.pyplot as plt
import numpy as np

class PostProcessor:
    def __init__(self, mmsePerAlgo, mmseCentral, cfg: Configuration):
        self.mmsePerAlgo = mmsePerAlgo
        self.mmseCentral = mmseCentral
        self.cfg = cfg
    
    def pre_process_mc_runs(self):
        # Ensure every MC run has data for the max number of iterations
        for idxMC in range(len(self.mmsePerAlgo)):
            for idxAlgo in range(len(self.cfg.algos)):
                for k in range(self.cfg.K):
                    if len(self.mmsePerAlgo[idxMC][idxAlgo][k]) < self.cfg.maxIter:
                        self.mmsePerAlgo[idxMC][idxAlgo][k] = np.append(
                            self.mmsePerAlgo[idxMC][idxAlgo][k],
                            np.ones(
                                self.cfg.maxIter-len(self.mmsePerAlgo[idxMC][idxAlgo][k])
                            ) * self.mmsePerAlgo[idxMC][idxAlgo][k][-1]
                        )
        # Mean / STD / max value / min value over MC runs
        self.mmsePerAlgoMean = []
        self.mmsePerAlgoStd = []
        self.mmsePerAlgoMax = []
        self.mmsePerAlgoMin = []
        for idxAlgo in range(len(self.cfg.algos)):
            self.mmsePerAlgoMean.append([])
            self.mmsePerAlgoStd.append([])
            self.mmsePerAlgoMax.append([])
            self.mmsePerAlgoMin.append([])            
            for k in range(self.cfg.K):
                arr = np.array([
                    self.mmsePerAlgo[idxMC][idxAlgo][k]
                    for idxMC in range(len(self.mmsePerAlgo))
                ])
                self.mmsePerAlgoMean[idxAlgo].append(np.mean(arr, axis=0))
                self.mmsePerAlgoStd[idxAlgo].append(np.std(arr, axis=0))
                self.mmsePerAlgoMax[idxAlgo].append(np.amax(arr, axis=0))
                self.mmsePerAlgoMin[idxAlgo].append(np.amin(arr, axis=0))
        if self.cfg.mcRuns > 1:
            arrC = np.array(self.mmseCentral)
            self.mmseCentralMean = np.mean(arrC, axis=0)
            self.mmseCentralStd = np.std(arrC, axis=0)
            self.mmseCentralMax = np.amax(arrC, axis=0)
            self.mmseCentralMin = np.amin(arrC, axis=0)
        else:
            self.mmseCentralMean = self.mmseCentral
            self.mmseCentralStd = np.zeros_like(self.mmseCentral)
            self.mmseCentralMax = self.mmseCentral
            self.mmseCentralMin = self.mmseCentral
        
    def plot_mmse(self):
        """Plot results."""
        # Pre-process the mutiple MC runs
        self.pre_process_mc_runs()
        if self.cfg.plotOnlyCost:
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(6.5, 3.5)
            for idxAlgo in range(len(self.cfg.algos)):
                # Current plot data
                dataMean = np.mean(np.array(self.mmsePerAlgoMean[idxAlgo]), axis=0)
                strELoss = 'E_\\mathrm{{MC\\,runs}}\\{{\\mathcal{{L}}\\}}'
                axes.loglog(
                    dataMean,
                    f'-C{idxAlgo}',
                    label=f'{self.cfg.algos[idxAlgo].upper()} (${strELoss}=${"{:.3g}".format(dataMean[-1], -4)})'
                )
                # Add shaded area for min/max
                dataMax = np.mean(np.array(self.mmsePerAlgoMax[idxAlgo]), axis=0)
                dataMin = np.mean(np.array(self.mmsePerAlgoMin[idxAlgo]), axis=0)
                axes.fill_between(
                    np.arange(len(dataMean)),
                    dataMin,
                    dataMax,
                    color=f'C{idxAlgo}',
                    alpha=0.1
                )
            if self.cfg.mode == 'batch':
                axes.hlines(np.mean(self.mmseCentralMean), 0, self.cfg.maxIter, 'k', linestyle="--", label=f'Centralized (${strELoss}=${"{:.3g}".format(np.mean(self.mmseCentral), -4)})')
            elif self.cfg.mode == 'online':
                axes.loglog(np.mean(self.mmseCentralMean, axis=0), '--k', label=f'Centralized (${strELoss}=${"{:.3g}".format(np.mean(self.mmseCentral, axis=0)[-1], -4)})')
            axes.set_xlabel("Iteration index")
            axes.set_ylabel("Cost $\\mathcal{L}$")
            axes.legend(loc='upper right')
            axes.set_xlim([0, self.cfg.maxIter])
            axes.grid()
            ti_str = f'$self.cfg.K={self.cfg.K}$, $M={self.cfg.Mk}$ mics/node, $\\mathrm{{self.cfg.snr}}={self.cfg.snr}$ dB, $\\mathrm{{self.cfg.snr}}_{{\\mathrm{{self}}}}={self.cfg.snSnr}$ dB, $\\mathrm{{self.cfg.gevd}}={self.cfg.gevd}$'
            if self.cfg.mode == 'online':
                ti_str += f', $self.cfg.B={self.cfg.B}$ ({int(self.cfg.overlapB * 100)}%ovlp), $\\beta={self.cfg.beta}$'
            elif self.cfg.mode == 'batch':
                ti_str += f', {self.cfg.nSamplesTot} samples'
            axes.set_title(ti_str)
        else:
            fig, axes = plt.subplots(2, len(self.cfg.algos), sharey='row', sharex='col')
            fig.set_size_inches(8.5, 3.5)
            for idxAlgo in range(len(self.cfg.algos)):
                if len(self.cfg.algos) == 1:
                    currAx = axes
                else:
                    currAx = axes[:, idxAlgo]
                if self.cfg.mode == 'batch':
                    xmax = len(self.mmsePerAlgo[idxAlgo][0])-1
                elif self.cfg.mode == 'online':
                    xmax = np.amax([len(self.mmsePerAlgo[idxAlgo][0])-1, len(self.mmseCentral[0])-1])
                for k in range(self.cfg.K):
                    currAx[0].loglog(self.mmsePerAlgo[idxAlgo][k], f'-C{k}', label=f"Node {k}")
                    if self.cfg.mode == 'batch':
                        currAx[0].hlines(self.mmseCentral[k], 0, xmax, f'C{k}', linestyle="--")
                    elif self.cfg.mode == 'online':
                        currAx[0].loglog(self.mmseCentral[k, :], f'--C{k}')
                currAx[0].set_xlabel(f"{self.cfg.algos[idxAlgo].upper()} iteration index")
                currAx[0].set_ylabel("MMSE per node")
                currAx[0].legend(loc='upper right')
                currAx[0].set_xlim([0, xmax])
                currAx[0].grid()
                #
                currAx[1].loglog(np.mean(np.array(self.mmsePerAlgo[idxAlgo]), axis=0), '-k', label=f'{self.cfg.algos[idxAlgo].upper()} ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(np.array(self.mmsePerAlgo[idxAlgo]), axis=0)[-1], -4)})')
                if self.cfg.mode == 'batch':
                    currAx[1].hlines(np.mean(self.mmseCentral), 0, xmax, 'k', linestyle="--", label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(self.mmseCentral), -4)})')
                elif self.cfg.mode == 'online':
                    currAx[1].loglog(np.mean(self.mmseCentral, axis=0), '--k', label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(self.mmseCentral, axis=0)[-1], -4)})')
                currAx[1].set_xlabel(f"{self.cfg.algos[idxAlgo].upper()} iteration index")
                currAx[1].set_ylabel("Cost")
                currAx[1].legend(loc='upper right')
                currAx[1].set_xlim([0, xmax])
                currAx[1].grid()
        #
        fig.tight_layout()	
        plt.show(block=False)

        return fig, axes

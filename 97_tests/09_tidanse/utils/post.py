import os
import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .config import Configuration

class PostProcessor:
    def __init__(
            self,
            mmsePerAlgo, mmseCentral,
            filtersPerAlgo, filtersCentral,
            vadSaved,
            cfg: Configuration, export, suffix=''
        ):
        self.mmsePerAlgo = mmsePerAlgo
        self.mmseCentral = mmseCentral
        self.filtersPerAlgo = filtersPerAlgo
        self.filtersCentral = filtersCentral
        self.vadSaved = vadSaved
        self.cfg = cfg
        self.export = export
        self.suffix = suffix

    def perform_post_processing(self):
        """Perform post-processing of results."""
        # Export subfolder path with date and time
        now = datetime.datetime.now()
        sf = self.cfg.exportFolder +\
            f'\\{Path(self.cfg.exportFolder).name}_{now.strftime("%Y%m%d")}\\' +\
            'res_' + now.strftime("%Y%m%d_%H%M%S")
        if self.suffix != '':
            sf += '__' + self.suffix
        if self.export:
            # Check if export folder exists
            if not os.path.exists(sf):
                os.makedirs(sf)
            # Save results
            np.save(os.path.join(sf, 'mmsePerAlgo.npy'), self.mmsePerAlgo)
            np.save(os.path.join(sf, 'mmseCentral.npy'), self.mmseCentral)
            # Save config as text file
            with open(os.path.join(sf, 'config.txt'), 'w') as f:
                f.write(self.cfg.to_string())
            # Save config as YAML file
            self.cfg.to_yaml(os.path.join(sf, 'config.yaml'))
        # Plot MMSE
        fig = self.plot_mmse()
        if self.export:
            fig.savefig(os.path.join(sf, 'mmse.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(sf, 'mmse.png'), bbox_inches='tight', dpi=300)
        # Plot filters
        figs = self.plot_filters()
        if self.export:
            for k in range(self.cfg.K):
                # figs[k].savefig(os.path.join(sf, f'filters_node{k}.pdf'), bbox_inches='tight')
                figs[k].savefig(os.path.join(sf, f'filters_node{k}.png'), bbox_inches='tight', dpi=300)

    def plot_filters(self):
        """Plot (TI-)DANSE filter coefficients for each node and each
        algorithm."""
        figs = dict([(k, None) for k in range(self.cfg.K)])
        mcRunIdx = 0  # only plot first MC run (arbitrary)
        # Determine fixed y-axis limits for all plots, per algorithm
        yLimMin = [np.inf for _ in range(len(self.cfg.algos))]
        yLimMax = [-np.inf for _ in range(len(self.cfg.algos))]
        for idxAlgo, algo in enumerate(self.cfg.algos):
            for k in range(self.cfg.K):
                currFilts = np.array([
                    w[k] for w in self.filtersPerAlgo[mcRunIdx][idxAlgo]
                ])
                yLimMin[idxAlgo] = min(yLimMin[idxAlgo], np.amin(np.abs(currFilts)))
                yLimMax[idxAlgo] = max(yLimMax[idxAlgo], np.amax(np.abs(currFilts)))
        for k in range(self.cfg.K):
            fig, axes = plt.subplots(len(self.cfg.algos), 1, sharex=True)
            fig.set_size_inches(8.5, 4.5)
            for idxAlgo, algo in enumerate(self.cfg.algos):
                if len(self.cfg.algos) == 1:
                    axes = [axes]
                currFilts = np.array([
                    w[k] for w in self.filtersPerAlgo[mcRunIdx][idxAlgo]
                ])
                for m in range(currFilts.shape[1]):
                    if m < self.cfg.Mk:
                        lab = f'$w_{{kk,{m}}}$'
                    else:
                        if algo == 'danse':
                            lab = f'$g_{{k{{-}}k,{m - self.cfg.Mk}}}$'
                        else:
                            lab = f'$g_{{k,{m - self.cfg.Mk}}}$'
                    # Plot filter coefficients
                    axes[idxAlgo].semilogy(np.abs(currFilts[:, m]), label=lab)
                # Plot VAD as shaded grey area
                if self.cfg.sigConfig.desiredSignalType == 'noise+pauses':
                    axes[idxAlgo].fill_between(
                        np.arange(len(self.vadSaved)),
                        np.zeros_like(self.vadSaved),
                        np.array(self.vadSaved) * yLimMax[idxAlgo],
                        color='grey',
                        alpha=0.2,
                        label='VAD'
                    )
                axes[idxAlgo].legend(loc='lower right')
                axes[idxAlgo].grid()
                if idxAlgo == len(self.cfg.algos) - 1:
                    axes[idxAlgo].set_xlabel('Iteration index')
                axes[idxAlgo].set_ylabel('Coefficient norm')
                axes[idxAlgo].set_title(f'Node {k}, {algo.upper()}')
                axes[idxAlgo].set_ylim([yLimMin[idxAlgo], yLimMax[idxAlgo]])
            fig.tight_layout()
            figs[k] = fig
        plt.close('all')
        return figs
    
    def pre_process_mc_runs(self):
        # Find the maximum iteration reached by any MC run
        maxIter = 0
        for idxMC in range(len(self.mmsePerAlgo)):
            for idxAlgo in range(len(self.cfg.algos)):
                for k in range(self.cfg.K):
                    maxIter = max(maxIter, len(self.mmsePerAlgo[idxMC][idxAlgo][k]))
        # Ensure every MC run has data for the max number of iterations
        for idxMC in range(len(self.mmsePerAlgo)):
            for idxAlgo in range(len(self.cfg.algos)):
                for k in range(self.cfg.K):
                    if len(self.mmsePerAlgo[idxMC][idxAlgo][k]) < maxIter:
                        self.mmsePerAlgo[idxMC][idxAlgo][k] = np.append(
                            self.mmsePerAlgo[idxMC][idxAlgo][k],
                            np.ones(
                                maxIter - len(self.mmsePerAlgo[idxMC][idxAlgo][k])
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
            self.mmseCentralMean = self.mmseCentral[0]
            self.mmseCentralStd = np.zeros_like(self.mmseCentral[0])
            self.mmseCentralMax = self.mmseCentral[0]
            self.mmseCentralMin = self.mmseCentral[0]
        
        return maxIter
        
    def plot_mmse(self, asLogLog=False):
        """Plot results."""
        # Pre-process the mutiple MC runs
        maxIter = self.pre_process_mc_runs()
        if self.cfg.mcRuns > 1:
            strLoss = 'E_\\mathrm{{MC\\,runs}}\\{{\\mathcal{{L}}\\}}'  # average over MC runs
        else:
            strLoss = '\\mathcal{{L}}'

        def _loss_for_legend(data):
            begIdx = int(len(data) / 5)
            value = "{:.3g}".format(np.nanmean(data[-begIdx]), -4)
            return f'${strLoss}=${value} over last {begIdx} iter.'
        
        if self.cfg.plotOnlyCost:
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(6.5, 3.5)
            for idxAlgo in range(len(self.cfg.algos)):
                # Current plot data
                dataMean = np.mean(np.array(self.mmsePerAlgoMean[idxAlgo]), axis=0)
                algoref = self.cfg.algos[idxAlgo].upper()
                if self.cfg.gevd:
                    if '-' in algoref:
                        algoref = algoref.split('-')[0] + '-GEVD-' + algoref.split('-')[1]
                    else:
                        algoref = 'GEVD-' + algoref
                axes.loglog(
                    dataMean,
                    f'-C{idxAlgo}',
                    label=f'{algoref} ({_loss_for_legend(dataMean)})'
                )
                # Add shaded area for min/max
                if self.cfg.mcRuns > 1:
                    dataMax = np.mean(np.array(self.mmsePerAlgoMax[idxAlgo]), axis=0)
                    dataMin = np.mean(np.array(self.mmsePerAlgoMin[idxAlgo]), axis=0)
                    axes.fill_between(
                        np.arange(maxIter),
                        dataMin,
                        dataMax,
                        color=f'C{idxAlgo}',
                        alpha=0.1
                    )
            centrAlgoStr = 'GEVD-MWF' if self.cfg.gevd else 'MWF'
            if self.cfg.mode == 'batch':
                axes.hlines(np.mean(self.mmseCentralMean), 0, maxIter, 'k', linestyle="--", label=f'Centralized {centrAlgoStr} (${strLoss}=${"{:.3g}".format(np.mean(self.mmseCentralMean), -4)})')
            elif self.cfg.mode == 'online':
                axes.loglog(np.mean(self.mmseCentralMean, axis=0), '--k', label=f'Centralized {centrAlgoStr} ({_loss_for_legend(np.mean(self.mmseCentralMean, axis=0))})')
            
            # Add grey transparent area for frames where VAD is 0
            if self.cfg.sigConfig.desiredSignalType == 'noise+pauses':
                axes.fill_between(
                    np.arange(maxIter),
                    np.zeros_like(self.vadSaved),
                    (1 - np.array(self.vadSaved)) * np.amax(axes.get_ylim()),
                    color='grey',
                    alpha=0.2,
                    label='VAD=0 frames'
                )
            
            axes.set_xlabel("Iteration index")
            axes.set_ylabel("Cost $\\mathcal{L}$")
            axes.legend(loc='upper right', fontsize='small')
            axes.set_xlim([0, maxIter])
            axes.grid()
            if not asLogLog:
                # Set x-axis to linear
                axes.set_xscale('linear')
            ti_str = f'$K={self.cfg.K}$, $M_{{k}}={self.cfg.Mk}$, $\\mathrm{{SNR}}={self.cfg.snr}$ dB, $\\mathrm{{SNR}}_{{\\mathrm{{SN}}}}={self.cfg.snSnr}$ dB'
            if self.cfg.mode == 'online':
                ti_str += f', $B={self.cfg.B}$, $\\beta={self.cfg.beta}$'
                if self.cfg.beta != self.cfg.betaRnn:
                    ti_str += f', $\\beta_{{\\mathbf{{R_{{nn}}}}}}={self.cfg.betaRnn}$'
            elif self.cfg.mode == 'batch':
                ti_str += f', {self.cfg.sigConfig.nSamplesBatch} samples'
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
                        currAx[0].hlines(self.mmseCentral[idxAlgo][k], 0, xmax, f'C{k}', linestyle="--")
                    elif self.cfg.mode == 'online':
                        currAx[0].loglog(self.mmseCentral[idxAlgo][k, :], f'--C{k}')
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

        return fig

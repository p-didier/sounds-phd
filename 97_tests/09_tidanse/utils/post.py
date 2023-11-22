import os
import sys
import datetime
import numpy as np
from pathlib import Path
import scipy.linalg as sla
import matplotlib.pyplot as plt
from .config import Configuration

class PostProcessor:
    def __init__(
            self,
            mmsePerAlgo, mmseCentral,
            filtersPerAlgo, filtersCentral,
            RyyPerAlgo, RnnPerAlgo,
            vadSaved,
            cfg: Configuration, export
        ):
        self.mmsePerAlgo = mmsePerAlgo
        self.mmseCentral = mmseCentral
        self.filtersPerAlgo = filtersPerAlgo
        self.filtersCentral = filtersCentral
        self.RyyPerAlgo = RyyPerAlgo
        self.RnnPerAlgo = RnnPerAlgo
        self.vadSaved = vadSaved
        self.cfg = cfg
        self.export = export

    def perform_post_processing(self):
        """Perform post-processing of results."""
        # Export subfolder path with date and time
        now = datetime.datetime.now()
        sf = self.cfg.exportFolder +\
            f'\\{Path(self.cfg.exportFolder).name}_{now.strftime("%Y%m%d")}\\' +\
            'res_' + now.strftime("%Y%m%d_%H%M%S")
        if self.cfg.suffix != '':
            sf += '__' + self.cfg.suffix
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
        # Plot Ryy and Rnn coefficients
        figs = self.plot_Ryy_Rnn()
        if self.export:
            for fig in figs:
                fig[1].savefig(os.path.join(sf, f'{fig[0]}.pdf'), bbox_inches='tight')
                fig[1].savefig(os.path.join(sf, f'{fig[0]}.png'), bbox_inches='tight', dpi=300)
        # Plot filters
        figs = self.plot_filters()
        if self.export:
            for k in range(self.cfg.K):
                figs[k].savefig(os.path.join(sf, f'filters_node{k}.pdf'), bbox_inches='tight')
                figs[k].savefig(os.path.join(sf, f'filters_node{k}.png'), bbox_inches='tight', dpi=300)

    def plot_Ryy_Rnn(self):
        """Plots the evolution of some of the SCMs' coefficients over
        (TI-)DANSE iterations."""

        def _title_maker(typ='n'):
            return f'$\\log_{{10}}(\\tilde{{\\mathbf{{R}}}}_{{\\mathbf{{{typ}}}_{{k}}\\mathbf{{{typ}}}_{{k}}}}^{{i}}[{{-}}1,{{-}}1])$' +\
                f'   =  $\\log_{{10}}(\\mathbb{{E}}\\{{\\eta_{{-k}}^{{i}}\\eta_{{-k}}^{{iH}}\\}})$'

        figs = []

        if self.cfg.mcRuns > 1:
            return None   # TODO: implement
        else:
            lineStyles = ['-', '--', '-.', ':']
            fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
            fig.set_size_inches(8.5, 5.5)
            # Set position of figure
            fig.canvas.manager.window.move(0,0)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            ylimMin, ylimMax = np.inf, -np.inf
            for idxAlgo, algo in enumerate(self.cfg.algos):
                for k in range(self.cfg.K):
                    dataRyy = np.array(self.RyyPerAlgo[0][idxAlgo][k])
                    dataRnn = np.array(self.RnnPerAlgo[0][idxAlgo][k])
                    toPlotRyy = np.log10(dataRyy[:, -1, -1])
                    toPlotRyy[np.isinf(toPlotRyy)] = np.nan
                    toPlotRyy = np.nan_to_num(toPlotRyy, nan=np.nanmin(toPlotRyy))
                    toPlotRnn = np.log10(dataRnn[:, -1, -1])
                    toPlotRnn[np.isinf(toPlotRnn)] = np.nan
                    toPlotRnn = np.nan_to_num(toPlotRnn, nan=np.nanmin(toPlotRnn))
                    axes[0].plot(toPlotRyy, f'{lineStyles[idxAlgo]}C{k}', label=f'{self.cfg.algoNames[algo]}, node {k}')
                    axes[1].plot(toPlotRnn, f'{lineStyles[idxAlgo]}C{k}', label=f'{self.cfg.algoNames[algo]}, node {k}')
                    ylimMin = min(ylimMin, np.amin(toPlotRyy), np.amin(toPlotRnn))
                    ylimMax = max(ylimMax, np.amax(toPlotRyy), np.amax(toPlotRnn))
            axes[0].set_title(_title_maker('y'))
            axes[1].set_title(_title_maker('n'))
            # Show VAD
            if self.cfg.sigConfig.desiredSignalType == 'noise+pauses':
                axes[0].fill_between(
                    np.arange(len(self.vadSaved)),
                    np.zeros_like(self.vadSaved),
                    (1 - np.array(self.vadSaved)) * axes[0].get_ylim()[1],
                    color='grey',
                    alpha=0.2,
                    label='VAD=0'
                )
                axes[1].fill_between(
                    np.arange(len(self.vadSaved)),
                    np.zeros_like(self.vadSaved),
                    (1 - np.array(self.vadSaved)) * axes[1].get_ylim()[1],
                    color='grey',
                    alpha=0.2,
                    label='VAD=0'
                )
            axes[0].grid()
            axes[1].grid()
            axes[0].set_ylim([
                ylimMin,
                ylimMax
            ])
            axes[0].set_xlim([0, len(toPlotRyy)])
            axes[0].legend(loc='upper right', fontsize='small')
            axes[1].set_xlabel('Iteration index $i$')
            fig.tight_layout()
            plt.show(block=False)
            figs.append(('Ryy_Rnn', fig))

            # Plot GEVLs
            markers = ['o', 'x', 'd', 'v', '^', '<', '>', 'p', 'h', '8']
            fig, axes = plt.subplots(1,1)
            fig.set_size_inches(8.5, 3.5)
            # Set position of figure
            fig.canvas.manager.window.move(0,0)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            for idxAlgo, algo in enumerate(self.cfg.algos):
                for k in range(self.cfg.K):
                    currRyy = np.array(self.RyyPerAlgo[0][idxAlgo][k])
                    currRnn = np.array(self.RnnPerAlgo[0][idxAlgo][k])
                    # Compute GEVLs
                    gevls = np.full(currRyy.shape[0], fill_value=np.nan)
                    for i in range(currRyy.shape[0]):
                        try:
                            allGEVLs, _ = sla.eigh(
                                currRyy[i, :, :],
                                currRnn[i, :, :]
                            )
                        except:
                            continue
                        gevls[i] = np.amax(allGEVLs)
                    axes.semilogy(gevls, f'{markers[idxAlgo]}{lineStyles[idxAlgo]}C{k}', label=f'{self.cfg.algoNames[algo]}, node {k}', markersize=3)
            axes.grid()
            # Show VAD
            if self.cfg.sigConfig.desiredSignalType == 'noise+pauses':
                axes.fill_between(
                    np.arange(len(self.vadSaved)),
                    np.zeros_like(self.vadSaved),
                    (1 - np.array(self.vadSaved)) * axes.get_ylim()[1],
                    color='grey',
                    alpha=0.2,
                    label='VAD=0'
                )
            axes.set_xlabel('Iteration index $i$')
            axes.set_ylabel('Max. GEVL')
            axes.set_xlim([0, len(gevls)])
            axes.legend(loc='upper right', fontsize='small')
            fig.tight_layout()
            plt.show(block=False)
            figs.append(('GEVLs', fig))

            return figs

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
                        (1 - np.array(self.vadSaved)) * axes[idxAlgo].get_ylim()[1],
                        color='grey',
                        alpha=0.2,
                        label='VAD=0'
                    )
                axes[idxAlgo].legend(loc='lower right')
                axes[idxAlgo].grid()
                if idxAlgo == len(self.cfg.algos) - 1:
                    axes[idxAlgo].set_xlabel('Iteration index')
                axes[idxAlgo].set_ylabel('Coefficient norm')
                algoref = algo.upper()
                if self.cfg.gevd:
                    if '-' in algoref:
                        algoref = algoref.split('-')[0] + '-GEVD-' + algoref.split('-')[1]
                    else:
                        algoref = 'GEVD-' + algoref
                axes[idxAlgo].set_title(f'Node {k}, {algoref}')
                if np.isinf(yLimMax[idxAlgo]):
                    yLimMax[idxAlgo] = axes[idxAlgo].get_ylim()[1]
                if np.isinf(yLimMin[idxAlgo]):
                    yLimMin[idxAlgo] = axes[idxAlgo].get_ylim()[0]
                axes[idxAlgo].set_ylim([yLimMin[idxAlgo], yLimMax[idxAlgo]])
            fig.tight_layout()
            plt.show(block=False)
            figs[k] = fig
        # plt.close('all')
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
                    xmax = len(self.mmsePerAlgoMean[idxAlgo][0])-1
                elif self.cfg.mode == 'online':
                    xmax = np.amax([len(self.mmsePerAlgoMean[idxAlgo][0])-1, len(self.mmseCentralMean[0])-1])
                for k in range(self.cfg.K):
                    currAx[0].loglog(self.mmsePerAlgoMean[idxAlgo][k], f'-C{k}', label=f"Node {k}")
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
                currAx[1].loglog(np.mean(np.array(self.mmsePerAlgoMean[idxAlgo]), axis=0), '-k', label=f'{self.cfg.algos[idxAlgo].upper()} ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(np.array(self.mmsePerAlgoMean[idxAlgo]), axis=0)[-1], -4)})')
                if self.cfg.mode == 'batch':
                    currAx[1].hlines(np.mean(self.mmseCentralMean), 0, xmax, 'k', linestyle="--", label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(self.mmseCentralMean), -4)})')
                elif self.cfg.mode == 'online':
                    currAx[1].loglog(np.mean(self.mmseCentralMean, axis=0), '--k', label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(self.mmseCentralMean, axis=0)[-1], -4)})')
                currAx[1].set_xlabel(f"{self.cfg.algos[idxAlgo].upper()} iteration index")
                currAx[1].set_ylabel("Cost")
                currAx[1].legend(loc='upper right')
                currAx[1].set_xlim([0, xmax])
                currAx[1].grid()
        #
        fig.tight_layout()	
        plt.show(block=False)

        return fig

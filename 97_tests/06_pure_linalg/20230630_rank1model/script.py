# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
sys.path.append('..')

import copy
import wavfile
import numpy as np
from pathlib import Path
from utils.plots import *
from utils.general import *
sys.path.append('C:\\Users\\pdidier\\Dropbox\\PC\\Documents\\sounds-phd')
from _general_fcts.class_methods.dataclass_methods import *

# Global parameters
PATH_TO_YAML = f'{Path(__file__).parent.absolute()}\\yaml\\script_parameters.yaml'

def main(pathToYaml: str = PATH_TO_YAML, p: ScriptParameters = None):
    """Main function (called by default when running script)."""

    # Get parameters
    if p is None:
        p: ScriptParameters = load_from_yaml(pathToYaml, ScriptParameters())
    elif pathToYaml is None:
        pass  # use parameters passed as argument
    else:
        raise ValueError('`pathToYaml` should be None if `p` is not None')

    # Set random seed
    np.random.seed(p.seed)
    rngState = np.random.get_state()

    if p.Mk is None:
        # For DANSE: randomly assign sensors to nodes, ensuring that each node
        # has at least one sensor
        channelToNodeMap = np.zeros(p.nSensors, dtype=int)
        for k in range(p.nNodes):
            channelToNodeMap[k] = k
        for n in range(p.nNodes, p.nSensors):
            channelToNodeMap[n] = np.random.randint(0, p.nNodes)
        # Sort
        channelToNodeMap = np.sort(channelToNodeMap)
    else:
        p.nNodes = len(p.Mk)
        p.nSensors = np.sum(p.Mk)
        # Assign sensors to nodes according to Mk
        channelToNodeMap = np.zeros(p.nSensors, dtype=int)
        for k in range(p.nNodes):
            idxStart = int(np.sum(p.Mk[:k]))
            idxEnd = idxStart + p.Mk[k]
            channelToNodeMap[idxStart:idxEnd] = k

    if isinstance(p.wolaParams.betaExt, (float, int)):
        p.wolaParams.betaExt = np.array([p.wolaParams.betaExt])

    for betaExtCurr in p.wolaParams.betaExt:
        if p.verbose:
            print(f'>>>>>>>> Running with betaExt = {betaExtCurr}')

        # Set RNG state back to original for each betaExt loop iteration
        np.random.set_state(rngState)

        # Set external beta
        wolaParamsCurr: WOLAparameters = copy.deepcopy(p.wolaParams)
        wolaParamsCurr.betaExt = betaExtCurr
        # Initialize dictionary where results are stored for plotting
        toDict = []
        for filterType in p.toCompute:
            if filterType.indiv_frames():
                # Compute number of iterations
                if filterType.online:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.nfft) # divide by frame size
                elif filterType.wola:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.hop) - 1  # divide by hop size
                
                if p.showDeltaPerNode:
                    toDict.append((filterType.to_str(), np.zeros((p.nMC, nIter, len(p.taus), p.nNodes))))
                else:
                    toDict.append((filterType.to_str(), np.zeros((p.nMC, nIter, len(p.taus)))))
            else:
                if p.showDeltaPerNode:
                    toDict.append((filterType.to_str(), np.zeros((p.nMC, len(p.durations), p.nNodes))))
                else:
                    toDict.append((filterType.to_str(), np.zeros((p.nMC, len(p.durations)))))
        metricsData = dict(toDict)

        for idxMC in range(p.nMC):
            print(f'Running Monte-Carlo iteration {idxMC+1}/{p.nMC}')

            # Generate signals
            pCurr = copy.deepcopy(p)
            pCurr.wolaParams = wolaParamsCurr
            # TODO: make that prettier vvvvvvvvv
            cleanSigs, noiseSignals, scalings, scalingsNoise, powers, vad =\
                generate_signals(pCurr)

            # Compute desired filters
            allFilters = get_filters(
                cleanSigs,
                noiseSignals,
                channelToNodeMap,
                gevdRank=1,
                toCompute=p.toCompute,
                wolaParams=wolaParamsCurr,
                taus=p.taus,
                durations=p.durations,
                b=p.b,
                batchFusionInOnline=p.useBatchModeFusionVectorsInOnlineDanse,
                noSSfusion=p.ignoreFusionForSSNodes,
                verbose=p.verbose,
                vad=vad
            )

            # Compute metrics
            RnnOracle, RnnOracleWOLA = get_oracle_noise_covariance_matrices(
                powers, scalingsNoise
            )
            for filterType in p.toCompute:
                currFilters = allFilters[filterType.to_str()]
                if filterType.indiv_frames():  # online or WOLA in non-batch mode
                    for idxTau in range(len(p.taus)):
                        if filterType.online:
                            # Compute oracle noise covariance matrix
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :],
                                scalings,
                                powers['clean'],
                                RnnOracle,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode,  # export all differences individually
                                sigma_nr=powers['allNoise']
                            )
                        elif filterType.wola:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :, :],
                                scalings,
                                powers['cleanWOLA'],
                                RnnOracleWOLA,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode,  # export all differences individually
                                sigma_nr=powers['allNoiseWOLA']
                            )
                        if p.showDeltaPerNode:
                            metricsData[filterType.to_str()][idxMC, :, idxTau, :] = currMetrics
                        else:
                            metricsData[filterType.to_str()][idxMC, :, idxTau] = currMetrics
                else:  # batch mode (WOLA or not)
                    for idxDur in range(len(p.durations)):
                        if filterType.wola:
                            sigmaSrEffective = copy.deepcopy(powers['cleanWOLA'])
                            # sigmaNrEffective = copy.deepcopy(sigmaNrWOLA)
                            RnnOracleEffective = copy.deepcopy(RnnOracleWOLA)
                        else:
                            sigmaSrEffective = copy.deepcopy(powers['clean'])
                            # sigmaNrEffective = copy.deepcopy(sigmaNr)
                            RnnOracleEffective = copy.deepcopy(RnnOracle)

                        currFiltCurrDur = currFilters[idxDur, :, :]
                        currMetrics = get_metrics(
                            p.nSensors,
                            currFiltCurrDur,
                            scalings,
                            sigmaSrEffective,
                            RnnOracleEffective,
                            channelToNodeMap,
                            filterType=filterType,
                            exportDiffPerFilter=p.showDeltaPerNode,  # export all differences individually
                            sigma_nr=powers['allNoiseWOLA'] if filterType.wola else powers['allNoise']
                        )
                        if p.showDeltaPerNode:
                            metricsData[filterType.to_str()][idxMC, idxDur, :] = currMetrics
                        else:
                            metricsData[filterType.to_str()][idxMC, idxDur] = currMetrics
        
        # If asked, get plotted results (waveforms and speech
        # enhancement metrics) for speech signals
        if p.targetSignalType == 'speech' and p.speechPlots:
            fig1, fig2, filteredSignals = speech_plots(
                p,
                allFilters,
                cleanSigs,
                noiseSignals,
                vad
            )
            if p.exportFolder is not None:
                if not Path(p.exportFolder).is_dir():
                    Path(p.exportFolder).mkdir(parents=True, exist_ok=True)
                fig1.savefig(f'{p.exportFolder}/speech_waveforms.png', dpi=300, bbox_inches='tight')
                fig2.savefig(f'{p.exportFolder}/speech_metrics.png', dpi=300, bbox_inches='tight')
                # Export filtered signals as wav
                if p.exportFilteredSignals:
                    for key in filteredSignals.keys():
                        print(f'Exporting filtered signal "{key}.wav"...')
                        wavfile.write(
                            f'{p.exportFolder}/{key}.wav',
                            normalize_toint16(filteredSignals[key][:, np.newaxis]),
                            p.wolaParams.fs,
                        )
        
        if p.exportFigures:
            # Plot results
            substr = ''
            for mk in p.Mk:
                substr += f'{mk},'
            substr = substr[:-1]
            figTitleSuffix = f'"{p.targetSignalType}", $\\{{M_k\\}} = \\{{{substr}\\}}$ '
            if any([t.danse and (t.online or t.wola) for t in p.toCompute]):
                figTitleSuffix += f'$\\beta_{{\\mathrm{{EXT}}}} = {np.round(betaExtCurr, 4)}$ '
            if wolaParamsCurr.singleFreqBinIndex is not None and\
                any([t.wola for t in p.toCompute]):
                figTitleSuffix += f"[WOLA's {wolaParamsCurr.singleFreqBinIndex + 1}-th freq. bin]"
            fig = plot_final(
                p.durations,
                p.taus,
                metricsData,
                fs=wolaParamsCurr.fs,
                L=wolaParamsCurr.nfft,
                R=wolaParamsCurr.hop,
                avgAcrossNodesFlag=not p.showDeltaPerNode,
                figTitleSuffix=figTitleSuffix if len(figTitleSuffix) > 0 else None,
                vad=vad
            )

            if p.exportFolder is not None:
                if len(p.durations) > 1:
                    fname = f'{p.exportFolder}/diff'
                else:
                    fname = f'{p.exportFolder}/betas'
                fname += f'_betaExt_0p{int(betaExtCurr * 1e3)}'
                for t in p.toComputeStrings:
                    fname += f'_{t}'
                if not Path(p.exportFolder).is_dir():
                    Path(p.exportFolder).mkdir(parents=True, exist_ok=True)
                fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
        
            # Wait for button press to close the figures
            # plt.waitforbuttonpress()
            # plt.close('all')

    return metricsData, vad


if __name__ == '__main__':
    sys.exit(main())
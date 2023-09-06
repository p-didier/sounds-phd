# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
sys.path.append('..')

import copy
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

            pCurr = copy.deepcopy(p)
            pCurr.wolaParams = wolaParamsCurr
            cleanSigs, noiseSignals, scalings, sigmaSr, sigmaNr,\
                sigmaSrWOLA, sigmaNrWOLA, vad = generate_signals(pCurr)
            # # Get scalings
            # if 'complex' in p.targetSignalType:
            #     scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors) +\
            #         1j * np.random.uniform(low=0.5, high=1, size=p.nSensors)
            # else:
            #     scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors)
            # # Get clean signals
            # nSamplesMax = int(np.amax(p.durations) * wolaParamsCurr.fs)
            # cleanSigs, _, vad = get_clean_signals(
            #     p,
            #     scalings,
            #     maxDelay=0.1,
            # )
            # if vad is not None:
            #     sigmaSr = np.sqrt(
            #         np.mean(
            #             np.abs(
            #                 cleanSigs[np.squeeze(vad).astype(bool), :]
            #             ) ** 2,
            #             axis=0
            #         )
            #     )
            #     sigmaSrWOLA = get_sigma_wola(
            #         cleanSigs[np.squeeze(vad).astype(bool), :],
            #         wolaParamsCurr
            #     )
            # else:
            #     sigmaSr = np.sqrt(np.mean(np.abs(cleanSigs) ** 2, axis=0))
            #     sigmaSrWOLA = get_sigma_wola(cleanSigs, wolaParamsCurr)

            # # Generate noise signals
            # sigmaNr = np.zeros(p.nSensors)
            # if wolaParamsCurr.singleFreqBinIndex is not None:
            #     sigmaNrWOLA = np.zeros((1, p.nSensors))
            # else:
            #     sigmaNrWOLA = np.zeros((wolaParamsCurr.nPosFreqs, p.nSensors))
            # if np.iscomplex(cleanSigs).any():
            #     noiseSignals = np.zeros((nSamplesMax, p.nSensors), dtype=np.complex128)
            # else:
            #     noiseSignals = np.zeros((nSamplesMax, p.nSensors))
            # for n in range(p.nSensors):
            #     # Generate random sequence with unit power
            #     if np.iscomplex(cleanSigs).any():
            #         randSequence = np.random.normal(size=nSamplesMax) +\
            #             1j * np.random.normal(size=nSamplesMax)
                    
            #     else:
            #         randSequence = np.random.normal(size=nSamplesMax)
            #     # Make unit power
            #     randSequence /= np.sqrt(np.mean(np.abs(randSequence) ** 2))
            #     # Scale to desired power
            #     noiseSignals[:, n] = randSequence * np.sqrt(p.selfNoisePower)
            #     # Check power
            #     sigmaNr[n] = np.sqrt(np.mean(np.abs(noiseSignals[:, n]) ** 2))
            #     if np.abs(sigmaNr[n] ** 2 - p.selfNoisePower) > 1e-6:
            #         raise ValueError(f'Noise signal power is {sigmaNr[n] ** 2} instead of {p.selfNoisePower}')
            #     sigmaNrWOLA[:, n] = get_sigma_wola(
            #         noiseSignals[:, n],
            #         wolaParamsCurr
            #     )

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

            if p.targetSignalType == 'speech' and p.listenToSpeech:
                listen_to_speech(p, allFilters, cleanSigs, noiseSignals)

            # Compute metrics
            for filterType in p.toCompute:
                currFilters = allFilters[filterType.to_str()]
                if filterType.indiv_frames():  # online or WOLA in non-batch mode
                    for idxTau in range(len(p.taus)):
                        if filterType.online:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :],
                                scalings,
                                sigmaSr,
                                sigmaNr,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                            )
                        elif filterType.wola:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :, :],
                                scalings,
                                sigmaSrWOLA,
                                sigmaNrWOLA,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                            )
                        if p.showDeltaPerNode:
                            metricsData[filterType.to_str()][idxMC, :, idxTau, :] = currMetrics
                        else:
                            metricsData[filterType.to_str()][idxMC, :, idxTau] = currMetrics
                else:  # batch mode (WOLA or not)
                    for idxDur in range(len(p.durations)):
                        if filterType.wola:
                            sigmaSrEffective = copy.deepcopy(sigmaSrWOLA)
                            sigmaNrEffective = copy.deepcopy(sigmaNrWOLA)
                        else:
                            sigmaSrEffective = copy.deepcopy(sigmaSr)
                            sigmaNrEffective = copy.deepcopy(sigmaNr)

                        currFiltCurrDur = currFilters[idxDur, :, :]
                        currMetrics = get_metrics(
                            p.nSensors,
                            currFiltCurrDur,
                            scalings,
                            sigmaSrEffective,
                            sigmaNrEffective,
                            channelToNodeMap,
                            filterType=filterType,
                            exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                        )
                        if p.showDeltaPerNode:
                            metricsData[filterType.to_str()][idxMC, idxDur, :] = currMetrics
                        else:
                            metricsData[filterType.to_str()][idxMC, idxDur] = currMetrics
        
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
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
from dataclasses import dataclass, field
sys.path.append('C:\\Users\\pdidier\\Dropbox\\PC\\Documents\\sounds-phd')
from _general_fcts.class_methods.dataclass_methods import *

@dataclass
class ScriptParameters:
    signalType: str = 'speech'  
    # ^^^ 'speech', 'noise_real', 'noise_complex', ...
    #     ... 'interrupt_noise_real', 'interrupt_noise_complex'.
    interruptionDuration: float = 0.1  # seconds
    interruptionPeriod: float = 0.5  # seconds
    targetSignalSpeechFile: str = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
    nSensors: int = 3
    nNodes: int = 3
    Mk: list[int] = field(default_factory=lambda: None)  # if None, randomly assign sensors to nodes
    selfNoisePower: float = 1
    minDuration: float = 1
    maxDuration: float = 10
    nDurationsBatch: int = 30
    fs: float = 8e3
    nMC: int = 10
    exportFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/figs/for20230823marcUpdate'
    taus: list[float] = field(default_factory=lambda: [2.])
    b: float = 0.1  # factor for determining beta from tau (online processing)
    toCompute: list[str] = field(default_factory=lambda: [
        'mwf_batch',
        'gevdmwf_batch',
        'danse_sim_batch',
        'gevddanse_sim_batch',
    ])
    seed: int = 0
    wolaParams: WOLAparameters = WOLAparameters(
        nfft=1024,
        hop=512,
        fs=fs,
        betaExt=.9,  # if ==0, no extra fusion vector relaxation
        startExpAvgAfter=2,  # frames
        startFusionExpAvgAfter=2,  # frames
        singleFreqBinIndex=99,  # if not None, only consider the freq. bin at this index in WOLA-DANSE
    )
    VADwinLength: float = 0.02  # seconds
    VADenergyDecrease_dB: float = 10  # dB
    # Booleans vvvv
    randomDelays: bool = False
    showDeltaPerNode: bool = False
    useBatchModeFusionVectorsInOnlineDanse: bool = False
    ignoreFusionForSSNodes: bool = True  # in DANSE, ignore fusion vector for single-sensor nodes
    exportFigures: bool = True
    verbose: bool = True
    useVAD: bool = True  # use VAD for online processing of nonsstationary signals
    loadVadIfPossible: bool = True  # if True, load VAD from file if possible
    # Strings vvvv
    vadFilesFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/vad_files'

    def __post_init__(self):
        if any(['wola' in t for t in self.toCompute]) and\
            'complex' in self.signalType:
                raise ValueError('WOLA not implemented for complex-valued signals')
        self.durations = np.linspace(
            self.minDuration,
            self.maxDuration,
            self.nDurationsBatch
        )
        if self.minDuration <= self.interruptionPeriod:
            raise ValueError('`minDuration` should be > `interruptionPeriod`')

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
            if 'online' in filterType or 'wola' in filterType:
                # Compute number of iterations
                if 'online' in filterType:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.nfft) # divide by frame size
                elif 'wola' in filterType:
                    nIter = int(np.amax(p.durations) *\
                        wolaParamsCurr.fs / wolaParamsCurr.hop) - 1  # divide by hop size
                
                if p.showDeltaPerNode:
                    toDict.append((filterType, np.zeros((p.nMC, nIter, len(p.taus), p.nNodes))))
                else:
                    toDict.append((filterType, np.zeros((p.nMC, nIter, len(p.taus)))))
            else:
                if p.showDeltaPerNode:
                    toDict.append((filterType, np.zeros((p.nMC, len(p.durations), p.nNodes))))
                else:
                    toDict.append((filterType, np.zeros((p.nMC, len(p.durations)))))
        metricsData = dict(toDict)

        for idxMC in range(p.nMC):
            print(f'Running Monte-Carlo iteration {idxMC+1}/{p.nMC}')

            # Get scalings
            scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors)
            # Get clean signals
            nSamplesMax = int(np.amax(p.durations) * wolaParamsCurr.fs)
            cleanSigs, _, vad = get_clean_signals(
                p,
                scalings,
                wolaParamsCurr.fs,
                maxDelay=0.1
            )
            if vad is not None:
                sigma_sr = np.sqrt(
                    np.mean(
                        np.abs(
                            cleanSigs[np.squeeze(vad).astype(bool), :]
                        ) ** 2,
                        axis=0
                    )
                )
            else:
                sigma_sr = np.sqrt(np.mean(np.abs(cleanSigs) ** 2, axis=0))
            sigma_sr_wola = get_sigma_wola(cleanSigs, wolaParamsCurr)  # TODO: implement for VAD cases

            # Generate noise signals
            sigma_nr = np.zeros(p.nSensors)
            if wolaParamsCurr.singleFreqBinIndex is not None:
                sigma_nr_wola = np.zeros((1, p.nSensors))
            else:
                sigma_nr_wola = np.zeros((wolaParamsCurr.nPosFreqs, p.nSensors))
            if np.iscomplex(cleanSigs).any():
                noiseSignals = np.zeros((nSamplesMax, p.nSensors), dtype=np.complex128)
            else:
                noiseSignals = np.zeros((nSamplesMax, p.nSensors))
            for n in range(p.nSensors):
                # Generate random sequence with unit power
                if np.iscomplex(cleanSigs).any():
                    randSequence = np.random.normal(size=nSamplesMax) +\
                        1j * np.random.normal(size=nSamplesMax)
                    
                else:
                    randSequence = np.random.normal(size=nSamplesMax)
                # Make unit power
                randSequence /= np.sqrt(np.mean(np.abs(randSequence) ** 2))
                # Scale to desired power
                noiseSignals[:, n] = randSequence * np.sqrt(p.selfNoisePower)
                # Check power
                sigma_nr[n] = np.sqrt(np.mean(np.abs(noiseSignals[:, n]) ** 2))
                if np.abs(sigma_nr[n] ** 2 - p.selfNoisePower) > 1e-6:
                    raise ValueError(f'Noise signal power is {sigma_nr[n] ** 2} instead of {p.selfNoisePower}')
                sigma_nr_wola[:, n] = get_sigma_wola(
                    noiseSignals[:, n],
                    wolaParamsCurr
                )

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
            for filterType in p.toCompute:
                currFilters = allFilters[filterType]
                if 'online' in filterType or 'wola' in filterType:
                    for idxTau in range(len(p.taus)):
                        if 'online' in filterType:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :],
                                scalings,
                                sigma_sr,
                                sigma_nr,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                            )
                        elif 'wola' in filterType:
                            currMetrics = get_metrics(
                                p.nSensors,
                                currFilters[idxTau, :, :, :, :],
                                scalings,
                                sigma_sr_wola,
                                sigma_nr_wola,
                                channelToNodeMap,
                                filterType=filterType,
                                exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                            )
                        if p.showDeltaPerNode:
                            metricsData[filterType][idxMC, :, idxTau, :] = currMetrics
                        else:
                            metricsData[filterType][idxMC, :, idxTau] = currMetrics
                else:
                    for idxDur in range(len(p.durations)):
                        currFiltCurrDur = currFilters[idxDur, :, :]
                        currMetrics = get_metrics(
                            p.nSensors,
                            currFiltCurrDur,
                            scalings,
                            sigma_sr,
                            sigma_nr,
                            channelToNodeMap,
                            filterType=filterType,
                            exportDiffPerFilter=p.showDeltaPerNode  # export all differences individually
                        )
                        if p.showDeltaPerNode:
                            metricsData[filterType][idxMC, idxDur, :] = currMetrics
                        else:
                            metricsData[filterType][idxMC, idxDur] = currMetrics
        
        if p.exportFigures:
            # Plot results
            figTitleSuffix = ''
            if any(['danse' in t for t in p.toCompute]):
                figTitleSuffix += f'$\\beta_{{\\mathrm{{EXT}}}} = {np.round(betaExtCurr, 4)}$'
            if wolaParamsCurr.singleFreqBinIndex is not None and\
                any(['wola' in t for t in p.toCompute]):
                figTitleSuffix += f", WOLA's {wolaParamsCurr.singleFreqBinIndex + 1}-th freq. bin"
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
                fname += f'_betaExt_0p{int(betaExtCurr * 1000)}'
                for t in p.toCompute:
                    fname += f'_{t}'
                if not Path(p.exportFolder).is_dir():
                    Path(p.exportFolder).mkdir(parents=True, exist_ok=True)
                fig.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
        
            # Wait for button press to close the figures
            plt.waitforbuttonpress()
            plt.close('all')

    return metricsData


if __name__ == '__main__':
    sys.exit(main())
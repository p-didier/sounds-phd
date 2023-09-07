import os
import sys
import copy
import time
import resampy
import numpy as np
import soundfile as sf
from numba import njit
import simpleaudio as sa
from utils.common import *
import pyroomacoustics as pra
from utils.batch_mode import *
from utils.online_mwf import *
import matplotlib.pyplot as plt
from utils.online_danse import *
from dataclasses import dataclass, field
sys.path.append('./danse')
from danse_toolbox.d_eval import get_metrics as get_metrics_from_toolbox

@dataclass
class FilterType:
    danse: bool = False
    nodeUpdatingStrategy: str = 'seq'  # 'seq' (sequential) or 'sim' (simultaneous)
    gevd: bool = False
    wola: bool = False
    online: bool = False
    batch: bool = False

    def indiv_frames(self):
        """Return True if time frames are to be treated individually."""
        return (self.online or self.wola) and not self.batch

    def to_str(self):
        """Return string representation of filter type."""
        if self.danse:
            s = f'danse_{self.nodeUpdatingStrategy}'
        else:
            s = 'mwf'
        if self.gevd:
            s += '_gevd'
        if self.wola:
            s += '_wola'
        if self.online:
            s += '_online'
        if self.batch:
            s += '_batch'
        return s


@dataclass
class RoomParameters:
    dimensions: np.ndarray = np.array([5, 5, 5])
        # ^^^ [x, y, z] dimensions of the room (in meters)
    nTargetSources: int = 1
    nInterferingSources: int = 1
    reverberationTime: float = 0.0  # seconds
    rirDuration: float = 0.1  # seconds
    #
    interferingSourceType: str = 'noise'  # 'babble' or 'noise'
    baseSNR: float = 0  # dB
    selfNoisePower: float = 0.01  # power of self-noise (per sensor)
    #
    minDistWalls: float = 0.5  # meters
    nodeDiameter: float = 0.1  # meters

    def __post_init__(self):
        # vvvv TEMPORARY WARNINGS vvvv
        if self.nTargetSources > 1:  # TODO: implement multiple target sources
            raise NotImplementedError('Multiple target sources not implemented yet')
        if self.nInterferingSources > 1:  # TODO: implement multiple interfering sources
            raise NotImplementedError('Multiple interfering sources not implemented yet')
        if self.interferingSourceType == 'babble':  # TODO: implement babble noise
            raise NotImplementedError('Babble noise not implemented yet')
        
        if len(self.dimensions) != 3:
            raise ValueError('`dimensions` should be a list of length 3')
        if self.nTargetSources < 1:
            raise ValueError('`nTargetSources` should be >= 1')
        # Check that distance to walls is large enough
        if np.any(np.array(self.dimensions) < 2 * self.minDistWalls):
            raise ValueError('`minDistWalls` is too large compared to `dimensions`')
        # Check that node diameter is small enough
        if np.any(np.array(self.dimensions) < self.nodeDiameter):
            raise ValueError('`nodeDiameter` is too large compared to `dimensions`')


@dataclass
class ScriptParameters:
    targetSignalType: str = 'speech'  
    # ^^^ 'speech', 'noise_real', 'noise_complex', ...
    #     ... 'interrupt_noise_real', 'interrupt_noise_complex'.
    interruptionDuration: float = 1  # seconds
    interruptionPeriod: float = 2  # seconds
    targetSignalSpeechFile: str = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
    minDuration: float = 3
    maxDuration: float = 10
    nDurationsBatch: int = 30
    # Related to acoustic scenario vvv
    nSensors: int = 3
    nNodes: int = 3
    Mk: list[int] = field(default_factory=lambda: None)  # if None, randomly assign sensors to nodes
    rank1model: bool = True  # if False, use actual RIRs model
    roomParams: RoomParameters = RoomParameters()
    selfNoisePower: float = 1
    localizedNoiseSource: bool = False  # if True, the noise component is also
        # composed of a localized source (i.e., not only self-noise), with
        # random scalings (TODO: + possibly random delays).
    localizedNoiseSourcePower: float = 1  # power of localized noise source
    # Other vvv
    fs: float = 8e3
    nMC: int = 10
    exportFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/figs/for20230823marcUpdate'
    taus: list[float] = field(default_factory=lambda: [2.])
    b: float = 0.1  # factor for determining beta from tau (online processing)
    toComputeStrings: list[str] = field(default_factory=lambda: [
        'mwf_batch',
        'gevdmwf_batch',
        'danse_sim_batch',
        'gevddanse_sim_batch',
    ])
    toCompute: list[FilterType] = field(default_factory=lambda: None)
    seed: int = 0
    wolaParams: WOLAparameters = WOLAparameters(
        fs=fs,
    )
    VADwinLength: float = 0.02  # seconds
    VADenergyDecrease_dB: float = 40  # dB
    # Booleans vvvv
    randomDelays: bool = False
    showDeltaPerNode: bool = False
    useBatchModeFusionVectorsInOnlineDanse: bool = False
    ignoreFusionForSSNodes: bool = False  # in DANSE, ignore fusion vector for single-sensor nodes
    exportFigures: bool = True
    verbose: bool = True
    useVAD: bool = True  # use VAD for online processing of nonsstationary signals
    loadVadIfPossible: bool = True  # if True, load VAD from file if possible
    speechPlots: bool = False  # if True, plots waveforms and metrics for speech signals
    exportFilteredSignals: bool = False  # if True, export filtered signals
    # Strings vvvv
    vadFilesFolder: str = '97_tests/06_pure_linalg/20230630_rank1model/vad_files'

    def __post_init__(self):
        if any(['wola' in t for t in self.toComputeStrings]) and\
            'complex' in self.targetSignalType:
                raise ValueError('WOLA not implemented for complex-valued signals')
        self.durations = np.linspace(
            self.minDuration,
            self.maxDuration,
            self.nDurationsBatch,
            endpoint=True   # include `maxDuration`
        )
        if self.wolaParams.fs != self.fs:
            self.wolaParams.fs = self.fs
        if 'interrupt' in self.targetSignalType and\
            self.minDuration <= self.interruptionPeriod:
            raise ValueError('`minDuration` should be > `interruptionPeriod`')
        # Convert strings to FilterType objects
        self.toCompute = []
        for t in self.toComputeStrings:
            self.toCompute.append(string_to_filtertype(t))


def string_to_filtertype(string):
    """Convert string to `FilterType` object."""
    currType = FilterType()
    if 'danse' in string:
        currType.danse = True  # default: False
        if 'sim' in string:
            currType.nodeUpdatingStrategy = 'sim'
        else:
            currType.nodeUpdatingStrategy = 'seq'

    if 'gevd' in string:
        currType.gevd = True  # default: False
    if 'wola' in string:
        if 'online' in string:
            raise ValueError('`wola` and `online` cannot be combined')
        currType.wola = True  # default: False
    elif 'online' in string:
        if 'wola' in string:
            raise ValueError('`wola` and `online` cannot be combined')
        currType.online = True  # default: False
    if 'batch' in string:
        if 'online' in string:
            raise ValueError('`batch` and `online` cannot be combined')
        currType.batch = True  # default: False
    return currType


def generate_signals(p: ScriptParameters):
    """Generate signals."""
    if p.rank1model:
        return generate_signals_rank1model(p)
    else:
        return generate_signals_actualRIRs(p)


def generate_signals_actualRIRs(p: ScriptParameters):

    pp = p.roomParams  # shorter name
    
    # Generate source positions
    Ns = pp.nTargetSources + pp.nInterferingSources
    sourcesCoords = np.zeros((Ns, 3))
    for k in range(Ns):
        sourcesCoords[k, :] = np.random.uniform(
            low=pp.minDistWalls,
            high=pp.dimensions - pp.minDistWalls,
            size=3
        )
    # targetCoords = sourcesCoords[:pp.nTargetSources, :]
    # interfCoords = sourcesCoords[pp.nTargetSources:, :]

    # Generate node and sensor positions
    nodeCoords = np.zeros((p.nNodes, 3))
    sensorCoords = [np.zeros((p.Mk[k], 3)) for k in range(p.nNodes)]
    for k in range(p.nNodes):
        delta = pp.nodeDiameter / 2 + pp.minDistWalls
        nodeCoords[k, :] = np.random.uniform(
            low=delta,
            high=pp.dimensions - delta,
            size=3
        )
        # Generate sensor positions
        for m in range(p.Mk[k]):
            localCoords = np.random.uniform(
                low=-pp.nodeDiameter / 2,
                high=pp.nodeDiameter / 2,
                size=3
            )
            # Center sensor positions around node position
            sensorCoords[k][m, :] = nodeCoords[k, :] + localCoords


    # Create a room using pyroomacoustics
    room = pra.ShoeBox(pp.dimensions, fs=p.fs, t0=0.0)

    # Add sources and microphones to the room
    for n in range(Ns):
        room.add_source(sourcesCoords[n, :])
    for k in range(p.nNodes):
        for m in range(p.Mk[k]):
            room.add_microphone(sensorCoords[k][m, :])

    # Compute RIRs
    room.compute_rir()
    rirs = room.rir
        # ^^^ [M x 1] list of [Ns x 1] lists of [nSamples x 1] arrays

    # Apply RIRs to sources
    cleanSigs, _, vad = get_clean_signals_fromRIRs(p, rirs)

    return cleanSigs, vad


def generate_signals_rank1model(p: ScriptParameters):
    """Generate signals using rank-1 model."""
    
    def _gen_noise(n, power, complexType=False):
        """Helper function: generate noise signal."""
        # Generate random sequence with unit power
        if complexType:
            randSequence = np.random.normal(size=n) +\
                1j * np.random.normal(size=n)
        else:
            randSequence = np.random.normal(size=n)
        # Make unit power
        randSequence /= np.sqrt(np.mean(np.abs(randSequence) ** 2))
        # Scale to desired power
        randSequence *= np.sqrt(power)
        return randSequence

    # Get scalings
    if 'complex' in p.targetSignalType:
        scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors) +\
            1j * np.random.uniform(low=0.5, high=1, size=p.nSensors)
    else:
        scalings = np.random.uniform(low=0.5, high=1, size=p.nSensors)
    # Get clean signals
    nSamplesMax = int(np.amax(p.durations) * p.wolaParams.fs)
    cleanSigs, _, vad = get_clean_signals(
        p,
        scalings,
        maxDelay=0.1
    )
    if vad is not None:
        sigmaSr = np.sqrt(
            np.mean(
                np.abs(
                    cleanSigs[np.squeeze(vad).astype(bool), :]
                ) ** 2,
                axis=0
            )
        )
        sigmaSrWOLA = get_sigma_wola(
            cleanSigs[np.squeeze(vad).astype(bool), :],
            p.wolaParams
        )
    else:
        sigmaSr = np.sqrt(np.mean(np.abs(cleanSigs) ** 2, axis=0))
        sigmaSrWOLA = get_sigma_wola(cleanSigs, p.wolaParams)

    # Generate noise signals
    sigmaSelfNoise = np.zeros(p.nSensors)
    sigmaLocNoise = np.zeros(p.nSensors)
    sigmaAllNoise = np.zeros(p.nSensors)
    if p.wolaParams.singleFreqBinIndex is not None:
        sigmaSelfNoiseWOLA = np.zeros((1, p.nSensors))
        sigmaLocNoiseWOLA = np.zeros((1, p.nSensors))
        sigmaAllNoiseWOLA = np.zeros((1, p.nSensors))
    else:
        sigmaSelfNoiseWOLA = np.zeros((p.wolaParams.nPosFreqs, p.nSensors))
        sigmaLocNoiseWOLA = np.zeros((p.wolaParams.nPosFreqs, p.nSensors))
        sigmaAllNoiseWOLA = np.zeros((p.wolaParams.nPosFreqs, p.nSensors))
    if np.iscomplex(cleanSigs).any():
        noiseSignals = np.zeros((nSamplesMax, p.nSensors), dtype=np.complex128)
    else:
        noiseSignals = np.zeros((nSamplesMax, p.nSensors))
    
    # If localized noise source, generate random sequence with desired power
    if p.localizedNoiseSource:
        localizedNoiseSignal = _gen_noise(
            nSamplesMax,
            p.localizedNoiseSourcePower,
            complexType=np.iscomplex(cleanSigs).any()
        )
        # Get powers
        sigmaLocNoise, sigmaLocNoise = _get_powers(
            localizedNoiseSignal,
            p.wolaParams
        )
        if 'complex' in p.targetSignalType:
            scalingsNoise = np.random.uniform(0.5, 1, size=p.nSensors) +\
                1j * np.random.uniform(0.5, 1, size=p.nSensors)
        else:
            scalingsNoise = np.random.uniform(0.5, 1, size=p.nSensors)
    else:
        scalingsNoise = None


    # Generate noise
    for n in range(p.nSensors):
        # Generate self-noise
        selfNoise = _gen_noise(  # one independent noise signal per sensor
            nSamplesMax,
            p.selfNoisePower,
            complexType=np.iscomplex(cleanSigs).any()
        )
        # Get powers
        sigmaSelfNoise[n], sigmaSelfNoiseWOLA[:, n] = _get_powers(
            selfNoise,
            p.wolaParams
        )
        # Add to noise signal
        noiseSignals[:, n] = selfNoise
        
        if p.localizedNoiseSource:
            # Add to noise signal
            noiseSignals[:, n] += localizedNoiseSignal * scalingsNoise[n]
        # Get powers
        sigmaAllNoise[n], sigmaAllNoiseWOLA[:, n] = _get_powers(
            noiseSignals[:, n],
            p.wolaParams
        )

    # Group powers
    powers = {
        'clean': sigmaSr,
        'cleanWOLA': sigmaSrWOLA,
        'selfNoise': sigmaSelfNoise,
        'selfNoiseWOLA': sigmaSelfNoiseWOLA,
        'locNoise': sigmaLocNoise,
        'locNoiseWOLA': sigmaLocNoiseWOLA,
        'allNoise': sigmaAllNoise,
        'allNoiseWOLA': sigmaAllNoiseWOLA
    }

    return cleanSigs, noiseSignals, scalings, scalingsNoise, powers, vad


def compute_filter(
        cleanSigs: np.ndarray,
        noiseOnlySigs: np.ndarray,
        type=FilterType(),
        rank=1,
        channelToNodeMap=None,  # only used for DANSE
        wolaParams: WOLAparameters=WOLAparameters(),
        batchFusionInOnline=False,
        noSSfusion=True,
        verbose=True,
        vad=None,
    ):
    """
    [1] Santiago Ruiz, Toon van Waterschoot and Marc Moonen, "Distributed
    combined acoustic echo cancellation and noise reduction in wireless
    acoustic sensor and actuator networks" - 2022
    """

    # Determine filter type
    if type.gevd:
        filterType = 'gevd'
    else:
        filterType = 'regular'

    if type.danse:  # (GEVD-)DANSE
        kwargsBatch = {
            'x': cleanSigs,
            'n': noiseOnlySigs,
            'referenceSensorIdx': 0,
            'channelToNodeMap': channelToNodeMap,
            'filterType': filterType,
            'rank': rank,
            'verbose': verbose,
            'vad': vad
        }
        if type.nodeUpdatingStrategy == 'sim':
            kwargsBatch['nodeUpdatingStrategy'] = 'simultaneous'
        elif type.nodeUpdatingStrategy == 'seq':
            kwargsBatch['nodeUpdatingStrategy'] = 'sequential'
        else:
            raise ValueError('Invalid node updating strategy')

        # Keyword-arguments for online-mode DANSE function
        kwargsOnline = copy.deepcopy(kwargsBatch)
        kwargsOnline['p'] = wolaParams
        kwargsOnline['ignoreFusionForSSNodes'] = noSSfusion
    
        if batchFusionInOnline:
            wBatchNetWide = run_batch_danse(**kwargsBatch)
            kwargsOnline['batchModeNetWideFilters'] = wBatchNetWide
        
        if type.batch:
            if type.wola:
                kwargsBatch['p'] = wolaParams  # add WOLA parameters
                # kwargsBatch['ignoreFusionForSSNodes'] = noSSfusion  # TODO: maybe!
                w = run_batch_danse_wola(**kwargsBatch)
            else:
                w = run_batch_danse(**kwargsBatch)
        elif type.online:
            w = run_online_danse(**kwargsOnline)
        elif type.wola:
            w = run_wola_danse(**kwargsOnline)

    else:  # (GEVD-)MWF
        kwargs = {
            'x': cleanSigs,
            'n': noiseOnlySigs,
            'filterType': filterType,
            'rank': rank,
            'p': wolaParams,
            'verbose': verbose,
            'vad': vad
        }
        if type.batch:
            if type.wola:
                w = run_batch_mwf_wola(**kwargs)
            else:
                # Delete keyword-arguments that are not needed for batch-mode
                kwargs.pop('p')
                kwargs.pop('verbose')

                w = run_batch_mwf(**kwargs)
        elif type.online:
            w = run_online_mwf(**kwargs)
        elif type.wola:
            w = run_wola_mwf(**kwargs)

    return w


def get_latent_signal(p: ScriptParameters, overridingType=None):
    """Get latent signal (speech or noise (possibly interrupted))."""
    dur = np.amax(p.durations)
    # Load target signal
    vad = None

    signalType = overridingType if overridingType is not None\
        else p.targetSignalType

    if signalType == 'speech':
        latentSignal, fs = sf.read(p.targetSignalSpeechFile)
        if fs != p.wolaParams.fs:
            # Resample
            latentSignal = resampy.resample(
                latentSignal,
                fs,
                p.wolaParams.fs
            )
        
        # Loop signal if needed to reach desired duration
        if dur > latentSignal.shape[0] / p.wolaParams.fs:
            nReps = int(np.ceil(dur * p.wolaParams.fs / latentSignal.shape[0]))
            latentSignal = np.tile(latentSignal, (nReps, 1)).T.flatten()
        # Truncate
        latentSignal = latentSignal[:int(dur * p.wolaParams.fs)]

        if p.useVAD:
            # Get VAD
            vad = get_vad(
                x=latentSignal,
                tw=p.VADwinLength,
                eFactdB=p.VADenergyDecrease_dB,
                fs=p.wolaParams.fs,
                loadIfPossible=p.loadVadIfPossible,
                vadFilesFolder=p.vadFilesFolder,
                verbose=p.verbose
            )
            # Normalize power (including VAD)
            latentSignal /= get_power(
                latentSignal[np.squeeze(vad).astype(bool)]
            )
        else:
            # Normalize power
            latentSignal /= get_power(latentSignal)
    elif 'noise' in signalType:
        if 'complex' in signalType:
            # Generate noise signal (complex-valued)
            latentSignal = np.random.normal(size=int(dur * p.wolaParams.fs)) +\
                1j * np.random.normal(size=int(dur * p.wolaParams.fs))
        else:
            # Generate noise signal (real-valued)
            latentSignal = np.random.normal(size=int(dur * p.wolaParams.fs))
        
        if 'interrupt' in signalType:
            # Add `p.interruptionDuration`-long interruptions every
            # `p.interruptionPeriod` seconds 
            nInterruptions = int(dur / p.interruptionPeriod)
            vad = np.ones((latentSignal.shape[0], 1))
            for k in range(nInterruptions):
                idxStart = int(
                    ((k + 1) * p.interruptionPeriod - p.interruptionDuration) *\
                        p.wolaParams.fs
                )
                idxEnd = int(idxStart + p.interruptionDuration * p.wolaParams.fs)
                latentSignal[idxStart:idxEnd] = 0
                vad[idxStart:idxEnd, 0] = 0
            # Normalize power (using VAD)
            latentSignal /= get_power(latentSignal[np.squeeze(vad).astype(bool)])
            if not p.useVAD:
                vad = None
        else:
            # Normalize power
            latentSignal /= get_power(latentSignal)

    return latentSignal, vad


def get_clean_signals(
        p: ScriptParameters,
        scalings,
        maxDelay=0.1
    ):
    """Get clean signals."""

    # Get latent signal
    latentSignal, vad = get_latent_signal(p)
    
    # Generate clean signals
    mat = np.tile(latentSignal, (p.nSensors, 1)).T
    # Random scalings
    cleanSigs = mat @ np.diag(scalings)  # rank-1 matrix (scalings only)
    # Random delays
    if p.randomDelays:
        for n in range(p.nSensors):
            delay = np.random.randint(0, int(maxDelay * p.wolaParams.fs))
            cleanSigs[:, n] = np.roll(cleanSigs[:, n], delay)

    return cleanSigs, latentSignal, vad


def get_clean_signals_fromRIRs(p: ScriptParameters, rirs):
    """Get clean signals from RIRs."""

    def _convolve(x, h):
        return np.convolve(x, h, mode='full')[:len(x)]

    # Get latent target signal
    latentTargetSignal, vad = get_latent_signal(p)
    # Get latent noise signals
    latentNoiseSignal, _ = get_latent_signal(
        p,
        overridingType=p.roomParams.interferingSourceType
    )
    # Adapt SNR using `p.roomParams.baseSNR`
    latentNoiseSignal *= np.sqrt(
        get_power(latentTargetSignal) / get_power(latentNoiseSignal)
    )
    latentNoiseSignal *= 10 ** (-p.roomParams.baseSNR / 20)

    # Stack latent signals for later use
    latentSignals = np.vstack((
        latentTargetSignal,
        latentNoiseSignal
    ))
    nSamples = latentTargetSignal.shape[0]

    # Get number of sources (shorter name)
    Ns = p.roomParams.nInterferingSources + p.roomParams.nTargetSources

    # Generate clean signals
    micSigs = np.zeros((nSamples, p.nSensors))
    noiseOnlySigs = np.zeros((nSamples, p.nSensors))
    targetOnlySigs = np.zeros((nSamples, p.nSensors))
    selfNoiseSigs = np.zeros((nSamples, p.nSensors))
    for m in range(p.nSensors):
        for s in range(Ns):
            currRir = rirs[m][s]
            # Convolve with RIR
            micSigs[:, m] += _convolve(
                latentSignals[s, :],
                currRir
            )
            if s < p.roomParams.nTargetSources:
                targetOnlySigs[:, m] += _convolve(
                    latentSignals[s, :],
                    currRir
                )
            else:
                noiseOnlySigs[:, m] += _convolve(
                    latentSignals[s, :],
                    currRir
                )
        # Generate self-noise 
        selfNoiseSigs[:, m] = np.random.normal(
            size=nSamples
        ) * np.sqrt(p.roomParams.selfNoisePower)
        # Add self-noise to source signals
        micSigs[:, m] += selfNoiseSigs[:, m]

    # Normalize power with respect to microphone #1
    # FIXME: this is not correct for now [PD~2023.09.05 5PM]
    if vad is None:
        micSigs /= get_power(micSigs[:, 0])
        targetOnlySigs /= get_power(micSigs[:, 0])
        noiseOnlySigs /= get_power(micSigs[:, 0])
        selfNoiseSigs /= get_power(micSigs[:, 0])
    else:
        micSigs /= get_power(micSigs[np.squeeze(vad).astype(bool), 0])
        targetOnlySigs /= get_power(micSigs[np.squeeze(vad).astype(bool), 0])
        noiseOnlySigs /= get_power(micSigs[np.squeeze(vad).astype(bool), 0])
        selfNoiseSigs /= get_power(micSigs[np.squeeze(vad).astype(bool), 0])

    stop = 1




def get_power(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def get_filters(
        cleanSigs,
        noiseSignals,
        channelToNodeMap,
        gevdRank=1,
        toCompute: list[FilterType]=[FilterType()],
        wolaParams: WOLAparameters=WOLAparameters(),
        taus=[2.],
        durations=[1.],
        b=0.1,
        batchFusionInOnline=False,
        noSSfusion=True,
        verbose=True,
        vad=None
    ):
    """Compute filters."""
    kwargs = {
        'channelToNodeMap': channelToNodeMap,
        'rank': gevdRank,
        'wolaParams': wolaParams,
        'verbose': verbose,
        'batchFusionInOnline': batchFusionInOnline,
        'noSSfusion': noSSfusion,
    }
    filters = {}
    for filterType in toCompute:

        if verbose:
            print(f'Computing {filterType.to_str()} filter(s)')

        if filterType.indiv_frames():
            # Only need to compute for longest duration
            # + computing for several tau's
            nSamples = int(np.amax(durations) * wolaParams.fs)
            # noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]
            # kwargs['noisySignals'] = noisySignals
            kwargs['cleanSigs'] = cleanSigs[:nSamples, :]
            kwargs['noiseOnlySigs'] = noiseSignals[:nSamples, :]
            kwargs['vad'] = vad[:nSamples, :] if vad is not None else None
            currFiltsAll = []
            # ^^^ shape: (nTaus, nSensors, nIters, nNodes) for 'online'
            # & (nTaus, nSensors, nIters, nFrequencies, nNodes) for 'wola'
            for idxTau in range(len(taus)):
                # Set beta based on tau
                if filterType.online:
                    normFact = kwargs['wolaParams'].nfft
                else:
                    normFact = kwargs['wolaParams'].hop
                
                kwargs['wolaParams'].betaMwf = np.exp(
                    np.log(b) /\
                        (taus[idxTau] * kwargs['wolaParams'].fs / normFact)
                )
                kwargs['wolaParams'].betaDanse = kwargs['wolaParams'].betaMwf

                currFiltsAll.append(
                    compute_filter(type=filterType, **kwargs)
                )
        
        else:  # <-- batch-mode
            # Only need to compute for a single dummy tau
            # + computing for several p.durations
            currFiltsAll = []
            # ^^^ shape: (nDurations, nSensors, nNodes)
            for dur in durations:
                nSamples = int(np.amax(dur) * wolaParams.fs)
                # noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]
                # kwargs['noisySignals'] = noisySignals
                kwargs['cleanSigs'] = cleanSigs[:nSamples, :]
                kwargs['noiseOnlySigs'] = noiseSignals[:nSamples, :]
                kwargs['vad'] = vad[:nSamples, :] if vad is not None else None

                currFiltsAll.append(
                    compute_filter(type=filterType, **kwargs)
                )
            
            if filterType.danse:
                # Ensure adequate array dimensions
                maxNiter = np.amax([f.shape[1] for f in currFiltsAll])
                for k in range(len(currFiltsAll)):
                    if currFiltsAll[k].shape[1] < maxNiter:
                        # Pad by repeating the converged value as many times
                        # as needed to reach `maxNiter`
                        toPad = currFiltsAll[k][:, -1, :]
                        currFiltsAll[k] = np.concatenate((
                            currFiltsAll[k],
                            toPad[:, np.newaxis, :].repeat(
                                maxNiter - currFiltsAll[k].shape[1],
                                axis=1
                            )
                        ), axis=1)
        filters[filterType.to_str()] = np.array(currFiltsAll)

    return filters


def get_metrics(
        nSensors,
        filters,
        scalings,
        sigma_sr,
        RnnOracle,
        channelToNodeMap,
        filterType: FilterType,
        computeForComparisonWithDANSE=True,
        exportDiffPerFilter=False,
        sigma_nr=None,
    ):
    
    # Compute baseline (FAS + SPF)
    RnnInv = np.linalg.inv(RnnOracle)  # inverse of noise covariance matrix
    # # Round entries up to closest integer
    # RnnInv = np.round(RnnInv)
    if filterType.wola:
        nBins = RnnOracle.shape[0]  # number of frequency bins
        if np.any(np.iscomplex(filters)):
            baseline = np.zeros((nSensors, nSensors, nBins), dtype=np.complex128)
            baseline2 = np.zeros((nSensors, nSensors, nBins), dtype=np.complex128)
        else:
            baseline = np.zeros((nSensors, nSensors, nBins))
            baseline2 = np.zeros((nSensors, nSensors, nBins))

        for m in range(nSensors):
            h = scalings / scalings[m]  # RTF
            # Matched filter
            mf = (RnnInv @ h) / (h.T.conj() @ RnnInv @ h)[:, np.newaxis]
            # Spectral post-filter
            spf = sigma_sr[:, m] ** 2 / (
                sigma_sr[:, m] ** 2 + 1 / (h.T.conj() @ RnnInv @ h)
            )
            spf = np.real_if_close(spf)  # real-valued
            baseline[:, m, :] = (mf * spf[:, np.newaxis]).T  # FAS BF + spectral post-filter

            # mf = h / (h.T.conj() @ h)  # matched filter
            # spf = sigma_sr[:, m] ** 2 / (
            #     sigma_sr[:, m] ** 2 + RnnOracle[:, m, m] / (h.T.conj() @ h)
            # )
            # baseline2[:, m, :] = np.outer(mf, spf)  # FAS BF + spectral post-filter
            # hs = (h.T.conj() @ h) * sigma_sr[:, m] ** 2
            # spf = hs / (hs + sigma_nr[:, m] ** 2)  # spectral post-filter (real-valued)
            # # spf = hs / (hs + RnnOracle[:, m, m])  # spectral post-filter (real-valued)
            # baseline[:, m, :] = np.outer(h / (h.T.conj() @ h), spf)  # FAS BF + spectral post-filter
    else:
        if np.any(np.iscomplex(filters)):
            baseline = np.zeros((nSensors, nSensors), dtype=np.complex128)
            # baseline2 = np.zeros((nSensors, nSensors), dtype=np.complex128)
        else:
            baseline = np.zeros((nSensors, nSensors))
            # baseline2 = np.zeros((nSensors, nSensors))
        for m in range(nSensors):
            h = scalings / scalings[m]  # RTF

            # Matched filter
            mf = (RnnInv @ h) / (h.T.conj() @ RnnInv @ h)
            # Spectral post-filter
            spf = sigma_sr[m] ** 2 / (
                sigma_sr[m] ** 2 + 1 / (h.T.conj() @ RnnInv @ h)
            )
            baseline[:, m] = mf * spf  # FAS BF + spectral post-filter

            # hs = (h.T.conj() @ h) * sigma_sr[m] ** 2
            # # spf = hs / (hs + sigma_nr[m] ** 2)  # spectral post-filter (real-valued)
            # spf = hs / (hs + 1)  # spectral post-filter (real-valued)
            # baseline2[:, m] = h / (h.T.conj() @ h) * spf  # FAS BF + spectral post-filter

    # Compute difference between normalized estimated filters
    # and normalized expected filters
    nFilters = filters.shape[-1]
    diffsPerCoefficient = [None for _ in range(nFilters)]
    currNode = 0
    for k in range(nFilters):
        if filterType.danse:  # DANSE case
            # Determine reference sensor index
            idxRef = np.where(channelToNodeMap == k)[0][0]
            if filterType.online:
                # Keep all iterations (all p.durations)
                currFilt = filters[:, :, k]
            elif filterType.wola and not filterType.batch:
                currFilt = filters[:, :, :, k]
            elif filterType.wola and filterType.batch:
                # Only keep last iteration (converged filter coefficients)
                currFilt = filters[:, -1, :, k]
            else:
                # Only keep last iteration (converged filter coefficients)
                currFilt = filters[:, -1, k]
        else:  # MWF case
            if computeForComparisonWithDANSE:
                if channelToNodeMap[k] == currNode:
                    idxRef = np.where(channelToNodeMap == channelToNodeMap[k])[0][0]
                    if filterType.online:
                        currFilt = filters[:, :, idxRef]
                    elif filterType.wola and not filterType.batch:
                        currFilt = filters[:, :, :, idxRef]
                    elif filterType.wola and filterType.batch:
                        currFilt = filters[:, :, idxRef]
                    else:
                        currFilt = filters[:, idxRef]
                    currNode += 1
                else:
                    diffsPerCoefficient[k] = np.nan
                    continue
            else:
                idxRef = copy.deepcopy(k)
                currFilt = filters[:, k]
        
        if filterType.wola:
            currBaseline = baseline[:, idxRef, :]
            # Average over nodes and frequency bins, keeping iteration index
            if 0:
                import matplotlib.pyplot as plt
                for kk in range(nFilters):
                    fig, axes = plt.subplots(1,1)
                    fig.set_size_inches(8.5, 3.5)
                    if filterType.danse:
                        idxRef = np.where(channelToNodeMap == channelToNodeMap[kk])[0][0]
                    else:
                        idxRef = kk
                    for m in range(nSensors):
                        axes.hlines(np.abs(baseline[m, idxRef, :]), 0, currFilt.shape[1], f'C{m}', label=f'Coefficient {m+1}')
                        axes.plot(np.abs(filters[m, :, :, idxRef]), f'C{m}--')
                    plt.legend()
                    axes.set_title(f'Node {kk+1}')
            
            if filterType.batch:
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline
                ), axis=(0, -1))
            else:
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline[:, np.newaxis, :]
                ), axis=(0, -1))
        else:
            currBaseline = baseline[:, idxRef]            
            if 0:
                import matplotlib.pyplot as plt
                for kk in range(nFilters):
                    fig, axes = plt.subplots(1,1)
                    fig.set_size_inches(8.5, 3.5)
                    idxRef = np.where(channelToNodeMap == channelToNodeMap[kk])[0][0]
                    for m in range(nSensors):
                        axes.hlines(np.abs(baseline[m, idxRef]), 0, currFilt.shape[1], f'C{m}', label=f'Coefficient {m+1}')
                        axes.plot(np.abs(filters[m, :, idxRef]), f'C{m}--')
                    plt.legend()
                    axes.set_title(f'Node {kk+1}')
            
            if len(currFilt.shape) == 2:
                # Average over nodes, keeping iteration index
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline[:, np.newaxis]
                ), axis=0)
            else:
                # Average over nodes
                diffsPerCoefficient[k] = np.mean(np.abs(
                    currFilt - currBaseline
                ))
    diffsPerCoefficient = np.array(diffsPerCoefficient, dtype=object)
    
    # Average absolute difference over filters
    diffsPerCoefficient_noNan = []
    for d in diffsPerCoefficient:
        if isinstance(d, np.ndarray):
            diffsPerCoefficient_noNan.append(d)
        elif not np.isnan(d):
            diffsPerCoefficient_noNan.append(d)

    if exportDiffPerFilter:
        diffs = np.array(diffsPerCoefficient_noNan).T
    else:
        diffs = np.nanmean(diffsPerCoefficient_noNan, axis=0)

    return diffs


def get_sigma_wola(x, params: WOLAparameters):
    """Compute WOLA-based signal power estimate."""
    # Check for multichannel input
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    
    win = get_window(params.winType, params.nfft)
    nIter = int((x.shape[0] - params.nfft) / params.hop) + 1
    sigmasSquared = np.zeros((nIter, params.nfft, x.shape[1]))
    for i in range(nIter):
        idxStart = i * params.hop
        idxEnd = idxStart + params.nfft
        windowedChunk = x[idxStart:idxEnd, :] * win[:, np.newaxis]
        Chunk = np.fft.fft(windowedChunk, axis=0) / np.sqrt(params.hop)
        sigmasSquared[i, :, :] = np.abs(Chunk) ** 2
    
    # Only keep positive frequencies
    sigmasSquared = sigmasSquared[:, :int(params.nfft / 2) + 1, :]

    # Average over iterations and get actual sigma (not squared)
    sigmaWOLA = np.sqrt(np.mean(sigmasSquared, axis=0))
    # Adapt array dimensions
    if params.singleFreqBinIndex is not None:
        sigmaWOLA = sigmaWOLA[params.singleFreqBinIndex, :]
        sigmaWOLA = sigmaWOLA[np.newaxis, :]
    else:
        sigmaWOLA = np.squeeze(sigmaWOLA)
    
    return sigmaWOLA


def get_vad(
        x,
        tw,
        eFactdB,
        fs,
        loadIfPossible=False,
        vadFilesFolder='',
        verbose=True
    ):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Signal.
    tw : int
        Window length in samples.
    eFactdB : float
        Energy factor [dB].
    fs : float
        Sampling frequency in Hz.
    loadIfPossible : bool
        If True, try to load VAD from file.
    vadFilesFolder : str
        Folder where VAD files are stored.

    Returns
    -------
    vad : [N x 1] np.ndarray (bool or int [0 or 1])
        VAD.
    """

    # Compute VAD filename
    vadFilename = f'{vadFilesFolder}/vad_{array_id(x)}_eFactdB_{dot_to_p(np.round(eFactdB, 1))}_Nw_{dot_to_p(np.round(tw, 3))}_fs_{dot_to_p(fs)}.npy'
    # Check if VAD can be loaded from file
    if loadIfPossible and os.path.isfile(vadFilename):
        # Load VAD from file
        if verbose:
            print(f'Loading VAD from file {vadFilename}')
        vad = np.load(vadFilename)
    else:
        # Compute VAD
        vad, _ = oracleVAD(
            x,
            tw=tw,
            eFactdB=eFactdB,
            fs=fs
        )
        np.save(vadFilename, vad)

    return vad

    
def oracleVAD(x, tw=0.02, eFactdB=20, fs=16000):

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = int(tw * fs)
    else:
        Nw = 1

    # Check for multiple channels
    if len(x.shape) > 1:
        nCh = x.shape[1]
    else:
        nCh = 1
        x = x[:, np.newaxis]
    
    # Compute VAD
    oVAD = np.zeros((n, nCh))
    for ch in range(nCh):
        print(f'Computing VAD for channel {ch+1}/{nCh}')
        # Compute VAD threshold
        eFact = 10 ** (eFactdB / 10)
        thrs = np.amax(x[:, ch] ** 2) / eFact
        for ii in range(n):
            # Extract chunk
            idxBeg = int(np.amax([ii - Nw // 2, 0]))
            idxEnd = int(np.amin([ii + Nw // 2, n]))
            # Compute VAD frame
            oVAD[ii, ch] = compute_VAD(x[idxBeg:idxEnd, ch], thrs)

    # Time vector
    t = np.arange(n) / fs

    return oVAD, t


@njit
def compute_VAD(chunk_x,thrs):
    """
    JIT-ed time-domain VAD computation
    
    (c) Paul Didier - 6-Oct-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """
    # Compute short-term signal energy
    energy = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if energy > thrs:
        VADout = 1
    else:
        VADout = 0
    return VADout




def array_id(
        a: np.ndarray, *,
        include_dtype=False,
        include_shape=False,
        algo = 'xxhash'
    ):
    """
    Computes a unique ID for a numpy array.
    From: https://stackoverflow.com/a/64756069

    Parameters
    ----------
    a : np.ndarray
        The array to compute the ID for.
    include_dtype : bool, optional
        Whether to include the dtype in the ID.
    include_shape : bool, optional
    """
    data = bytes()
    if include_dtype:
        data += str(a.dtype).encode('ascii')
    data += b','
    if include_shape:
        data += str(a.shape).encode('ascii')
    data += b','
    data += a.tobytes()
    if algo == 'sha256':
        import hashlib
        return hashlib.sha256(data).hexdigest().upper()
    elif algo == 'xxhash':
        import xxhash
        return xxhash.xxh3_64(data).hexdigest().upper()
    else:
        assert False, algo


def dot_to_p(x):
    """
    Converts a float to a string with a 'p' instead of a '.'.
    """
    return str(x).replace('.', 'p')


def speech_plots(
        p: ScriptParameters,
        allFilters,
        x,
        n,
        vad
    ):
    """Produce plots (waveforms and metrics) for a speech-like
    target signal."""

    dhatTarget = {}   # for target signal(s)
    dhatInterf = {}   # for interfering signal(s)

    for filterType in p.toCompute:
        ref = filterType.to_str()
        currFilters = allFilters[ref]
        if filterType.wola and p.wolaParams.singleFreqBinIndex is None:
            if filterType.batch:
                currFilt = currFilters[-1, :, -1, :, :]  # get max. duration filter
                dhatTarget[ref] = np.zeros((x.shape[0], p.nNodes))
                dhatInterf[ref] = np.zeros((x.shape[0], p.nNodes))
                for k in range(p.nNodes):
                    currFilt_k = currFilt[:, :, k]
                    # Apply filter
                    dhatTarget[ref][:, k] = apply_filter_wola(
                        x,
                        currFilt_k,
                        p.wolaParams
                    )
                    dhatInterf[ref][:, k] = apply_filter_wola(
                        n,
                        currFilt_k,
                        p.wolaParams
                    )
            else:
                nTaus = currFilters.shape[0]
                dhatTarget[ref] = np.zeros((x.shape[0], nTaus, p.nNodes))
                dhatInterf[ref] = np.zeros((x.shape[0], nTaus, p.nNodes))
                for idxTau in range(nTaus):
                    currFilt = currFilters[idxTau, :, :, :, :]
                    for k in range(p.nNodes):
                        currFilt_k = currFilt[:, :, :, k]
                        # Apply filter
                        dhatTarget[ref][:, idxTau, k] = apply_filter_wola(
                            x,
                            currFilt_k,
                            p.wolaParams
                        )
                        dhatInterf[ref][:, idxTau, k] = apply_filter_wola(
                            n,
                            currFilt_k,
                            p.wolaParams
                        )
        elif filterType.online:
            raise NotImplementedError
        elif filterType.batch:
            raise NotImplementedError
        else:
            print('Cannot obtain full-band filtered signals.')

    # Get time-domain signals
    y = x + n  # untreated mic signals
    filteredSignals = {}
    for idx, ref in enumerate(dhatTarget.keys()):
        if len(dhatTarget[ref].shape) == 3:
            for idxTau in range(dhatTarget[ref].shape[1]):
                for k in range(p.nNodes):
                    filteredSignals[f'{ref}_tau{idxTau+1}_node{k+1}'] =\
                        dhatTarget[ref][:, idxTau, k] +\
                        dhatInterf[ref][:, idxTau, k]
        else:
            for k in range(p.nNodes):
                filteredSignals[f'{ref}_node{k+1}'] = dhatTarget[ref][:, k] +\
                        dhatInterf[ref][:, k]

    # Plot signals
    figWaveforms, axes = plt.subplots(1, 1)
    figWaveforms.set_size_inches(8.5, 3.5)
    refSensorIdx = 0
    # Plot waveforms above one another
    delta = np.amax(np.abs(y[:, refSensorIdx])) * 1.5
    for idx, ref in enumerate(dhatTarget.keys()):
        if len(dhatTarget[ref].shape) == 3:
            for idxTau in range(dhatTarget[ref].shape[1]):
                # TODO: redundant with previous for-loops
                # Full filtered signal
                yCurr = dhatTarget[ref][:, idxTau, refSensorIdx] +\
                        dhatInterf[ref][:, idxTau, refSensorIdx]
                axes.plot(
                    yCurr + delta * idx,
                    label=f'{ref} ($\\tau_{{{idxTau + 1}}}$)'
                )
        else:
            # Full filtered signal
            yCurr = dhatTarget[ref][:, refSensorIdx] +\
                    dhatInterf[ref][:, refSensorIdx]
            axes.plot(
                yCurr + delta * idx,
                label=ref
            )
    axes.plot(x[:, refSensorIdx] + delta * len(dhatTarget.keys()), label=f'Target signal (mic {refSensorIdx + 1})')
    axes.plot(y[:, refSensorIdx] + delta * (len(dhatTarget.keys()) + 1), label=f'Microphone signal (mic {refSensorIdx + 1})')
    plt.legend()
    

    # Speech-enhancement metrics
    metricsDicts = dict(
        (ref, [{} for _ in range(p.nNodes)])\
            for ref in dhatTarget.keys()
    )
    # For all filter types, skip first 2.5 seconds (also for batch-mode
    # estimates, to make metrics comparable).
    startIdx = int(2.5 * p.fs)  # (HARDCODED)
    for filterType in p.toCompute:
        ref = filterType.to_str()
        for k in range(p.nNodes):
            if len(dhatTarget[ref].shape) == 3:
                for idxTau in range(dhatTarget[ref].shape[1]):
                    metricsDicts[ref][k] = get_metrics_from_toolbox(
                        clean=x[:, k],
                        noiseOnly=n[:, k],
                        noisy=y[:, k],
                        filtSpeech=dhatTarget[ref][:, idxTau, k],
                        filtNoise=dhatInterf[ref][:, idxTau, k],
                        fs=p.fs,
                        vad=vad,
                        startIdx=startIdx
                    )
            else:
                metricsDicts[ref][k] = get_metrics_from_toolbox(
                    clean=x[:, k],
                    noiseOnly=n[:, k],
                    noisy=y[:, k],
                    filtSpeech=dhatTarget[ref][:, k],
                    filtNoise=dhatInterf[ref][:, k],
                    fs=p.fs,
                    vad=vad,
                    startIdx=startIdx
                )

    # Compute and plot metrics
    figMetrics = plot_metrics_dict(metricsDicts)

    return figWaveforms, figMetrics, filteredSignals


def plot_metrics_dict(metricsDicts: dict, metricsToPlot=['snr', 'stoi']):
    """Plots increase in speech enhancement metrics as bar plots,
    for each node, and each considered algorithm."""

    nSubplots = len(metricsDicts[list(metricsDicts.keys())[0]])  # == nNodes
    fig, axes = plt.subplots(
        len(metricsToPlot),
        nSubplots,
        sharex=True,
        sharey='row'
    )
    fig.set_size_inches(8.5, 3.5)
    for k in range(nSubplots):
        for ii, ref in enumerate(metricsDicts.keys()):  # for each algorithm
            currSet = metricsDicts[ref][k]
            for jj, metric in enumerate(metricsToPlot):
                axes[jj, k].bar(
                    x=ii,
                    height=currSet[metric].after,
                    width=0.5,
                    label=ref
                )
        # Add reference: noisy signal
        currSet = metricsDicts[list(metricsDicts.keys())[0]][k]
        for jj, metric in enumerate(metricsToPlot):
            axes[jj, k].bar(
                x=len(metricsDicts.keys()),
                height=currSet[metric].before,
                width=0.5,
                label='No enhancement'
            )
        
        for jj, metric in enumerate(metricsToPlot):
            axes[jj, k].set_title(f'Node {k+1}, {metric}')
            axes[jj, k].grid()
            axes[jj, k].set_xticklabels([])
            if metric == 'stoi':
                axes[jj, k].set_ylim([0, 1])
    axes[0, 0].legend()
    
    return fig


def apply_filter_wola(y: np.ndarray, h: np.ndarray, p: WOLAparameters):
    """Apply filter in WOLA-domain."""

    # Transform to WOLA-domain
    yWola, _ = to_wola(
        p, y, vad=None, verbose=False
    )
    
    # Apply filter in WOLA-domain to each time-frequency bin
    filteredY = np.zeros(
        (yWola.shape[0], yWola.shape[1]),
        dtype=np.complex128
    )
    for i in range(yWola.shape[0]):
        currFrame = yWola[i, :, :]
        if len(h.shape) == 2:
            currFilt = h.T
            op = 'ij,ij->i'
        else:
            currFilt = h[:, i, :].T
            op = 'ij,ij->i'
        
        filteredY[i, :] = np.einsum(
            op,
            currFrame,
            currFilt.conj()
        )
    
    # Transform back to time-domain
    out = from_wola(p, filteredY)

    # Pad or truncate to original length
    if out.shape[0] < y.shape[0]:
        out = np.concatenate((
            out,
            np.zeros((y.shape[0] - out.shape[0], 1))
        ), axis=0)
    elif out.shape[0] > y.shape[0]:
        out = out[:y.shape[0], :]

    if 0:
        in_line_listen(y[:, 0], p.fs, 3)
        time.sleep(3)
        in_line_listen(out, p.fs, 3)

    return np.squeeze(out)


def in_line_listen(signal, fs, duration):
    """Listen to signal."""
    nSamples = int(duration * fs)
    audio_array = signal[:nSamples] *  32767 / max(abs(signal[:nSamples]))
    audio_array = audio_array.astype(np.int16)
    sa.play_buffer(audio_array, 1, 2, int(8e3))


def normalize_toint16(nparray):
    """Normalizes a NumPy array to integer 16.
    Parameters
    ----------
    nparray : np.ndarray
        Input array to be normalized.

    Returns
    ----------
    nparrayNormalized : np.ndarray
        Normalized array.
    """
    amplitude = np.iinfo(np.int16).max
    nparrayNormalized = (amplitude * nparray / \
        np.amax(nparray) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
    return nparrayNormalized


def get_oracle_noise_covariance_matrices(
        powers: dict,
        scalingsLocNoise=None
    ):
    """Computes the oracle noise covariance matrix from a noise signal."""

    nSensors = powers['selfNoise'].shape[0]
    Rnn = powers['selfNoise'] ** 2 * np.eye(nSensors)
    if scalingsLocNoise is not None:
        Rnn += powers['locNoise'] ** 2 * np.outer(
            scalingsLocNoise, scalingsLocNoise
        )
    
    RnnWOLA = np.zeros(
        (powers['selfNoiseWOLA'].shape[0], nSensors, nSensors),
        dtype=np.complex128
    )
    for kappa in range(powers['selfNoiseWOLA'].shape[0]):
        RnnWOLA[kappa, :, :] = powers['selfNoiseWOLA'][kappa, :] ** 2 * np.eye(nSensors)
        if scalingsLocNoise is not None:
            RnnWOLA[kappa, :, :] += powers['locNoiseWOLA'][kappa, :] ** 2 * np.outer(
                scalingsLocNoise, scalingsLocNoise
            )
    # Re-set as real-valued if possible
    RnnWOLA = np.real_if_close(RnnWOLA)
        
    return Rnn, RnnWOLA


def _get_powers(x, wolaParams: WOLAparameters):
    # Get power
    sigma = np.sqrt(np.mean(np.abs(x) ** 2))
    # Get power for the WOLA case
    sigmaWOLA = get_sigma_wola(x, wolaParams)

    return sigma, sigmaWOLA
from dataclasses import dataclass, field
import sys, warnings, copy
from pathlib import PurePath, Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as sig
# Find path to root folder
if not any("_general_fcts" in s for s in sys.path):
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/_general_fcts')
import class_methods.dataclass_methods as met
from plotting.twodim import plot_side_room


@dataclass
class SamplingRateOffsets:
    """Sampling rate/time offsets class, containing all necessary info for
    applying, estimating, and compensation SROs/STOs"""
    SROsppm: list[float] = field(default_factory=list)     # SROs [ppm] to be applied to each node (taking Node#1 (idx = 0) as reference)
    STOsppm: list[float] = field(default_factory=list)     # STOs [ppm] to be applied to each node (taking Node#1 (idx = 0) as reference)

    def __post_init__(self):
        if len(self.SROsppm) < self.nNodes:
            warnings.warn(f'There are more nodes ({self.nNodes}) than SROs ({len(self.SROsppm)}). Settings last nodes to SRO = 0.')
            self.SROsppm += [0. for _ in range(self.nNodes - len(self.SROsppm))]
        if len(self.SROsppm) > self.nNodes:
            warnings.warn(f'There are less nodes ({self.nNodes}) than SROs ({len(self.SROsppm)}). Discarding last ({len(self.SROsppm) - self.nNodes}) SRO(s).')
            self.SROsppm = self.SROsppm[:self.nNodes]


@dataclass
class ProgramSettings(object):
    """Class for keeping track of global simulation settings"""
    # Signal generation parameters
    acousticScenarioPath: str = ''          # path to acoustic scenario to be used
    signalDuration: float = 1               # total signal duration [s]
    desiredSignalFile: list[str] = field(default_factory=list)            # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)              # list of paths to noise signal file(s)
    baseSNR: int = 0                        # SNR between dry desired signals and dry noise
    referenceSensor: int = 0                # Index of the reference sensor at each node
    stftWinLength: int = 1024               # STFT frame length [samples]
    stftFrameOvlp: float = 0.5              # STFT frame overlap [%]
    stftWin: np.ndarray = np.hanning(stftWinLength)  # STFT window
    # VAD parameters
    VADwinLength: float = 40e-3             # VAD window length [s]
    VADenergyFactor: float = 400            # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    # DANSE parameters
    danseUpdating: str = 'sequential'       # node-updating scheme: "sequential" or "simultaneous"
    initialWeightsAmplitude: float = 1      # maximum amplitude of initial random filter coefficients
    expAvgBeta: float = 0.99                # exponential average constant: Ryy[l] = beta*Ryy[l-1] + (1-beta)*y[l]*y[l]^H
    performGEVD: bool = False               # if True, perform GEVD in DANSE
    GEVDrank: int = 1                       # GEVD rank approximation (only used is <performGEVD> is True)
    computeLocalEstimate: bool = False      # if True, compute also an estimate of the desired signal using only the local sensor observations
    # Broadcasting parameters
    broadcastLength: int = 8                # number of (compressed) signal samples to be broadcasted at a time to other nodes [samples]
    # Speech enhancement metrics parameters
    gammafwSNRseg: float = 0.2              # gamma exponent for fwSNRseg computation
    frameLenfwSNRseg: float = 0.03          # time window duration for fwSNRseg computation [s]
    # SROs parameters
    SROsppm: list[float] = field(default_factory=list)   # sampling rate offsets [ppm]
    compensateSROs: bool = False            # if True, estimate + compensate SROs dynamically
    # Other parameters
    plotAcousticScenario: bool = False      # if true, plot visualization of acoustic scenario. 
    acScenarioPlotExportPath: str = ''      # path to directory where to export the acoustic scenario plot
    randSeed: int = 12345                   # random generator(s) seed

    def __post_init__(self) -> None:
        # Checks on class attributes
        self.stftEffectiveFrameLen = int(self.stftWinLength * (1 - self.stftFrameOvlp))
        if 0. not in self.SROsppm:
            raise ValueError('At least one node should have an SRO of 0 ppm (base sampling frequency).')
        if not isinstance(self.desiredSignalFile, list):
            self.desiredSignalFile = [self.desiredSignalFile]
        if not isinstance(self.noiseSignalFile, list):
            self.noiseSignalFile = [self.noiseSignalFile]
        # Check window constraints
        if not sig.check_NOLA(self.stftWin, self.stftWinLength, self.stftEffectiveFrameLen):
            raise ValueError('Window NOLA contrained is not respected.')
        # Check lengths consistency
        if self.broadcastLength > self.stftWinLength:
            raise ValueError(f'Broadcast length is too large ({self.broadcastLength}, with STFT frame size {self.stftWinLength}).')

        return self

    def __repr__(self):
        path = Path(self.acousticScenarioPath)
        string = f"""
--------- Program settings ---------
Acoustic scenario: '{path.parent.name} : {path.name}'
{self.signalDuration} seconds signals using desired file(s):
\t{[PurePath(f).name for f in self.desiredSignalFile]}
and noise signal file(s):
\t{[PurePath(f).name for f in self.noiseSignalFile]}
with a base SNR btw. dry signals of {self.baseSNR} dB.
------ DANSE settings ------
Exponential averaging constant: beta = {self.expAvgBeta}.
"""
        if self.performGEVD:
            string += f'GEVD with R = {self.GEVDrank}.' 
        if (np.array(self.SROsppm) != 0).any():
            string += f'\n------ SRO settings ------'
            for idxNode in range(len(self.SROsppm)):
                string += f'\nSRO Node {idxNode + 1} = {self.SROsppm[idxNode]} ppm'
                if self.SROsppm[idxNode] == 0:
                    string += ' (base sampling freq.)'
        else:
            string += f'\nPerfectly synchronized network, no SROs'
        string += '\n'
        return string

    def load(self, filename: str):
        return met.load(self, filename)

    def save(self, filename: str):
        met.save(self, filename)


@dataclass
class AcousticScenario(object):
    """Class for keeping track of acoustic scenario parameters"""
    rirDesiredToSensors: np.ndarray     # RIRs between desired sources and sensors
    rirNoiseToSensors: np.ndarray       # RIRs between noise sources and sensors
    desiredSourceCoords: np.ndarray     # Coordinates of desired sources
    sensorCoords: np.ndarray            # Coordinates of sensors
    sensorToNodeTags: np.ndarray        # Tags relating each sensor to its node
    noiseSourceCoords: np.ndarray       # Coordinates of noise sources
    roomDimensions: np.ndarray          # Room dimensions   
    absCoeff: float                     # Absorption coefficient
    samplingFreq: int                   # Sampling frequency
    numNodes: int                       # Number of nodes in network
    distBtwSensors: float               # Distance btw. sensors at one node

    def __post_init__(self):
        self.numDesiredSources = self.desiredSourceCoords.shape[0]
        self.numSensors = self.sensorCoords.shape[0]
        self.numNoiseSources = self.noiseSourceCoords.shape[0]    
        self.numSensorPerNode = np.unique(self.sensorToNodeTags, return_counts=True)[-1]
        return self

    def plot(self):

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(121)
        plot_side_room(ax, self.roomDimensions[0:2], 
                    self.desiredSourceCoords[:, [0,1]], 
                    self.noiseSourceCoords[:, [0,1]], 
                    self.sensorCoords[:, [0,1]], self.sensorToNodeTags)
        ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
        #
        ax = fig.add_subplot(122)
        plot_side_room(ax, self.roomDimensions[1:], 
                    self.desiredSourceCoords[:, [1,2]], 
                    self.noiseSourceCoords[:, [1,2]],
                    self.sensorCoords[:, [1,2]],
                    self.sensorToNodeTags)
        ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')
        # Add info
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        boxText = ''
        for ii in range(self.numNodes):
            for jj in range(self.desiredSourceCoords.shape[0]):
                d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.desiredSourceCoords[jj,:]))
                boxText += f'Node {ii + 1}$\\to$D{jj + 1}={np.round(d, 2)}m\n'
            for jj in range(self.noiseSourceCoords.shape[0]):
                d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.noiseSourceCoords[jj,:]))
                boxText += f'Node {ii + 1}$\\to$N{jj + 1}={np.round(d, 2)}m\n'
        boxText = boxText[:-1]
        ax.text(1.1, 0.9, boxText, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        #
        fig.tight_layout()
        return fig


@dataclass
class EnhancementMeasures:
    """Class for storing speech enhancement metrics values"""
    snr: dict         # Unweighted SNR
    fwSNRseg: dict    # Frequency-weighted segmental SNR
    stoi: dict        # Short-Time Objective Intelligibility


@dataclass
class Signals(object):
    """Class to store data output by the signal generation routine"""
    dryNoiseSources: np.ndarray                         # Dry noise source signals
    drySpeechSources: np.ndarray                        # Dry desired (speech) source signals
    wetIndivNoiseSources: np.ndarray                    # Wet (convolved with RIRs) noise source signals, per indiv. noise source
    wetIndivSpeechSources: np.ndarray                   # Wet (convolved with RIRs) desired source signals, per indiv. desired source
    wetNoise: np.ndarray                                # Wet (convolved with RIRs) noise source signals, all sources mixed
    wetSpeech: np.ndarray                               # Wet (convolved with RIRs) desired source signals, all sources mixed
    sensorSignals: np.ndarray                           # Sensor signals (all sources + RIRs)
    VAD: np.ndarray                                     # Voice Activity Detector (1 = voice presence; 0 = noise only)
    timeVector: np.ndarray                              # Time vector (for time-domain signals)
    sensorToNodeTags: np.ndarray                        # Tags relating each sensor to its node
    desiredSigEst: np.ndarray = np.array([])            # Desired signal(s) estimates for each node, in time-domain -- using full-observations vectors (also data coming from neighbors)
    desiredSigEstLocal: np.ndarray = np.array([])       # Desired signal(s) estimates for each node, in time-domain -- using only local observations (not data coming from neighbors)
    desiredSigEst_STFT: np.ndarray = np.array([])       # Desired signal(s) estimates for each node, in STFT domain -- using full-observations vectors (also data coming from neighbors)
    desiredSigEstLocal_STFT: np.ndarray = np.array([])  # Desired signal(s) estimates for each node, in STFT domain -- using only local observations (not data coming from neighbors)
    stftComputed: bool = False                          # Set to true when the STFTs are computed
    fs: int = 16e3                                      # Sampling frequency [samples/s]
    referenceSensor: int = 0                            # Index of the reference sensor at each node
    timeStampsSROs: np.ndarray = np.array([])           # Time stamps for each node in the presence of the SROs (see ProgramSettings)
    masterClockNodeIdx: int = 0                         # Index of node to be used as "master clock" (0 ppm SRO)

    def __post_init__(self):
        """Defines useful fields for Signals object"""
        localSensorNumbering = np.zeros(len(self.sensorToNodeTags))
        currTag = self.sensorToNodeTags[0]
        count = 0
        for ii in range(len(self.sensorToNodeTags)):
            if self.sensorToNodeTags[ii] != currTag:
                count = 0
                currTag = self.sensorToNodeTags[ii]
            else:
                count += 1
            localSensorNumbering[ii] = count
        self.localSensorNumbering = localSensorNumbering
        #
        self.numNodes = len(np.unique(self.sensorToNodeTags))
        self.numSensors = len(self.sensorToNodeTags)
        _, self.nSensorPerNode = np.unique(self.sensorToNodeTags, return_counts=True)

    def get_all_stfts(self, fs, settings: ProgramSettings):
        """Derives time-domain signals' STFT representations
        given certain settings.

        Parameters
        ----------
        fs : int
            Sampling frequency [samples/s].
        settings : ProgramSettings object
            Settings (contains window, window length, overlap).
        """
        self.wetNoise_STFT, self.freqBins, self.timeFrames = get_stft(self.wetNoise, fs, settings)
        self.wetSpeech_STFT, _, _ = get_stft(self.wetSpeech, fs, settings)
        self.sensorSignals_STFT, _, _ = get_stft(self.sensorSignals, fs, settings)

        self.stftComputed = True

        return self

    def plot_signals(self, nodeIdx, sensorIdx, settings: ProgramSettings):
        """Creates a visual representation of the signals at a particular sensor.
        Parameters
        ----------
        nodeNum : int
            Index of the node to inspect.
        sensorNum : int
            Index of the sensor of nodeNum to inspect.
        settings : ProgramSettings object
            Program settings, containing info that can be included on the plots.
        """
        # Useful variables
        indicesSensors = np.argwhere(self.sensorToNodeTags == nodeIdx + 1)
        effectiveSensorIdx = indicesSensors[sensorIdx]
        # Useful booleans
        desiredSignalsAvailable = len(self.desiredSigEst) > 0

        fig = plt.figure(figsize=(8,4))
        if self.stftComputed:
            ax = fig.add_subplot(1,2,1)
        else:
            ax = fig.add_subplot(1,1,1)
            print('STFTs were not yet computed. Plotting only waveforms.')
        delta = np.amax(np.abs(self.sensorSignals))
        ax.plot(self.timeVector, self.wetSpeech[:, effectiveSensorIdx], label='Desired')
        ax.plot(self.timeVector, self.VAD * np.amax(self.wetSpeech[:, effectiveSensorIdx]) * 1.1, 'k-', label='VAD')
        # ax.plot(self.timeVector, self.wetNoise[:, effectiveSensorIdx] - 2*delta, label='Noise-only')
        ax.plot(self.timeVector, self.sensorSignals[:, effectiveSensorIdx] - 2*delta, label='Noisy')
        if desiredSignalsAvailable:        
            if settings.computeLocalEstimate:
                ax.plot(self.timeVector, self.desiredSigEstLocal[:, nodeIdx] - 4*delta, label='Enhanced (local)')
                deltaNextWaveform = 6*delta
            else:
                deltaNextWaveform = 4*delta
            ax.plot(self.timeVector, self.desiredSigEst[:, nodeIdx] - deltaNextWaveform, label='Enhanced (global)')
        ax.set_yticklabels([])
        ax.set(xlabel='$t$ [s]')
        ax.grid()
        plt.legend(loc=(0.01, 0.5), fontsize=8)
        plt.title(f'Node {nodeIdx + 1}, sensor {sensorIdx + 1} -- $\\beta = {settings.expAvgBeta}$')
        #
        if self.stftComputed:
            # Get color plot limits
            limLow = 20 * np.log10(np.amin([np.amin(np.abs(self.wetSpeech_STFT[:, :, effectiveSensorIdx])), 
                                    np.amin(np.abs(self.wetNoise_STFT[:, :, effectiveSensorIdx])), 
                                    np.amin(np.abs(self.sensorSignals_STFT[:, :, effectiveSensorIdx]))]))
            limLow = np.amax([-100, limLow])  # ensures that pure silences (machine precision zeros) do not bring the limit too low
            limHigh = 20 * np.log10(np.amax([np.amax(np.abs(self.wetSpeech_STFT[:, :, effectiveSensorIdx])), 
                                    np.amax(np.abs(self.wetNoise_STFT[:, :, effectiveSensorIdx])), 
                                    np.amax(np.abs(self.sensorSignals_STFT[:, :, effectiveSensorIdx]))]))
            # Number of subplot rows
            nRows = 3
            if desiredSignalsAvailable:
                nRows += 1
            if not settings.computeLocalEstimate:
                nRows -= 1
            # Plot
            ax = fig.add_subplot(nRows,2,2)     # Wet desired signal
            data = 20 * np.log10(np.abs(np.squeeze(self.wetSpeech_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Desired')
            plt.xticks([])
            # ax = fig.add_subplot(nRows,2,4)     # Wet noise only
            # data = 20 * np.log10(np.abs(np.squeeze(self.wetNoise_STFT[:, :, effectiveSensorIdx])))
            # stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Noise-only')
            # plt.xticks([])
            ax = fig.add_subplot(nRows,2,4)     # Sensor signals
            data = 20 * np.log10(np.abs(np.squeeze(self.sensorSignals_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Noisy')
            if desiredSignalsAvailable:
                plt.xticks([])
                if settings.computeLocalEstimate:
                    ax = fig.add_subplot(nRows,2,6)     # Enhanced signals (local)
                    data = 20 * np.log10(np.abs(np.squeeze(self.desiredSigEstLocal_STFT[:, :, nodeIdx])))
                    lab = 'Enhanced (local)'
                    if settings.performGEVD:
                        lab += f' ($\mathbf{{GEVD}}$ $R$={settings.GEVDrank})'
                    stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], lab)
                    ax.set(xlabel='$t$ [s]')
                    plt.xticks([])
                ax = fig.add_subplot(nRows,2,nRows*2)     # Enhanced signals (global)
                data = 20 * np.log10(np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, nodeIdx])))
                lab = 'Enhanced (global)'
                if settings.performGEVD:
                    lab += f' ($\mathbf{{GEVD}}$ $R$={settings.GEVDrank})'
                stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], lab)
                ax.set(xlabel='$t$ [s]')
            else:
                ax.set(xlabel='$t$ [s]')
        plt.tight_layout()

    def plot_enhanced_stft(self, bestNodeIdx, worstNodeIdx, perf: EnhancementMeasures):
        """Plots the STFT of the enhanced signal (best and worse nodes) side by side.

        Parameters
        ----------
        bestNodeIdx : int
            Index of the best performing node.
        worstNodeIdx : int
            Index of the worst performing node. 
        perf : EnhancementMeasures object
            Signal enhancement evaluation metrics.
        """
        # Get data
        dataBest = np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, bestNodeIdx]))
        dataWorst = np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, worstNodeIdx]))
        # Define plot limits
        limLow  = np.amin(np.concatenate((20*np.log10(dataBest[np.abs(dataBest) > 0]), 20*np.log10(dataWorst[np.abs(dataWorst) > 0])), axis=-1))
        limHigh  = np.amax(np.concatenate((20*np.log10(dataBest[np.abs(dataBest) > 0]), 20*np.log10(dataWorst[np.abs(dataWorst) > 0])), axis=-1))

        fig = plt.figure(figsize=(10,4))
        # Best node
        ax = fig.add_subplot(121)
        stft_subplot(ax, self.timeFrames, self.freqBins, 20*np.log10(dataBest), [limLow, limHigh])
        plt.title(f'Best: N{bestNodeIdx + 1} ({np.round(perf.stoi[f"Node{bestNodeIdx + 1}"][0] * 100, 2)}% STOI)')
        plt.xlabel('$t$ [s]')
        # Worst node
        ax = fig.add_subplot(122)
        colorb = stft_subplot(ax, self.timeFrames, self.freqBins, 20*np.log10(dataWorst), [limLow, limHigh])
        plt.title(f'Worst: N{worstNodeIdx + 1} ({np.round(perf.stoi[f"Node{worstNodeIdx + 1}"][0] * 100, 2)}% STOI)')
        plt.xlabel('$t$ [s]')
        colorb.set_label('[dB]')
        return fig

    def export_wav(self, folder):
        """Exports the enhanced, noisy, and desired signals as WAV files.

        Parameters
        ----------
        folder : str
            Folder where to create the "wav" folder where the files are to be exported.

        Returns
        ----------
        fnames : dict
            Full paths of exported files, sorted by type.
        """
        # Check path validity
        if not Path(f'{folder}/wav').is_dir():
            Path(f'{folder}/wav').mkdir()
            print(f'Created .wav export folder "{folder}/wav".')
        fname_noisy    = []
        fname_desired  = []
        fname_enhanced = []
        for idxNode in range(self.numNodes):
            if idxNode == 1:
                idxSensor = self.referenceSensor
            else:
                idxSensor = self.referenceSensor + np.sum(self.nSensorPerNode[:idxNode])
            #
            fname_noisy.append(f'{folder}/wav/noisy_N{idxNode + 1}_Sref{self.referenceSensor + 1}.wav')
            data = normalize_toint16(self.sensorSignals[:, idxSensor])
            wavfile.write(fname_noisy[-1], int(self.fs), data)
            #
            fname_desired.append(f'{folder}/wav/desired_N{idxNode + 1}_Sref{self.referenceSensor + 1}.wav')
            data = normalize_toint16(self.wetSpeech[:, idxSensor])
            wavfile.write(fname_desired[-1], int(self.fs), data)
            #
            if len(self.desiredSigEst) > 0:  # if enhancement has been performed
                fname_enhanced.append(f'{folder}/wav/enhanced_N{idxNode + 1}.wav')
                data = normalize_toint16(self.desiredSigEst[:, idxNode])
                wavfile.write(fname_enhanced[-1], int(self.fs), data)
        print(f'Signals exported in folder "{folder}/wav/".')
        # WAV files names dictionary
        fnames = dict([('Noisy', fname_noisy), ('Desired', fname_desired), ('Enhanced', fname_enhanced)])
        return fnames


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
    nparrayNormalized = (amplitude*nparray/np.amax(nparray)).astype(np.int16)
    return nparrayNormalized


@dataclass
class Results:
    """Class for storing simulation results"""
    signals: Signals = field(init=False)                       # all signals involved in run
    enhancementEval: EnhancementMeasures = field(init=False)   # speech enhancement evaluation metrics
    acousticScenario: AcousticScenario = field(init=False)     # acoustic scenario considered

    def load(self, filename: str):
        return met.load(self, filename)

    def save(self, filename: str, light=False):
        """Exports results as pickle archive
        If `light` is True, export a lighter version (not all results, just the minimum)
        """
        if light:
            mycls = copy.copy(self)
            delattr(mycls, 'signals')
            met.save(mycls, filename)
        else:
            met.save(self, filename)

    def plot_enhancement_metrics(self):
        """Creates a visual representation of DANSE performance results."""
        # Useful variables
        _, sensorCounts = np.unique(self.signals.sensorToNodeTags, return_counts=True)
        barWidth = 1 / np.amax(sensorCounts)
        numNodes = self.signals.desiredSigEst.shape[1]
        
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(1, 3, 1)   # Unweighted SNR
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.snr)
        ax.set(title='$\Delta$SNR (before/after filtering)', ylabel='[dB]')
        ax = fig.add_subplot(1, 3, 2)   # fwSNRseg
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.fwSNRseg)
        ax.set(title='fwSNRseg', ylabel='[dB]')
        ax = fig.add_subplot(1, 3, 3)   # STOI
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.stoi)
        ax.set(title='STOI')
        ax.set_ylim(0,1)
        return fig

        
def get_stft(x, fs, settings: ProgramSettings):
    """Derives time-domain signals' STFT representation
    given certain settings.
    Parameters
    ----------
    x : [N x C] np.ndarray (float)
        Time-domain signal(s).
    fs : int
        Sampling frequency [samples/s].
    settings : ProgramSettings object
        Settings (contains window, window length, overlap)

    Returns
    -------
    out : [Nf x Nt x C] np.ndarray (complex)
        STFT-domain signal(s).
    f : [Nf x 1] np.ndarray (real)
        STFT frequency bins.
    t : [Nt x 1] np.ndarray (real)
        STFT time frames.
    """
    for channel in range(x.shape[-1]):
        f, t, tmp = sig.stft(x[:, channel],
                            fs=fs,
                            window=settings.stftWin,
                            nperseg=settings.stftWinLength,
                            noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
                            return_onesided=True)
        if channel == 0:
            out = np.zeros((tmp.shape[0], tmp.shape[1], x.shape[-1]), dtype=complex)
        out[:, :, channel] = tmp
    return out, f, t


def metrics_subplot(numNodes, ax, barWidth, data):
    """Helper function for <Results.plot_enhancement_metrics()>."""
    xTicks = []
    xTickLabels = []
    for idxNode in range(numNodes):
        numSensors = len(data[f'Node{idxNode + 1}'])
        if not isinstance(data[f'Node{idxNode + 1}'][0], float):
            toPlot = [lst[-1] for lst in data[f'Node{idxNode + 1}']]
        else:
            toPlot = data[f'Node{idxNode + 1}']
        xTicksCurr = idxNode + 1 - (numSensors / 2 + 0.5) * barWidth + np.arange(numSensors) * barWidth
        ax.bar(xTicksCurr, toPlot, width=barWidth, color=f'C{idxNode}', edgecolor='k')
        xTicks += list(xTicksCurr)
        xTickLabels += [f'N{idxNode + 1}S{ii + 1}' for ii in range(numSensors)]
    plt.xticks(xTicks, xTickLabels, fontsize=8)
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()

    

def stft_subplot(ax, t, f, data, vlims, label=''):
    """Helper function for <Signals.plot_signals()>."""
    # Text boxes properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #
    mappable = ax.pcolormesh(t, f / 1e3, data, vmin=vlims[0], vmax=vlims[1])
    ax.set(ylabel='$f$ [kHz]')
    if label != '':
        ax.text(0.025, 0.9, label, fontsize=8, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    ax.yaxis.label.set_size(8)
    cb = plt.colorbar(mappable)
    return cb
from dataclasses import dataclass, field, fields
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
class PrintoutsParameters:
    events_parser: bool = False             # controls printouts in `events_parser()` function
    danseProgress: bool = True              # controls printouts during DANSE processing (indicating loop process in %)
    externalFilterUpdates: bool = False     # controls printouts at DANSE external filter updates (for broadcasting)


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
    samplingFrequency: float = 16000.       # [samples/s] base sampling frequency
    acousticScenarioPath: str = ''          # path to acoustic scenario to be used
    signalDuration: float = 1               # [s] total signal duration
    desiredSignalFile: list[str] = field(default_factory=list)            # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)              # list of paths to noise signal file(s)
    baseSNR: int = 0                        # [dB] SNR between dry desired signals and dry noise
    selfnoiseSNR: int = -50                 # [dB] microphone self-noise SNR
    referenceSensor: int = 0                # Index of the reference sensor at each node
    stftWinLength: int = 1024               # [samples] STFT frame length
    stftFrameOvlp: float = 0.5              # [/100%] STFT frame overlap
    stftWin: np.ndarray = np.hanning(stftWinLength)  # STFT window
    # VAD parameters
    VADwinLength: float = 40e-3             # [s] VAD window length
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    # DANSE parameters
    danseUpdating: str = 'sequential'       # node-updating scheme: "sequential" or "simultaneous"
    timeBtwExternalFiltUpdates: float = 0   # [s] minimum time between 2 consecutive external filter update (i.e. filters that are used for broadcasting)
                                            #  ^---> If 0: equivalent to updating coefficients every `chunkSize * (1 - chunkOverlap)` new captured samples 
    filterDomain: str = 'f'                 # domain in which to compute the DANSE local node filters 
                                            #  ^---> ["t" for time-domain, "f" for frequency- (i.e., STFT-) domain]
    chunkSize: int = 1024                   # [samples] processing chunk size
    chunkOverlap: float = 0.5               # amount of overlap between consecutive chunks; in [0,1) -- full overlap [=1] unauthorized.
    danseWindow: np.ndarray = np.hanning(chunkSize)  # DANSE window for FFT/IFFT operations
    initialWeightsAmplitude: float = 1.     # maximum amplitude of initial random filter coefficients
    expAvg50PercentTime: float = 2.         # [s] Time in the past at which the value is weighted by 50% via exponential averaging
                                            # -- Used to compute beta in, e.g.: Ryy[l] = beta * Ryy[l - 1] + (1 - beta) * y[l] * y[l]^H
    performGEVD: bool = True                # if True, perform GEVD in DANSE
    GEVDrank: int = 1                       # GEVD rank approximation (only used is <performGEVD> is True)
    computeLocalEstimate: bool = False      # if True, compute also an estimate of the desired signal using only local sensor observations
    bypassFilterUpdates: bool = False       # if True, only update covariance matrices, do not update filter coefficients (no adaptive filtering)
    # Broadcasting parameters
    broadcastLength: int = 8                # [samples] number of (compressed) signal samples to be broadcasted at a time to other nodes
    # Speech enhancement metrics parameters
    gammafwSNRseg: float = 0.2              # gamma exponent for fwSNRseg computation
    frameLenfwSNRseg: float = 0.03          # [s] time window duration for fwSNRseg computation
    # SROs parameters
    SROsppm: list[float] = field(default_factory=list)   # [ppm] sampling rate offsets
    compensateSROs: bool = False            # if True, estimate + compensate SROs dynamically
    # Other parameters
    plotAcousticScenario: bool = False      # if true, plot visualization of acoustic scenario. 
    acScenarioPlotExportPath: str = ''      # path to directory where to export the acoustic scenario plot
    randSeed: int = 12345                   # random generator(s) seed
    printouts: PrintoutsParameters = PrintoutsParameters()    # boolean parameters for printouts

    def __post_init__(self) -> None:
        # Create new attributes
        self.stftEffectiveFrameLen = int(self.stftWinLength * (1 - self.stftFrameOvlp))
        self.expAvgBeta = np.exp(np.log(0.5) / (self.expAvg50PercentTime * self.samplingFrequency / self.stftEffectiveFrameLen))
        # Checks on class attributes
        if isinstance(self.SROsppm, int) or isinstance(self.SROsppm, float):
            if self.SROsppm == 0:
                self.SROsppm = [self.SROsppm]
            else:
                raise ValueError('At least one node should have an SRO of 0 ppm (base sampling frequency).')
        elif 0 not in self.SROsppm:
            if self.SROsppm == []:
                print('Empty SROppm parameter: setting SROppm = [0.]')
            else:
                raise ValueError('At least one node should have an SRO of 0 ppm (base sampling frequency).')
        # Adapt formats
        if not isinstance(self.desiredSignalFile, list):
            self.desiredSignalFile = [self.desiredSignalFile]
        if not isinstance(self.noiseSignalFile, list):
            self.noiseSignalFile = [self.noiseSignalFile]
        # Check that SNR is reasonably high
        if self.baseSNR < -3:
            inp = input(f'The base SNR value is low ({self.baseSNR} dB). Continue? [y/n]  ')
            if inp not in ['y', 'Y']:
                raise ValueError('Run aborted.')
        # Check window constraints
        if not sig.check_NOLA(self.stftWin, self.stftWinLength, self.stftEffectiveFrameLen):
            raise ValueError('Window NOLA contrained is not respected.')
        if len(self.stftWin) != self.stftWinLength:
            print(f'!! The specified STFT window length does not match the actual window. Setting `stftWin` as a Hann window of length {self.stftWinLength}.')
            self.stftWin = np.hanning(self.stftWinLength)
        if len(self.danseWindow) != self.chunkSize:
            print(f'!! The specified DANSE chunk size does not match the window length. Setting `danseWindow` as a Hann window of length {self.chunkSize}.')
            self.danseWindow = np.hanning(self.chunkSize)
        # Check lengths consistency
        if self.broadcastLength > self.stftEffectiveFrameLen:
            raise ValueError(f'Broadcast length ({self.broadcastLength}) is too large for STFT frame size {self.stftWinLength} and {int(self.stftFrameOvlp * 100)}% overlap.')
        # Check that chunk overlap makes sense
        if self.chunkOverlap >= 1:
            raise ValueError(f'The processing time chunk overlap cannot be equal to or greater than 1 (current value: {self.chunkOverlap}).')

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
Microphone self-noise SNR: {self.selfnoiseSNR} dB.
------ DANSE settings ------
Exponential averaging 50% attenuation time: tau = {self.expAvg50PercentTime} s.
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

    def load(self, filename: str, silent=False):
        return met.load(self, filename, silent)

    def save(self, filename: str):
        # Save most important parameters as quickly-readable .txt file
        met.save_as_txt(self, filename)
        # Save data as archive
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
    pesq: dict        # Perceptual Evaluation of Speech Quality


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
    fs: np.ndarray = np.array([])                       # Sensor-specific sampling frequencies [samples/s]
    referenceSensor: int = 0                            # Index of the reference sensor at each node
    timeStampsSROs: np.ndarray = np.array([])           # Time stamps for each node in the presence of the SROs (see ProgramSettings for SRO values)
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
        if len(self.desiredSigEst) > 0:
            self.desiredSigEst_STFT, _, _ = get_stft(self.desiredSigEst, fs, settings)
        else:
            print("!! Desired signal (`desiredSigEst`) unavailable -- Cannot compute its STFT.")
        if len(self.desiredSigEstLocal) > 0:
            self.desiredSigEstLocal_STFT, _, _ = get_stft(self.desiredSigEstLocal, fs, settings)
        else:
            print("!! Desired signal (`desiredSigEstLocal`) unavailable -- Cannot compute its STFT.")

        self.stftComputed = True

        return self

    def plot_signals(self, nodeIdx, settings: ProgramSettings, stoiImpLocalVsGlobal=None):
        """Creates a visual representation of the signals at a particular sensor.
        Parameters
        ----------
        nodeNum : int
            Index of the node to inspect.
        settings : ProgramSettings object
            Program settings, containing info that can be included on the plots.
        stoiImpLocalVsGlobal : float
            [dB] Local vs. global signal estimate STOI improvement (only used is `settings.computeLocalEstimate` is True).
        """

        # Disable divide-by-zero warnings
        np.seterr(divide = 'ignore') 

        # Useful variables
        indicesSensors = np.argwhere(self.sensorToNodeTags == nodeIdx + 1)
        effectiveSensorIdx = indicesSensors[settings.referenceSensor]
        # Useful booleans
        desiredSignalsAvailable = len(self.desiredSigEst) > 0

        # PLOT
        fig = plt.figure(figsize=(8,4))
        if self.stftComputed:
            ax = fig.add_subplot(1,2,1)
        else:
            ax = fig.add_subplot(1,1,1)
            print('STFTs were not yet computed. Plotting only waveforms.')

        # -------- WAVEFORMS -------- 
        delta = np.amax(np.abs(self.sensorSignals))
        ax.plot(self.timeVector, self.wetSpeech[:, effectiveSensorIdx], label='Desired')
        ax.plot(self.timeVector, self.VAD * np.amax(self.wetSpeech[:, effectiveSensorIdx]) * 1.1, 'k-', label='VAD')
        # ax.plot(self.timeVector, self.wetNoise[:, effectiveSensorIdx] - 2*delta, label='Noise-only')
        ax.plot(self.timeVector, self.sensorSignals[:, effectiveSensorIdx] - 2*delta, label='Noisy')
        # -------- Desired signal estimate waveform -------- 
        if desiredSignalsAvailable:        
            # -------- Desired signal _local_ estimate waveform -------- 
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
        # Compute time at which the exponential weight decreases to 50% of its initial value
        self.expAvgTau = np.log(0.5) * settings.stftEffectiveFrameLen / (np.log(settings.expAvgBeta) * self.fs[nodeIdx])
        # Set title
        plt.title(f'Node {nodeIdx + 1}, s.{settings.referenceSensor + 1} - $\\beta = {np.round(settings.expAvgBeta, 4)}$ ($\\tau_{{50\%}} = {np.round(self.expAvgTau, 2)}$s)')
        
        # -------- STFTs -------- 
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
            stft_subplot(ax, self.timeFrames, self.freqBins[:, effectiveSensorIdx], data, [limLow, limHigh], 'Desired')
            plt.xticks([])
            if settings.computeLocalEstimate:
                txt = f'{np.round(stoiImpLocalVsGlobal, 2)}'
                if np.round(stoiImpLocalVsGlobal, 2) > 0:
                    txt = f'+{txt}'
                ax.set_title(f'Local vs. global STOI improvement: {txt}')
            ax = fig.add_subplot(nRows,2,4)     # Sensor signals
            data = 20 * np.log10(np.abs(np.squeeze(self.sensorSignals_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins[:, effectiveSensorIdx], data, [limLow, limHigh], 'Noisy')
            
            # -------- Desired signal estimate STFT -------- 
            if desiredSignalsAvailable:
                plt.xticks([])
                # -------- Desired signal _local_ estimate STFT -------- 
                if settings.computeLocalEstimate:
                    ax = fig.add_subplot(nRows,2,6)     # Enhanced signals (local)
                    data = 20 * np.log10(np.abs(np.squeeze(self.desiredSigEstLocal_STFT[:, :, nodeIdx])))
                    lab = 'Enhanced (local)'
                    if settings.performGEVD:
                        lab += f' ($\mathbf{{GEVD}}$ $R$={settings.GEVDrank})'
                    stft_subplot(ax, self.timeFrames, self.freqBins[:, effectiveSensorIdx], data, [limLow, limHigh], lab)
                    ax.set(xlabel='$t$ [s]')
                    plt.xticks([])
                ax = fig.add_subplot(nRows,2,nRows*2)     # Enhanced signals (global)
                data = 20 * np.log10(np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, nodeIdx])))
                lab = 'Enhanced (global)'
                if settings.performGEVD:
                    lab += f' ($\mathbf{{GEVD}}$ $R$={settings.GEVDrank})'
                stft_subplot(ax, self.timeFrames, self.freqBins[:, effectiveSensorIdx], data, [limLow, limHigh], lab)
                ax.set(xlabel='$t$ [s]')
            else:
                ax.set(xlabel='$t$ [s]')
        plt.tight_layout()

        # Re-enable divide-by-zero warnings
        np.seterr(divide = 'warn') 

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

        plotLocalEstimate = len(self.desiredSigEstLocal) > 0
        nSubplotsRows = 1
        if plotLocalEstimate:
            nSubplotsRows = 2

        # Get corresponding sensor indices (for selecting the correct frequency vector)
        bestNodeSensorIdx = np.argwhere(self.sensorToNodeTags == bestNodeIdx + 1)[0]
        worstNodeSensorIdx = np.argwhere(self.sensorToNodeTags == worstNodeIdx + 1)[0]
        
        # Get data
        dataBest = np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, bestNodeIdx]))
        dataWorst = np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, worstNodeIdx]))
        if plotLocalEstimate:
            dataBestLocal = np.abs(np.squeeze(self.desiredSigEstLocal_STFT[:, :, bestNodeIdx]))
            dataWorstLocal = np.abs(np.squeeze(self.desiredSigEstLocal_STFT[:, :, worstNodeIdx]))

        # Define color axis limits
        fullData = np.concatenate((20*np.log10(dataBest[np.abs(dataBest) > 0]),\
                20*np.log10(dataWorst[np.abs(dataWorst) > 0])
                ), axis=-1)
        if plotLocalEstimate:
            fullData = np.concatenate((fullData,\
                20*np.log10(dataBestLocal[np.abs(dataBestLocal) > 0]),\
                20*np.log10(dataWorstLocal[np.abs(dataWorstLocal) > 0])), axis=-1)

        climLow  = np.amin(fullData)
        climHigh = np.amax(fullData)
        # Define frequency (y-) axis limits
        ylimLow  = 0
        ylimHigh  = np.amax(self.freqBins, axis=0)[self.masterClockNodeIdx]

        # Fix 0's before dB operation
        dataBest[np.abs(dataBest) == 0] = np.finfo(float).eps
        dataWorst[np.abs(dataWorst) == 0] = np.finfo(float).eps
        if plotLocalEstimate:
            dataBestLocal[np.abs(dataBestLocal) == 0] = np.finfo(float).eps
            dataWorstLocal[np.abs(dataWorstLocal) == 0] = np.finfo(float).eps

        fig = plt.figure(figsize=(11,5))
        # Best node
        ax = fig.add_subplot(int(nSubplotsRows * 100 + 21))
        stft_subplot(ax, self.timeFrames, self.freqBins[:, bestNodeSensorIdx], 20*np.log10(np.abs(dataBest)), [climLow, climHigh])
        ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
        plt.title(f'Best: N{bestNodeIdx + 1} (STOI = {np.round(perf.stoi[f"Node{bestNodeIdx + 1}"].after, 2)})')
        plt.xlabel('$t$ [s]')
        # Worst node
        ax = fig.add_subplot(int(nSubplotsRows * 100 + 22))
        colorb = stft_subplot(ax, self.timeFrames, self.freqBins[:, worstNodeSensorIdx], 20*np.log10(np.abs(dataWorst)), [climLow, climHigh])
        ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
        plt.title(f'Worst: N{worstNodeIdx + 1} (STOI = {np.round(perf.stoi[f"Node{worstNodeIdx + 1}"].after, 2)})')
        plt.xlabel('$t$ [s]')
        colorb.set_label('[dB]')
        if plotLocalEstimate:
            ax = fig.add_subplot(int(nSubplotsRows * 100 + 23))
            stft_subplot(ax, self.timeFrames, self.freqBins[:, bestNodeSensorIdx], 20*np.log10(np.abs(dataBestLocal)), [climLow, climHigh])
            ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
            plt.title(f'Local estimate: N{bestNodeIdx + 1} (STOI = {np.round(perf.stoi[f"Node{bestNodeIdx + 1}"].afterLocal, 2)})')
            plt.xlabel('$t$ [s]')
            # Worst node
            ax = fig.add_subplot(int(nSubplotsRows * 100 + 24))
            colorb = stft_subplot(ax, self.timeFrames, self.freqBins[:, worstNodeSensorIdx], 20*np.log10(np.abs(dataWorstLocal)), [climLow, climHigh])
            ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
            plt.title(f'Local estimate: N{worstNodeIdx + 1} (STOI = {np.round(perf.stoi[f"Node{worstNodeIdx + 1}"].afterLocal, 2)})')
            plt.xlabel('$t$ [s]')
            colorb.set_label('[dB]')
        plt.tight_layout()

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
        folderShort = met.shorten_path(folder)
        # Check path validity
        if not Path(f'{folder}/wav').is_dir():
            Path(f'{folder}/wav').mkdir()
            print(f'Created .wav export folder ".../{folderShort}/wav".')
        fname_noisy    = []
        fname_desired  = []
        fname_enhanced = []
        for idxNode in range(self.numNodes):
            # Network-wide sensor index
            idxSensor = self.referenceSensor + np.sum(self.nSensorPerNode[:idxNode])
            #
            fname_noisy.append(f'{folder}/wav/noisy_N{idxNode + 1}_Sref{self.referenceSensor + 1}.wav')
            data = normalize_toint16(self.sensorSignals[:, idxSensor])
            wavfile.write(fname_noisy[-1], int(self.fs[idxSensor]), data)
            #
            fname_desired.append(f'{folder}/wav/desired_N{idxNode + 1}_Sref{self.referenceSensor + 1}.wav')
            data = normalize_toint16(self.wetSpeech[:, idxSensor])
            wavfile.write(fname_desired[-1], int(self.fs[idxSensor]), data)
            #
            if len(self.desiredSigEst) > 0:  # if enhancement has been performed
                fname_enhanced.append(f'{folder}/wav/enhanced_N{idxNode + 1}.wav')
                data = normalize_toint16(self.desiredSigEst[:, idxNode])
                wavfile.write(fname_enhanced[-1], int(self.fs[idxSensor]), data)
            #
            if len(self.desiredSigEstLocal) > 0:  # if enhancement has been performed and local estimate computed
                fname_enhanced.append(f'{folder}/wav/enhancedLocal_N{idxNode + 1}.wav')
                data = normalize_toint16(self.desiredSigEstLocal[:, idxNode])
                wavfile.write(fname_enhanced[-1], int(self.fs[idxSensor]), data)
        print(f'Signals exported in folder ".../{folderShort}/wav".')
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
    nparrayNormalized = (amplitude * nparray / np.amax(nparray) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
    return nparrayNormalized


@dataclass
class Results:
    """Class for storing simulation results"""
    signals: Signals = field(init=False)                       # all signals involved in run
    enhancementEval: EnhancementMeasures = field(init=False)   # speech enhancement evaluation metrics
    acousticScenario: AcousticScenario = field(init=False)     # acoustic scenario considered

    def load(self, foldername: str, silent=False):
        return met.load(self, foldername, silent)

        
    def __repr__(self):
        string = f'<Results> object:'
        return string

    def save(self, foldername: str, light=False):
        """Exports results as pickle archive
        If `light` is True, export a lighter version (not all results, just the minimum)
        """
        if light:
            mycls = copy.copy(self)
            delattr(mycls, 'signals')
            met.save(mycls, foldername)
        else:
            met.save(self, foldername)

    def plot_enhancement_metrics(self):
        """Creates a visual representation of DANSE performance results."""
        # Useful variables
        barWidth = 1
        numNodes = self.signals.desiredSigEst.shape[1]
        
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(1, 4, 1)   # Unweighted SNR
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.snr)
        ax.set(title='SNR', ylabel='[dB]')
        plt.legend()
        #
        ax = fig.add_subplot(1, 4, 2)   # fwSNRseg
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.fwSNRseg)
        ax.set(title='fwSNRseg', ylabel='[dB]')
        #
        ax = fig.add_subplot(1, 4, 3)   # STOI
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.stoi)
        ax.set(title='STOI')
        #
        ax = fig.add_subplot(1, 4, 4)   # PESQ
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.pesq)
        ax.set(title='PESQ')

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
    f : [Nf x C] np.ndarray (real)
        STFT frequency bins, per channel (because of different sampling rates).
    t : [Nt x 1] np.ndarray (real)
        STFT time frames.
    """

    if x.ndim == 1:
        x = x[:, np.newaxis]

    for channel in range(x.shape[-1]):
        
        if x.shape[-1] == 1 and isinstance(fs, float):
            fs = [fs]   # from float to list

        fcurr, t, tmp = sig.stft(x[:, channel],
                            fs=fs[channel],
                            window=settings.stftWin,
                            nperseg=settings.stftWinLength,
                            noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
                            return_onesided=True)
        if channel == 0:
            out = np.zeros((tmp.shape[0], tmp.shape[1], x.shape[-1]), dtype=complex)
            f = np.zeros((tmp.shape[0], x.shape[-1]))
        out[:, :, channel] = tmp
        f[:, channel] = fcurr

    # Flatten array in case of single-channel data
    if x.shape[-1] == 1:
        f = np.array([i[0] for i in f])

    return out, f, t


def metrics_subplot(numNodes, ax, barWidth, data):
    """Helper function for <Results.plot_enhancement_metrics()>.
    
    Parameters
    ----------
    numNodes : int
        Number of nodes in network.
    ax : Axes handle
        Axes handle to plot on.
    barWidth : float
        Width of bars for bar plot.
    data : dict of np.ndarrays of floats /or/ dict of np.ndarrays of [3 x 1] lists of floats
        Speech enhancement metric(s) per node.
    """

    flagZeroBar = False     # flag for plotting a horizontal line at `metric = 0`

    for idxNode in range(numNodes):
        if idxNode == 0:    # only add legend labels to first node
            ax.bar(idxNode - barWidth / 6, data[f'Node{idxNode + 1}'].before, width=barWidth / 3, color='tab:orange', edgecolor='k', label='Before')
            ax.bar(idxNode + barWidth / 6, data[f'Node{idxNode + 1}'].after, width=barWidth / 3, color='tab:blue', edgecolor='k', label='After')
        else:
            ax.bar(idxNode - barWidth / 6, data[f'Node{idxNode + 1}'].before, width=barWidth / 3, color='tab:orange', edgecolor='k')
            ax.bar(idxNode + barWidth / 6, data[f'Node{idxNode + 1}'].after, width=barWidth / 3, color='tab:blue', edgecolor='k')

        if data[f'Node{idxNode + 1}'].after < 0 or data[f'Node{idxNode + 1}'].before < 0:
            flagZeroBar = True
    plt.xticks(np.arange(numNodes), [f'N{ii + 1}' for ii in range(numNodes)], fontsize=8)
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    if flagZeroBar:
        ax.hlines(0, - barWidth/2, numNodes - 1 + barWidth/2, colors='k', linestyles='dashed')     # plot horizontal line at `metric = 0`

    

def stft_subplot(ax, t, f, data, vlims, label=''):
    """Helper function for <Signals.plot_signals()>."""
    # Text boxes properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #
    mappable = ax.pcolormesh(t, f / 1e3, data, vmin=vlims[0], vmax=vlims[1], shading='auto')
    ax.set(ylabel='$f$ [kHz]')
    if label != '':
        ax.text(0.025, 0.9, label, fontsize=8, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    ax.yaxis.label.set_size(8)
    cb = plt.colorbar(mappable)
    return cb
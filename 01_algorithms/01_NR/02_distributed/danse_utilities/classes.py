from dataclasses import dataclass, field
from multiprocessing.managers import ValueProxy
from operator import contains
import sys, os
from pathlib import PurePath, Path
from threading import local
import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
from scipy.io import wavfile
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from class_methods import dataclass_methods
from plotting.twodim import plot_side_room
from mySTFT.calc_STFT import calcSTFT

@dataclass
class ProgramSettings(object):
    """Class for keeping track of global simulation settings"""
    # Signal generation parameters
    acousticScenarioPath: str               # path to acoustic scenario CSV file to be used
    signalDuration: float                   # total signal duration [s]
    desiredSignalFile: list                 # list of paths to desired signal file(s)
    noiseSignalFile: list                   # list of paths to noise signal file(s)
    baseSNR: int                            # SNR between dry desired signals and dry noise
    referenceSensor: int = 0                # Index of the reference sensor at each node
    stftWinLength: int = 1024               # STFT frame length [samples]
    stftFrameOvlp: float = 0.5              # STFT frame overlap [%]
    # VAD parametesr
    VADwinLength: float = 40e-3             # VAD window length [s]
    VADenergyFactor: float = 400            # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    # DANSE parameters
    timeBtwConsecUpdates: float = 0.4       # time between consecutive DANSE updates [s]
    initialWeightsAmplitude: float = 1      # maximum amplitude of initial random filter coefficients
    expAvgBeta: float = 0.99                # exponential average constant (Ryy[l] = beta*Ryy[l-1] + (1-beta)*y[l]*y^H[l])
    minNumAutocorrUpdates: int = 10         # minimum number of autocorrelation matrices update before first filter coefficients update
    # useOLA: bool = True                     # if True, use overlap-add process with self.OLAoverlap as frame overlap
    # OLAoverlap: float = 0.5                 # OLA frame overlap (only used if self.useOLA is True)
    # WOLAwindow: str = 'sqrthann'            # WOLA window type (values allowed: "sqrthann")
    # Speech enhancement metrics parameters
    gammafwSNRseg: float = 0.2              # gamma exponent for fwSNRseg computation
    frameLenfwSNRseg: float = 0.03          # time window duration for fwSNRseg computation [s]
    # Other parameters
    plotAcousticScenario: bool = False      # if true, plot visualization of acoustic scenario. 
    acScenarioPlotExportPath: str = ''      # path to directory where to export the acoustic scenario plot
    randSeed: int = 12345                   # random generator(s) seed

    def __post_init__(self) -> None:
        # Checks on class attributes
        if self.acousticScenarioPath[-4:] != '.csv':
            self.acousticScenarioPath += '.csv'
            print('Automatically appended ".csv" to string setting "acousticScenarioPath".')
        self.stftEffectiveFrameLen = int(self.stftWinLength * self.stftFrameOvlp)
        return self

    def __repr__(self):
        string = f"""--------- Program settings ---------
        Acoustic scenario: '{self.acousticScenarioPath[self.acousticScenarioPath.rfind('/', 0, self.acousticScenarioPath.rfind('/')) + 1:-4]}'
        {self.signalDuration} seconds signals using desired file(s):
        \t{[PurePath(f).name for f in self.desiredSignalFile]}
        and noise signal file(s):
        \t{[PurePath(f).name for f in self.noiseSignalFile]}
        with a base SNR btw. dry signals of {self.baseSNR} dB.
        ------ DANSE settings ------
        Exponential averaging constant: beta = {self.expAvgBeta}.
        """
        return string

    @classmethod
    def load(cls, filename: str):
        return dataclass_methods.load(cls, filename)

    def save(self, filename: str):
        dataclass_methods.save(self, filename)


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
    desiredSigEst: np.ndarray = np.array([])            # Desired signal(s) estimates for each node, in time-domain
    desiredSigEst_STFT: np.ndarray = np.array([])       # Desired signal(s) estimates for each node, in STFT domain
    stftComputed: bool = False                          # Set to true when the STFTs are computed
    fs: int = 16e3                                      # Sampling frequency [samples/s]
    referenceSensor: int = 0                            # Index of the reference sensor at each node

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

    def get_all_stfts(self, fs, L, eL):
        """Derives time-domain signals' STFT representations
        given certain settings.
        Parameters
        ----------
        fs : int
            Sampling frequency [samples/s].
        L : int
            STFT window length [samples].
        eL : int
            STFT _effective_ window length (inc. overlap) [samples].
        """
        # New fields:
            # XXXX_STFT: STFT-domain representation of time-domain signal XXXX
            # freqBins: STFT-domain frequency bins
            # timeFrames: STFT-domain time frames
        self.wetNoise_STFT, self.freqBins, self.timeFrames = get_stft(self.wetNoise, fs, L, eL)
        self.wetSpeech_STFT, _, _ = get_stft(self.wetSpeech, fs, L, eL)
        self.sensorSignals_STFT, _, _ = get_stft(self.sensorSignals, fs, L, eL)

        self.stftComputed = True

        return self

    def plot_signals(self, nodeIdx, sensorIdx, beta=None):
        """Creates a visual representation of the signals at a particular sensor.
        Parameters
        ----------
        nodeNum : int
            Index of the node to inspect.
        sensorNum : int
            Index of the sensor of nodeNum to inspect.
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
        delta = np.amax(self.sensorSignals)
        ax.plot(self.timeVector, self.wetSpeech[:, effectiveSensorIdx], label='Desired')
        ax.plot(self.timeVector, self.VAD * np.amax(self.wetSpeech[:, effectiveSensorIdx]) * 1.1, 'k-', label='VAD')
        ax.plot(self.timeVector, self.wetNoise[:, effectiveSensorIdx] - 2*delta, label='Noise-only')
        ax.plot(self.timeVector, self.sensorSignals[:, effectiveSensorIdx] - 4*delta, label='Noisy')
        if desiredSignalsAvailable:        
            ax.plot(self.timeVector, self.desiredSigEst[:, nodeIdx] - 6*delta, label='Enhanced')
        ax.set_yticklabels([])
        ax.set(xlabel='$t$ [s]')
        ax.grid()
        plt.legend(loc=(0.01, 0.5), fontsize=8)
        ti = f'Node {nodeIdx + 1}, sensor {sensorIdx + 1}'
        if beta is not None:
            ti += f' -- $\\beta = {beta}$'
        plt.title(ti)
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
            # Plot
            ax = fig.add_subplot(nRows,2,2)     # Wet desired signal
            data = 20 * np.log10(np.abs(np.squeeze(self.wetSpeech_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Desired')
            plt.xticks([])
            ax = fig.add_subplot(nRows,2,4)     # Wet noise only
            data = 20 * np.log10(np.abs(np.squeeze(self.wetNoise_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Noise-only')
            plt.xticks([])
            ax = fig.add_subplot(nRows,2,6)     # Sensor signals
            data = 20 * np.log10(np.abs(np.squeeze(self.sensorSignals_STFT[:, :, effectiveSensorIdx])))
            stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Noisy')
            if desiredSignalsAvailable:
                plt.xticks([])
                ax = fig.add_subplot(nRows,2,8)     # Enhanced signals
                data = 20 * np.log10(np.abs(np.squeeze(self.desiredSigEst_STFT[:, :, nodeIdx])))
                stft_subplot(ax, self.timeFrames, self.freqBins, data, [limLow, limHigh], 'Enhanced')
                ax.set(xlabel='$t$ [s]')
            else:
                ax.set(xlabel='$t$ [s]')
        plt.tight_layout()

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
            wavfile.write(fname_noisy[-1], self.fs, data)
            #
            fname_desired.append(f'{folder}/wav/desired_N{idxNode + 1}_Sref{self.referenceSensor + 1}.wav')
            data = normalize_toint16(self.wetSpeech[:, idxSensor])
            wavfile.write(fname_desired[-1], self.fs, data)
            #
            if len(self.desiredSigEst) > 0:  # if enhancement has been performed
                fname_enhanced.append(f'{folder}/wav/enhanced_N{idxNode + 1}.wav')
                data = normalize_toint16(self.desiredSigEst[:, idxNode])
                wavfile.write(fname_enhanced[-1], self.fs, data)
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
class EnhancementMeasures:
    """Class for storing speech enhancement metrics values"""
    snr: dict         # Unweighted SNR
    fwSNRseg: dict    # Frequency-weighted segmental SNR
    sisnr: dict       # Speech-Intelligibility-weighted SNR
    stoi: dict        # Short-Time Objective Intelligibility


@dataclass
class Results:
    """Class for storing simulation results"""
    signals: Signals = field(init=False)                       # all signals involved in run
    enhancementEval: EnhancementMeasures = field(init=False)   # speech enhancement evaluation metrics
    acousticScenario: AcousticScenario = field(init=False)     # acoustic scenario considered

    def save(self, foldername: str):
        """Exports DANSE performance results to directory."""
        if not Path(foldername).is_dir():
            Path(foldername).mkdir(parents=True)
            print(f'Created output directory "{foldername}".')
        pickle.dump(self.signals, gzip.open(f'{foldername}/signals.pkl.gz', 'wb'))
        pickle.dump(self.enhancementEval, gzip.open(f'{foldername}/enhanc_metrics.pkl.gz', 'wb'))
        pickle.dump(self.acousticScenario, gzip.open(f'{foldername}/asc.pkl.gz', 'wb'))
        print(f'DANSE performance results exported to directory\n"{foldername}".')

    @classmethod
    def load(cls, foldername: str):
        """Imports DANSE performance results from directory."""
        if not Path(foldername).is_dir():
            raise ValueError(f'The folder "{foldername}" cannot be found.')
        p = cls()
        p.signals = pickle.load(gzip.open(f'{foldername}/signals.pkl.gz', 'r'))
        p.enhancementEval = pickle.load(gzip.open(f'{foldername}/enhanc_metrics.pkl.gz', 'r'))
        p.acousticScenario = pickle.load(gzip.open(f'{foldername}/asc.pkl.gz', 'r'))
        print(f'DANSE performance results loaded from directory\n"{foldername}".')
        return p

    def plot_enhancement_metrics(self):
        """Creates a visual representation of DANSE performance results."""
        # Useful variables
        _, sensorCounts = np.unique(self.signals.sensorToNodeTags, return_counts=True)
        barWidth = 1 / np.amax(sensorCounts)
        
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(2, 2, 1)   # Unweighted SNR
        numNodes = self.signals.desiredSigEst.shape[1]
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.snr)
        ax.set(ylabel='$\Delta$SNR [dB]')
        ax = fig.add_subplot(2, 2, 2)   # SI-SNR
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.sisnr)
        ax.set(ylabel='$\Delta$SI-SNR [dB]')
        ax = fig.add_subplot(2, 2, 3)   # fwSNRseg
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.fwSNRseg)
        ax.set(ylabel='fwSNRseg')
        ax = fig.add_subplot(2, 2, 4)   # STOI
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.stoi)
        ax.set(ylabel='STOI')
        ax.set_ylim(0,1)
        plt.show()
        return fig

        
def get_stft(mySignal, fs, stftWinLength, stftEffectiveFrameLen):
    """Derives time-domain signals' STFT representation
    given certain settings.
    Parameters
    ----------
    mySignal : [N x C] np.ndarray (float)
        Time-domain signal(s).
    fs : int
        Sampling frequency [samples/s].
    stftWinLength : int
        STFT window length [samples].
    stftEffectiveFrameLen : int
        STFT _effective_ window length (inc. overlap) [samples].

    Returns
    -------
    mySignal_STFT : [Nf x Nt x C] np.ndarray (complex)
        STFT-domain signal(s).
    freqBins : [Nf x 1] np.ndarray (real)
        STFT frequency bins.
    timeFrames : [Nt x 1] np.ndarray (real)
        STFT time frames.
    """
    
    mySignal_STFT, freqBins = calcSTFT(mySignal, Fs=fs, 
                                    win=np.hanning(stftWinLength), 
                                    N_STFT=stftWinLength, 
                                    R_STFT=stftEffectiveFrameLen, 
                                    sides='onesided')

    timeFrames = np.linspace(0, 1, num=mySignal_STFT.shape[1]) * mySignal.shape[0] / fs

    return mySignal_STFT, freqBins, timeFrames


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
        ax.bar(xTicksCurr, toPlot, width=barWidth, color='tab:gray', edgecolor='k')
        xTicks += list(xTicksCurr)
        xTickLabels += [f'N{idxNode + 1}S{ii + 1}' for ii in range(numSensors)]
    plt.xticks(xTicks, xTickLabels, fontsize=8)
    ax.grid()

    

def stft_subplot(ax, t, f, data, vlims, label):
    """Helper function for <Signals.plot_signals()>."""
    # Text boxes properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #
    mappable = ax.pcolormesh(t, f / 1e3, data, vmin=vlims[0], vmax=vlims[1])
    ax.set(ylabel='$f$ [kHz]')
    ax.text(0.025, 0.9, label, fontsize=8, transform=ax.transAxes,
        verticalalignment='top', bbox=props)
    ax.yaxis.label.set_size(8)
    plt.colorbar(mappable)
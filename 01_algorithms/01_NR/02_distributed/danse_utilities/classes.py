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
import metrics.eval_enhancement as evenh
if not any("01_acoustic_scenes" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes')
from utilsASC.classes import AcousticScenario


@dataclass
class SROdata:
    estMethod : str         # SRO estimation method
    compensation : bool     # if True, compensation was applied, else, not
    residuals : np.ndarray  # SRO residuals through time
    estimate : np.ndarray   # SRO estimates through time
    groundTruth : list[float] = field(default_factory=list)  # ground truth SROs per node

    def plotSROdata(self, xaxistype='iterations', fs=16000, Ns=512):
        """Show evolution of SRO estimates / residuals through time.
        
        Parameters
        ----------
        xaxistype : str
            Type of x-axis ticks/label: "iterations" = DANSE iteration indices
            "time" = time instants [s]
        fs : float or int
            Sampling frequency of the reference node [Hz].
        Ns : int
            Number of new samples per DANSE iteration. 

        Returns
        -------
        fig : figure handle
            Figure handle for further processing.
        """
        
        nNodes = len(self.residuals)

        fig = plt.figure(figsize=(6,2))
        ax = fig.add_subplot(111)
        for k in range(nNodes):
            if self.compensation:
                ax.plot(self.residuals[k] * 1e6, f'C{k}-', label=f'$\\Delta\\hat{{\\varepsilon}}(k={k+1},q_{{k,1}})$')
                ax.plot(self.estimate[k] * 1e6, f'C{k}--', label=f'$\\hat{{\\varepsilon}}(k={k+1},q_{{k,1}})$')
            else:
                ax.plot(self.residuals[k] * 1e6, f'C{k}-', label=f'$\\hat{{\\varepsilon}}(k={k+1},q_{{k,1}})$')
            ax.hlines(y=(self.groundTruth[(k+1) % nNodes] - self.groundTruth[k]) * 1e6,
                        xmin=0, xmax=len(self.residuals[0]), colors=f'C{k}', linestyles='dotted', label=f'$\\varepsilon(k={k+1},q_{{k,1}})$')
        ax.grid()
        ax.set_ylabel('[ppm]')
        ax.set_xlabel('DANSE iteration $i$ [-]')
        ax.set_xlim([0, len(self.residuals[k])])
        if xaxistype == 'time':
            xticks = np.linspace(start=0, stop=len(self.residuals[0]), num=9)
            ax.set_xticks(xticks)
            ax.set_xticklabels(np.round(xticks * Ns / fs, 1))
            ax.set_xlabel('Time $t$ [s]')
        plt.legend(bbox_to_anchor=(1.05, 1.05))
        plt.title('SRO estimation through time')
        plt.tight_layout()

        return fig


@dataclass
class PrintoutsParameters:
    events_parser: bool = False             # controls printouts in `events_parser()` function
    danseProgress: bool = True              # controls printouts during DANSE processing (indicating loop process in %)
    externalFilterUpdates: bool = False     # controls printouts at DANSE external filter updates (for broadcasting)


@dataclass
class CohDriftSROEstimationParameters():
    """
    Dataclass containing the required parameters for the
    "Coherence drift" SRO estimation method.

    Attributes
    ----------
    alpha : float
        Exponential averaging constant.
    segLength : int 
        Number of DANSE filter updates per SRO estimation segment
    estEvery : int
        Estimate SRO every `estEvery` signal frames.
    startAfterNupdates : int 
        Minimum number of DANSE filter updates before first SRO estimation
    estimationMethod : str
        SRO estimation methods once frequency-wise estimates are obtained.
        Options: "gs" (golden section search in time domain [1]), 
                "mean" (similar to Online WACD implementation [2]),
                "ls" (least-squares estimate over frequency bins [3])
    alphaEps : float
        Residual SRO incrementation factor:
        $\\hat{\\varepsilon}^i = \\hat{\\varepsilon}^{i-1} + `alphaEps` * \\Delta\\varepsilon^i$

    References
    ----------
    [1] Gburrek, Tobias, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach.
        "On Synchronization of Wireless Acoustic Sensor Networks in the
        Presence of Time-Varying Sampling Rate Offsets and Speaker Changes."
        ICASSP 2022-2022 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2022.
        
    [2] Chinaev, Aleksej, et al. "Online Estimation of Sampling Rate
        Offsets in Wireless Acoustic Sensor Networks with Packet Loss."
        2021 29th European Signal Processing Conference (EUSIPCO). IEEE, 2021.
        
    [3] Bahari, Mohamad Hasan, Alexander Bertrand, and Marc Moonen.
        "Blind sampling rate offset estimation for wireless acoustic sensor
        networks through weighted least-squares coherence drift estimation."
        IEEE/ACM Transactions on Audio, Speech, and Language Processing 25.3
        (2017): 674-686.
    """
    alpha : float = .95                 
    segLength : int = 10                # segment length: use phase angle between values
                                        # spaced by `segLength` signal frames
                                        # to estimate the SRO
    estEvery : int = 1                  # estimate SRO every `estEvery` signal frames
    startAfterNupdates : int = 11       # only start estimating the SRO after `startAfterNupdates`
                                        # signal frames
    estimationMethod : str = 'gs'       # options: "gs" (golden section search in time domain [1]), 
                                        # "mean" (similar to Online WACD implementation [2]),
                                        # "ls" (least-squares estimate over frequency bins [3])
    alphaEps : float = .05              # residual SRO incrementation factor
    loop : str = 'closed'               # SRO estimation + compensation loop type
                                        # "closed": feedback loop, using SRO-compensated signals for estimation
                                        # "open": no feedback, using SRO-uncompensated signals for estimation
    
@dataclass
class DWACDParameters():
    """
    Dataclass containing the required parameters for the
    Dynamic Weighted Average Coherence Drift method for 
    SRO estimation.

    Attributes
    ----------
    seg_len : int
        Length of the segments used for coherence estimation (= Length
        of the segments used for power spectral density (PSD)
        estimation based on a Welch method)
    seg_shift : int
        Shift of the segments used for coherence estimation (The SRO is
        estimated every seg_shift samples)
    temp_dist : int
        Amount of samples between the two consecutive coherence
        functions
    alpha : float
        Smoothing factor used for the autoregressive smoothing for time
        averaging of the complex conjugated coherence product
    src_activity_th : float
        If the amount of time with source activity within one segment
        is smaller than the threshold src_activity_th the segment will
        not be used to update th average coherence product.
    settling_time : int
        Amount of segments after which the SRO is estimated for the
        first time
    frame_shift_welch : int
        Frame shift used for the Welch method utilized for
        PSD estimation
    fft_size : int
        Frame size / FFT size used for the Welch method utilized for
        PSD estimation
    nFiltUpdatePerSeg : int 
        Number of DANSE filter updates per DWACD segment
    """
    seg_len : int = 2**13           # 2**13: default value from https://github.com/fgnt/paderwasn
    seg_shift : int = 2**11         # 2**11: default value from https://github.com/fgnt/paderwasn
    temp_dist : int = 2**13         # 2**13: default value from https://github.com/fgnt/paderwasn
    alpha : float = .95             # .95  : default value from https://github.com/fgnt/paderwasn
    src_activity_th : float = .75   # .75  : default value from https://github.com/fgnt/paderwasn
    settling_time : int = 40        # 40   : default value from https://github.com/fgnt/paderwasn
    frame_shift_welch: int = 2**9  
    fft_size : int = 2**12
    nFiltUpdatePerSeg : int = 1     # automatically adjusted in `__post_init__` of `ProgramSettings` object

@dataclass
class SamplingRateOffsets():
    """Sampling rate/time offsets class, containing all necessary info for
    applying, estimating, and compensation SROs/STOs"""
    SROsppm: list[float] = field(default_factory=list)     # SROs [ppm] to be applied to each node (taking Node#1 (idx = 0) as reference)
    compensateSROs: bool = False            # if True, compensate SROs
    estimateSROs: str = 'Oracle'            # SRO estimation method. If 'Oracle', no estimation: using oracle if `compensateSROs == True`
    STOinducedDelays: list[float] = field(default_factory=list)     # [s] STO-induced time delays between nodes (different starts of recording)
    compensateSTOs: bool = False            # if True, compensate STOs
    estimateSTOs: bool = False              # if True, estimate STOs
    dwacd: DWACDParameters = DWACDParameters()  # parameters for Dynamic Weighted Average Coherence Drift SRO estimation
    cohDriftMethod: CohDriftSROEstimationParameters = CohDriftSROEstimationParameters()     # parameters for "Coherence drift" SRO estimation method
    plotResult: bool = False                # if True, plot results via function `sro_subfcns.SROdata.plotSROdata()`

    def __post_init__(self):
        # Base checks
        if self.estimateSROs not in ['Oracle', 'CohDrift', 'DWACD']:
            raise ValueError(f'The SRO estimation method provided ("{self.estimateSROs}") is invalid. Possible options: "Oracle", "CohDrift", "DWACD".')
        
        elif isinstance(self.SROsppm, float) or isinstance(self.SROsppm, int):
            self.SROsppm = [self.SROsppm]
        if isinstance(self.SROsppm, list):
            if len(self.SROsppm) > 0 and all(v == 0 for v in self.SROsppm) and self.compensateSROs:
                inn = input('No SROs involved -- no need to compensate. Set `compensateSROs` to `False`? [y/n]  ')
                if inn in ['Y', 'y']:
                    print('Setting `compensateSROs` to `False`.')
                    self.compensateSROs = False
                    self.plotResult = False     # <-- no need to plot
                else:
                    print('Keeping `compensateSROs` as `True`.')


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
    stftWin: np.ndarray = np.array([])      # STFT window
    wasnTopology: str = 'fully_connected'   # WASN topology (fully connected or ad hoc)
    # VAD parameters
    VADwinLength: float = 40e-3             # [s] VAD window length
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    # DANSE parameters
    danseUpdating: str = 'sequential'       # node-updating scheme: "sequential" or "simultaneous"
    timeBtwExternalFiltUpdates: float = 0   # [s] minimum time between 2 consecutive external filter update (i.e. filters that are used for broadcasting)
                                            #  ^---> If 0: equivalent to updating coefficients every `chunkSize * (1 - chunkOverlap)` new captured samples 
    chunkSize: int = 1024                   # [samples] processing chunk size
    chunkOverlap: float = 0.5               # amount of overlap between consecutive chunks; in [0,1) -- full overlap [=1] unauthorized.
    danseWindow: np.ndarray = np.hanning(chunkSize)     # DANSE window for FFT/IFFT operations
    initialWeightsAmplitude: float = 1.     # maximum amplitude of initial random filter coefficients
    expAvg50PercentTime: float = 2.         # [s] Time in the past at which the value is weighted by 50% via exponential averaging
                                            # -- Used to compute beta in, e.g.: Ryy[l] = beta * Ryy[l - 1] + (1 - beta) * y[l] * y[l]^H
    performGEVD: bool = True                # if True, perform GEVD in DANSE
    GEVDrank: int = 1                       # GEVD rank approximation (only used is <performGEVD> is True)
    computeLocalEstimate: bool = False      # if True, compute also an estimate of the desired signal using only local sensor observations
    bypassFilterUpdates: bool = False       # if True, only update covariance matrices, do not update filter coefficients (no adaptive filtering)
    broadcastDomain: str = 't'              # inter-node data broadcasting domain: frequency 'wholeChunk' or time 't' [default]
    # Broadcasting parameters
    broadcastLength: int = 8                # [samples] number of (compressed) signal samples to be broadcasted at a time to other nodes
    # Speech enhancement metrics parameters
    gammafwSNRseg: float = 0.2              # gamma exponent for fwSNRseg computation
    frameLenfwSNRseg: float = 0.03          # [s] time window duration for fwSNRseg computation
    dynamicMetricsParams: evenh.DynamicMetricsParameters = evenh.DynamicMetricsParameters()   # dynamic objective speech enhancement metrics computation parameters
    # Asynchronicity (SRO/STO) parameters
    asynchronicity: SamplingRateOffsets = SamplingRateOffsets()   # all-things SROs/STOs
    # Other parameters
    plotAcousticScenario: bool = False      # if true, plot visualization of acoustic scenario. 
    acScenarioPlotExportPath: str = ''      # path to directory where to export the acoustic scenario plot
    randSeed: int = 12345                   # random generator(s) seed
    printouts: PrintoutsParameters = PrintoutsParameters()    # boolean parameters for printouts

    def __post_init__(self) -> None:
        # Base attribute checks
        validTopologies = ['fully_connected', 'adhoc']
        if self.wasnTopology not in validTopologies:
            raise ValueError(f'The WASN topology requested ("{self.wasnTopology}") is invalid (accepted values: {validTopologies}).')
        # Create new attributes
        self.stftEffectiveFrameLen = int(self.stftWinLength * (1 - self.stftFrameOvlp))
        self.expAvgBeta = np.exp(np.log(0.5) / (self.expAvg50PercentTime * self.samplingFrequency / self.stftEffectiveFrameLen))
        # --- SROs ---        
        if self.asynchronicity.estimateSROs == 'DWACD':
            # Add parameters to DWACD method object
            self.asynchronicity.dwacd.frame_shift_welch = self.stftEffectiveFrameLen
            self.asynchronicity.dwacd.fft_size = self.stftWinLength
            nFilterUpdatesBtwConsecutiveDWACDSROupdates = self.asynchronicity.dwacd.seg_shift // self.stftEffectiveFrameLen
            if nFilterUpdatesBtwConsecutiveDWACDSROupdates < 1:
                raise ValueError(f'Too quick DWACD SRO estimation updates (every {self.asynchronicity.dwacd.seg_shift} samples) for the chosen DANSE filter update interval ({self.stftEffectiveFrameLen} samples).')
            else:
                self.asynchronicity.dwacd.nFiltUpdatePerSeg = nFilterUpdatesBtwConsecutiveDWACDSROupdates
        # Check for frequency-domain broadcasting option
        if self.broadcastDomain == 'f':
            if self.broadcastLength != self.stftWinLength / 2:
                val = input(f'Frequency-domain broadcasting only allows L=Ns. Current value of L: {self.broadcastLength}. Change to Ns (error otherwise)? [y]/n  ')
                if val in ['y', 'Y']:
                    self.broadcastLength = self.stftWinLength / 2
                else:
                    raise ValueError(f'When broadcasting in the freq.-domain, L must be equal to Ns.')
        elif self.broadcastDomain != 't':
            raise ValueError(f'The broadcasting domain must be "t" or "f" (current value: "{self.broadcastDomain}").')
        # Adapt formats
        self.asynchronicity.SROsppm = sto_sro_formatting(self.asynchronicity.SROsppm)
        self.asynchronicity.STOinducedDelays = sto_sro_formatting(self.asynchronicity.STOinducedDelays)
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
        self.stftWin = np.hanning(self.stftWinLength)
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
        if self.stftEffectiveFrameLen % self.broadcastLength != 0:
            raise ValueError(f'The broadcast length L should be a divisor of the effective STFT length Ns\n(currently: Ns/L={self.stftEffectiveFrameLen}/{self.broadcastLength}={self.stftEffectiveFrameLen / self.broadcastLength}!=1)') 
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
        if (np.array(self.asynchronicity.SROsppm) != 0).any():
            string += f'\n------ SRO settings ------'
            for idxNode in range(len(self.asynchronicity.SROsppm)):
                string += f'\nSRO Node {idxNode + 1} = {self.asynchronicity.SROsppm[idxNode]} ppm'
                if self.asynchronicity.SROsppm[idxNode] == 0:
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


def sto_sro_formatting(arg):
    """Formatting function. Transforms a list or a float into an array.
    Ensures the presence of at least one `0` inside the array.
    
    Parameters
    ----------
    arg : input list / float / array.

    Returns
    -------
    out : np.ndarray.
        Formatted array.
    """

    if isinstance(arg, int) or isinstance(arg, float):
        if arg == 0:
            out = np.array([arg])     # transform to NumPy array
        else:
            raise ValueError('At least one node should have an SRO of 0 ppm (base sampling frequency).')
    elif 0 not in arg:
        if arg == []:
            out = np.array([0.])     # set to 0 and transform to NumPy array
        else:
            raise ValueError('At least one node should have an SRO of 0 ppm (base sampling frequency).')
    else:
        out = np.array(arg)           # transform to NumPy array

    return out


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
        ax.plot(self.timeStampsSROs[:, nodeIdx], self.wetSpeech[:, effectiveSensorIdx], label='Desired')
        ax.plot(self.timeStampsSROs[:, nodeIdx], self.VAD * np.amax(self.wetSpeech[:, effectiveSensorIdx]) * 1.1, 'k-', label='VAD')
        # ax.plot(self.timeVector, self.wetNoise[:, effectiveSensorIdx] - 2*delta, label='Noise-only')
        ax.plot(self.timeStampsSROs[:, nodeIdx], self.sensorSignals[:, effectiveSensorIdx] - 2*delta, label='Noisy')
        # -------- Desired signal estimate waveform -------- 
        if desiredSignalsAvailable:        
            # -------- Desired signal _local_ estimate waveform -------- 
            if settings.computeLocalEstimate:
                ax.plot(self.timeStampsSROs[:, nodeIdx], self.desiredSigEstLocal[:, nodeIdx] - 4*delta, label='Enhanced (local)')
                deltaNextWaveform = 6*delta
            else:
                deltaNextWaveform = 4*delta
            ax.plot(self.timeStampsSROs[:, nodeIdx], self.desiredSigEst[:, nodeIdx] - deltaNextWaveform, label='Enhanced (global)')
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
                ax.set_title(f'Local vs. global eSTOI improvement: {txt}')
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
        plt.title(f'Best: Node {bestNodeIdx + 1} (eSTOI = {np.round(perf.stoi[f"Node{bestNodeIdx + 1}"].after, 2)})')
        plt.xlabel('$t$ [s]')
        # Worst node
        ax = fig.add_subplot(int(nSubplotsRows * 100 + 22))
        colorb = stft_subplot(ax, self.timeFrames, self.freqBins[:, worstNodeSensorIdx], 20*np.log10(np.abs(dataWorst)), [climLow, climHigh])
        ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
        plt.title(f'Worst: Node {worstNodeIdx + 1} (eSTOI = {np.round(perf.stoi[f"Node{worstNodeIdx + 1}"].after, 2)})')
        plt.xlabel('$t$ [s]')
        colorb.set_label('[dB]')
        if plotLocalEstimate:
            ax = fig.add_subplot(int(nSubplotsRows * 100 + 23))
            stft_subplot(ax, self.timeFrames, self.freqBins[:, bestNodeSensorIdx], 20*np.log10(np.abs(dataBestLocal)), [climLow, climHigh])
            ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
            plt.title(f'Local estimate: Node {bestNodeIdx + 1} (eSTOI = {np.round(perf.stoi[f"Node{bestNodeIdx + 1}"].afterLocal, 2)})')
            plt.xlabel('$t$ [s]')
            # Worst node
            ax = fig.add_subplot(int(nSubplotsRows * 100 + 24))
            colorb = stft_subplot(ax, self.timeFrames, self.freqBins[:, worstNodeSensorIdx], 20*np.log10(np.abs(dataWorstLocal)), [climLow, climHigh])
            ax.set_ylim([ylimLow / 1e3, ylimHigh / 1e3])
            plt.title(f'Local estimate: Node {worstNodeIdx + 1} (eSTOI = {np.round(perf.stoi[f"Node{worstNodeIdx + 1}"].afterLocal, 2)})')
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
class Results(object):
    """Class for storing simulation results"""
    signals : Signals = field(init=False)                       # all signals involved in run
    enhancementEval : EnhancementMeasures = field(init=False)   # speech enhancement evaluation metrics
    acousticScenario : AcousticScenario = field(init=False)     # acoustic scenario considered
    sroData : SROdata = field(init=False)                       # SRO estimation data

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

    def plot_enhancement_metrics(self, plotLocal=False):
        """Creates a visual representation of DANSE performance results.

        Parameters
        ----------
        plotLocal : bool
            If True, also plot local enhancement metrics.
        
        Returns
        -------
        fig1 : matplotlib figure handle
            Figure for batch metrics, computed over entire signals.
        fig2 : matplotlib figure handle
            Figure for online (dynamic) metrics, computed over chunks of signals.
        """

        # Useful variables
        barWidth = 1
        numNodes = self.signals.desiredSigEst.shape[1]
        
        fig1 = plt.figure(figsize=(10,3))
        ax = fig1.add_subplot(1, 4, 1)   # Unweighted SNR
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.snr)
        ax.set(title='SNR', ylabel='[dB]')
        plt.legend()
        #
        ax = fig1.add_subplot(1, 4, 2)   # fwSNRseg
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.fwSNRseg)
        ax.set(title='fwSNRseg', ylabel='[dB]')
        #
        ax = fig1.add_subplot(1, 4, 3)   # STOI
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.stoi)
        ax.set(title='eSTOI')
        ax.set_ylim([0, 1])
        #
        ax = fig1.add_subplot(1, 4, 4)   # PESQ
        metrics_subplot(numNodes, ax, barWidth, self.enhancementEval.pesq)
        ax.set(title='PESQ')

        # Check where dynamic metrics were computed
        flagsDynMetrics = np.zeros(len(fields(self.enhancementEval)), dtype=bool)
        for ii, field in enumerate(fields(self.enhancementEval)):
            if getattr(self.enhancementEval, field.name)['Node1'].dynamicFlag:
                flagsDynMetrics[ii] = True

        nDynMetrics = np.sum(flagsDynMetrics)

        if nDynMetrics > 0:
            # Prepare subplots for dynamic metrics
            if nDynMetrics < 4:
                nRows, nCols = 1, nDynMetrics
            else:
                nRows, nCols = 2, int(np.ceil(nDynMetrics / 2))
            fig2, axes = plt.subplots(nRows, nCols)
            fig2.set_figheight(2.5 * nRows)
            fig2.set_figwidth(5 * nCols)
            axes = axes.flatten()   # flatten axes array for easy indexing
            
            # Select dictionary elements
            dynMetricsNames = [fields(self.enhancementEval)[ii].name\
                                for ii in range(len(fields(self.enhancementEval))) if flagsDynMetrics[ii]]
            dynMetrics = [getattr(self.enhancementEval, n) for n in dynMetricsNames]

            # Plot
            for ii, dynMetric in enumerate(dynMetrics):
                for nodeRef, value in dynMetric.items():        # loop over nodes
                    metric = value.dynamicMetric
                    idxColor = int(nodeRef[-1]) - 1
                    axes[ii].plot(metric.timeStamps, metric.before, color=f'C{idxColor}', linestyle='--', label=f'{nodeRef}: Before')
                    axes[ii].plot(metric.timeStamps, metric.after, color=f'C{idxColor}', linestyle='-',label=f'{nodeRef}: After')
                    if plotLocal:
                        axes[ii].plot(metric.timeStamps, metric.afterLocal, color=f'C{idxColor}', linestyle=':',label=f'{nodeRef}: After (local)')
                axes[ii].grid()
                axes[ii].set_title(dynMetricsNames[ii])
                if ii == 0:
                    axes[ii].legend(loc='lower left', fontsize=8)
                # if ii >= nCols * nRows / 2: # TODO: only set x label to lower row of plots
                axes[ii].set_xlabel('$t$ [s]')  
            plt.tight_layout()
        else:
            fig2 = None

        return fig1, fig2

        
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


def dynamic_metric_subplot(dynObject, ref):

    stop = 1

    return None
    

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
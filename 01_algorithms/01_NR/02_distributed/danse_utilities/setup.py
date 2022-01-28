import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import scipy.signal as sig
from pathlib import Path, PurePath
from scipy.signal._arraytools import zero_ext

from . import classes           # <-- classes for DANSE
from . import danse_scripts     # <-- scripts for DANSE
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import *
import VAD
from metrics import eval_enhancement
sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes')
from utilsASC.classes import AcousticScenario


def run_experiment(settings: classes.ProgramSettings):
    """Wrapper - runs a DANSE experiment given certain settings.
    
    Parameters
    ----------
    settings : ProgramSettings object
        The settings for the current run.

    Returns
    -------
    results : Results object
        The experiment results.
    """

    # Generate base signals (and extract acoustic scenario)
    mySignals, asc = generate_signals(settings)

    # Convert all DANSE input signals to the STFT domain
    mySignals.get_all_stfts(asc.samplingFreq, settings.stftWinLength, settings.stftEffectiveFrameLen)

    # DANSE
    mySignals.desiredSigEst_STFT = danse(mySignals, asc, settings)

    # --------------- Post-process ---------------
    # Back to time-domain
    mySignals.desiredSigEst, mySignals.timeVector = get_istft(mySignals.desiredSigEst_STFT,
                                                            asc.samplingFreq, settings)
    # Compute speech enhancement evaluation metrics
    enhancementEval = evaluate_enhancement_outcome(mySignals, settings)
    # Build output object
    results = classes.Results()
    results.signals = mySignals
    results.enhancementEval = enhancementEval
    results.acousticScenario = asc

    return results


def resample_for_sro(x, baseFs, SROppm):
    """Resamples a vector given an SRO and a base sampling frequency.

    Parameters
    ----------
    x : [N x 1] np.ndarray
        Signal to be resampled.
    baseFs : float or int
        Base sampling frequency [samples/s].
    SROppm : float [TODO: AS OF 21/01/2022, IMPLEMENTED ONLY FOR SROppm >= 0]
        SRO [ppm].

    Returns
    -------
    xResamp : [N x 1] np.ndarray
        Resampled signal
    t : [N x 1] np.ndarray
        Corresponding resampled time stamp vector.
    """
    tOriginal = np.arange(len(x)) / baseFs
    fsSRO = baseFs * (1 + SROppm / 1e6)
    numSamplesPostResamp = int(fsSRO / baseFs * len(x))
    xResamp, t = sig.resample(x, num=numSamplesPostResamp, t=tOriginal)
    if len(xResamp) >= len(x):
        xResamp = xResamp[:len(x)]
        t = t[:len(x)]
    else:
        raise ValueError('SRO < 0 NOT YET IMPLEMENTED.')
    return xResamp, t


def apply_sro(sigs, baseFs, sensorToNodeTags, SROsppm, showSRO=False):
    """Applies sampling rate offsets (SROs) to signals.

    Parameters
    ----------
    sigs : [N x Ns] np.ndarray
        Signals onto which to apply SROs (<Ns> sensors with <N> samples each).
    baseFs : float
        Base sampling frequency [samples/s]
    sensorToNodeTags : [Ns x 1] np.ndarray
        Tags linking each sensor (channel) to a node (i.e. an SRO).
    SROsppm : [Nn x 1] np.ndarray or list
        SROs per node [ppm].
    showSRO : bool
        If True, plots a visualization of the applied SROs.

    Returns
    -------
    sigsOut : [N x Ns] np.ndarray
        Signals after SROs application.
    timeVectorOut : [N x Nn] np.ndarray
        Corresponding sensor-specific time stamps vectors.
    """
    # Extract useful variables
    numSamples = sigs.shape[0]
    numSensors = sigs.shape[-1]
    numNodes = len(np.unique(sensorToNodeTags))
    if numNodes != len(SROsppm):
        if (np.array(SROsppm) == 0).all():
            SROsppm = [0 for _ in range(numNodes)]
        else:
            raise ValueError('Number of sensors does not match number of given SRO values, and some of the latter are non-zero.')

    # Base time stamps
    timeVector = np.arange(sigs.shape[0]) / baseFs

    sigsOut       = np.zeros((numSamples, numSensors))
    timeVectorOut = np.zeros((numSamples, numNodes))
    for idxSensor in range(numSensors):
        idxNode = sensorToNodeTags[idxSensor] - 1
        if SROsppm[idxNode] != 0:
            sigsOut[:, idxSensor], timeVectorOut[:, idxNode] = resample_for_sro(sigs[:, idxSensor], baseFs, SROsppm[idxNode])
        else:
            sigsOut[:, idxSensor] = sigs[:, idxSensor]
            timeVectorOut[:, idxNode] = timeVector

    # Plot
    if showSRO:
        minimumObservableDrift = 1   # plot enough samples to observe drifts of at least that many samples on all signals
        smallestDriftFrequency = np.amin(SROsppm[SROsppm != 0]) / 1e6 * baseFs  # [samples/s]
        samplesToPlot = int(minimumObservableDrift / smallestDriftFrequency * baseFs)
        markerFormats = ['o','v','^','<','>','s','*','D']
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        for idxNode in range(numNodes):
            allSensors = np.arange(numSensors)
            idxSensor = allSensors[sensorToNodeTags == (idxNode + 1)]
            if isinstance(idxSensor, np.ndarray):
                idxSensor = idxSensor[0]
            markerline, _, _ = ax.stem(timeVectorOut[:samplesToPlot, idxNode], sigsOut[:samplesToPlot, idxSensor],
                    linefmt=f'C{idxNode}', markerfmt=f'C{idxNode}{markerFormats[idxNode % len(markerFormats)]}',
                    label=f'Node {idxNode + 1} - $\\varepsilon={SROsppm[idxNode]}$ ppm')
            markerline.set_markerfacecolor('none')
        ax.set(xlabel='$t$ [s]', title='SROs visualization')
        ax.grid()
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


    return sigsOut, timeVectorOut


def evaluate_enhancement_outcome(sigs: classes.Signals, settings: classes.ProgramSettings, minNumDANSEupdates=10):
    """Wrapper for computing and storing evaluation metrics after signal enhancement.

    Parameters
    ----------
    sigs : Signals object
        The signals before and after enhancement.
    settings : ProgramSettings object
        The settings for the current run.
    minNumDANSEupdates : int
        Number of DANSE updates before which to discard the signal before
        computing any metric, to avoid bias due to non-stationary noise power
        in the enhanced signals.

    Returns
    -------
    measures : EnhancementMeasures object
        The enhancement metrics.
    """
    # Check whether the signals have been enhanced and stored
    if sigs.desiredSigEst.ndim == 0:
        raise ValueError('The desired signals have not been included in the Signals object\nCannot compute speech enhancement metrics')

    # Derive number of sensors per node
    _, numSensorsPerNode = np.unique(sigs.sensorToNodeTags, return_counts=True)
    # Derive total number of nodes
    numNodes = np.amax(sigs.sensorToNodeTags)
    # Derive starting sample for metrics computations
    startIdx = int(np.ceil(minNumDANSEupdates * settings.timeBtwConsecUpdates * sigs.fs))
    print(f"""
    Computing speech enhancement metrics from the {startIdx + 1}'th sample on
    (avoid SNR bias due to highly non-stationary noise power in first DANSE iterations)...
    """)

    snr      = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Unweighted SNR
    fwSNRseg = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Frequency-weighted segmental SNR
    stoi     = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Short-Time Objective Intelligibility
    tStart = time.perf_counter()    # time computation
    for idxNode in range(numNodes):
        for idxSensor in range(numSensorsPerNode[idxNode]):
            trueIdxSensor = sum(numSensorsPerNode[:idxNode]) + idxSensor
            print(f'Computing signal enhancement evaluation metrics for node {idxNode + 1}/{numNodes} (sensor {idxSensor + 1}/{numSensorsPerNode[idxNode]})...')
            out0, out1, out2 = eval_enhancement.get_metrics(
                                    sigs.wetSpeech[startIdx:, trueIdxSensor],
                                    sigs.sensorSignals[startIdx:, trueIdxSensor],
                                    sigs.desiredSigEst[startIdx:, idxNode], 
                                    sigs.fs,
                                    sigs.VAD[startIdx:],
                                    settings.gammafwSNRseg,
                                    settings.frameLenfwSNRseg
                                    )
            snr[f'Node{idxNode + 1}'].append(out0)
            fwSNRseg[f'Node{idxNode + 1}'].append(out1)
            stoi[f'Node{idxNode + 1}'].append(out2)
    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    measures = classes.EnhancementMeasures(fwSNRseg=fwSNRseg,
                                            stoi=stoi,
                                            snr=snr)

    return measures


def get_istft(X, fs, settings: classes.ProgramSettings):
    """Derives STFT-domain signals' time-domain representation
    given certain settings.

    Parameters
    ----------
    X : [Nf x Nt x C] np.ndarray (complex)
        STFT-domain signal(s).
    fs : int
        Sampling frequency [samples/s].
    settings : ProgramSettings

    Returns
    -------
    x : [N x C] np.ndarray (real)
        Time-domain signal(s).
    t : [N x 1] np.ndarray (real)
        Time vector.
    """
    
    for channel in range(X.shape[-1]):
        _, tmp = sig.istft(X[:, :, channel], 
                                    fs=fs,
                                    window=np.hanning(settings.stftWinLength), 
                                    nperseg=settings.stftWinLength, 
                                    noverlap=settings.stftEffectiveFrameLen,
                                    input_onesided=True)
        if channel == 0:
            x = np.zeros((len(tmp), X.shape[-1]))
        x[:, channel] = tmp

    targetLength = int(np.round(settings.signalDuration * fs))
    if x.shape[0] < targetLength:
        x = np.concatenate((x, np.full((targetLength - x.shape[0], x.shape[1]), np.finfo(float).eps)))
    elif x.shape[0] > targetLength:
        x = x[:targetLength, :]

    t = np.arange(x.shape[0]) / fs

    return x, t


def danse(signals: classes.Signals, asc: classes.AcousticScenario, settings: classes.ProgramSettings):
    """Main wrapper for DANSE computations.

    Parameters
    ----------
    signals : Signals object
        The microphone signals and their relevant attributes.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.

    Returns
    -------
    desiredSigEst_STFT : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes. 
    """
    # Prepare signals for Fourier transforms
    frameSize = settings.stftWinLength
    nNewSamplesPerFrame = settings.stftEffectiveFrameLen
    y = signals.sensorSignals
    # 1) Extend signal on both ends to ensure that the first frame is centred on t = 0 (see <scipy.signal.stft>'s `boundary` argument (default: `zeros`))
    y = zero_ext(y, frameSize // 2, axis=0)
    # 2) Zero-pad signal if necessary and derive number of time frames
    if not (y.shape[0] - frameSize) % nNewSamplesPerFrame == 0:
        nadd = (-(y.shape[0] - frameSize) % nNewSamplesPerFrame) % frameSize  # see <scipy.signal.stft>'s `padded` argument (default: `True`)
        print(f'Padding {nadd} zeros to the signals in order to fit FFT size')
        y = np.concatenate((y, np.zeros([nadd, y.shape[-1]])), axis=0)
        if not (y.shape[0] - frameSize) % nNewSamplesPerFrame == 0:   # double-check
            raise ValueError('There is a problem with the zero-padding...')

    # DANSE it up
    desiredSigEst_STFT = danse_scripts.danse_sequential(y, asc, settings, signals.VAD, signals.timeStampsSROs)
    
    return desiredSigEst_STFT


def whiten(sig):
    """Renders a signal zero-mean and unit-variance."""
    return (sig - np.mean(sig))/np.std(sig)


def pre_process_signal(rawSignal, desiredDuration, originalFs, targetFs):
    """Truncates/extends, resamples, centers, and scales a signal to match a target.

    Parameters
    ----------
    rawSignal : [N_in x 1] np.ndarray
        Raw signal to be processed.
    desiredDuration : float
        Desired signal duration [s].
    originalFs : int
        Original raw signal sampling frequency [samples/s].
    targetFs : int
        Target sampling frequency [samples/s].

    Returns
    -------
    sig_out : [N_out x 1] np.ndarray
        Processed signal.
    """

    signalLength = int(desiredDuration * targetFs)   # desired signal length [samples]
    while len(rawSignal) < signalLength:
        rawSignal = np.concatenate([rawSignal, rawSignal])             # extend too short signals
    if len(rawSignal) > signalLength:
        rawSignal = rawSignal[:desiredDuration * originalFs]  # truncate too long signals
    # Resample signal so that its sampling frequency matches the target
    rawSignal = sig.resample(rawSignal, signalLength) 
    # Whiten signal 
    sig_out = whiten(rawSignal) 
    return sig_out


def generate_signals(settings: classes.ProgramSettings):
    """Generates signals based on acoustic scenario and raw files.
    Parameters
    ----------
    settings : ProgramSettings object
        The settings for the current run.

    Returns
    -------
    micSignals : [N x Nsensors] np.ndarray
        Sensor signals in time domain.
    SNRs : [Nsensors x 1] np.ndarray
        Pre-enhancement, raw sensor signals SNRs.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    """

    # Load acoustic scenario
    asc = AcousticScenario().load(settings.acousticScenarioPath)

    # Detect conflicts
    if asc.numDesiredSources > len(settings.desiredSignalFile):
        raise ValueError(f'{settings.desiredSignalFile} "desired" signal files provided while {asc.numDesiredSources} are needed.')
    if asc.numNoiseSources > len(settings.noiseSignalFile):
        raise ValueError(f'{settings.noiseSignalFile} "noise" signal files provided while {asc.numNoiseSources} are needed.')

    # Desired signal length [samples]
    signalLength = int(settings.signalDuration * asc.samplingFreq) 

    # Load + pre-process dry desired signals and build wet desired signals
    dryDesiredSignals = np.zeros((signalLength, asc.numDesiredSources))
    wetDesiredSignals = np.zeros((signalLength, asc.numDesiredSources, asc.numSensors))
    oVADsourceSpecific = np.zeros((signalLength, asc.numDesiredSources))
    # Compute oracle VAD
    for ii in range(asc.numDesiredSources):
        
        rawSignal, fsRawSignal = sf.read(settings.desiredSignalFile[ii])
        dryDesiredSignals[:, ii] = pre_process_signal(rawSignal, settings.signalDuration, fsRawSignal, asc.samplingFreq)

        # Voice Activity Detection
        thrsVAD = np.amax(dryDesiredSignals[:, ii] ** 2)/settings.VADenergyFactor
        oVADsourceSpecific[:, ii], _ = VAD.oracleVAD(dryDesiredSignals[:, ii], settings.VADwinLength, thrsVAD, asc.samplingFreq)

        # Convolve with RIRs to create wet signals
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]

    # Get VAD consensus
    oVADsourceSpecific = np.sum(oVADsourceSpecific, axis=1)
    oVAD = np.zeros_like(oVADsourceSpecific)
    oVAD[oVADsourceSpecific == asc.numDesiredSources] = 1

    # Load + pre-process dry noise signals and build wet noise signals
    dryNoiseSignals = np.zeros((signalLength, asc.numNoiseSources))
    wetNoiseSignals = np.zeros((signalLength, asc.numNoiseSources, asc.numSensors))
    for ii in range(asc.numNoiseSources):

        rawSignal, fsRawSignal = sf.read(settings.noiseSignalFile[ii])
        tmp = pre_process_signal(rawSignal, settings.signalDuration, fsRawSignal, asc.samplingFreq)
        # Set SNR
        dryNoiseSignals[:, ii] = 10**(-settings.baseSNR / 20) * tmp

        # Convolve with RIRs to create wet signals
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryNoiseSignals[:, ii], asc.rirNoiseToSensors[:, jj, ii])
            wetNoiseSignals[:, ii, jj] = tmp[:signalLength]

    # Build speech-only and noise-only signals
    wetNoise = np.sum(wetNoiseSignals, axis=1)
    wetSpeech = np.sum(wetDesiredSignals, axis=1)
    wetNoise_norm = wetNoise / np.amax(wetNoise + wetSpeech)    # Normalize
    wetSpeech_norm = wetSpeech / np.amax(wetNoise + wetSpeech)  # Normalize

    # --- Apply SROs ---
    wetNoise_norm, _ = apply_sro(wetNoise_norm, asc.samplingFreq, asc.sensorToNodeTags, settings.SROsppm)
    wetSpeech_norm, timeStampsSROs = apply_sro(wetSpeech_norm, asc.samplingFreq, asc.sensorToNodeTags, settings.SROsppm)

    # Build sensor signals
    sensorSignals = wetSpeech_norm + wetNoise_norm

    # Time vector
    timeVector = np.arange(signalLength) / asc.samplingFreq

    # Build output class object
    signals = classes.Signals(dryNoiseSources=dryNoiseSignals,
                                drySpeechSources=dryDesiredSignals,
                                wetIndivNoiseSources=wetNoiseSignals,
                                wetIndivSpeechSources=wetDesiredSignals,
                                wetNoise=wetNoise_norm,
                                wetSpeech=wetSpeech_norm,
                                sensorSignals=sensorSignals,
                                VAD=oVAD,
                                timeVector=timeVector,
                                sensorToNodeTags=asc.sensorToNodeTags,
                                fs=asc.samplingFreq,
                                referenceSensor=settings.referenceSensor,
                                timeStampsSROs=timeStampsSROs,
                                )
                                
    # Check validity of chosen reference sensor
    if (signals.nSensorPerNode < settings.referenceSensor + 1).any():
        conflictIdx = signals.nSensorPerNode[signals.nSensorPerNode < settings.referenceSensor + 1]
        raise ValueError(f'The reference sensor index chosen ({settings.referenceSensor}) conflicts with the number of sensors in node(s) {conflictIdx}.')

    return signals, asc
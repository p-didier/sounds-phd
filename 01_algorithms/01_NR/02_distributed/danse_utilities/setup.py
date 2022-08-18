from copy import copy
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
from pathlib import Path, PurePath
from scipy.signal._arraytools import zero_ext
import resampy

from . import classes           # <-- classes for DANSE
from . import danse_scripts     # <-- scripts for DANSE
if not any("_general_fcts" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import *
import VAD
from metrics import eval_enhancement
if not any("01_acoustic_scenes" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
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

    print('\nGenerating simulation signals...')
    # Generate base signals (and extract acoustic scenario)
    mySignals, asc = generate_signals(settings)

    # DANSE
    print('Launching danse()...')
    mySignals.desiredSigEst, mySignals.desiredSigEstLocal, sroData, tStartForMetrics = launch_danse(mySignals, asc, settings)

    print('Computing STFTs...')
    # Convert all DANSE input signals to the STFT domain
    mySignals.get_all_stfts(mySignals.fs, settings)

    # --------------- Post-process ---------------
    # Compute speech enhancement evaluation metrics
    enhancementEval, startIdx = evaluate_enhancement_outcome(mySignals, settings, tStartForMetrics)
    # Build output object
    results = classes.Results()
    results.signals = mySignals
    results.enhancementEval = enhancementEval
    results.acousticScenario = asc
    results.sroData = sroData
    results.other = classes.MiscellaneousData()
    results.other.metricsStartIdx = startIdx

    return results


def apply_sto(x, stoDelay, fs):
    """Applies a given sampling time offset (STO) to a signal.
    ---
    As of 11/04/2022: only applying full-sample STOs. 
    TODO: include possibility of applying sub-sample STOs.
    ---
    NOTE: STOs are always applied backwards: data is discarded
    from the beginning of the vector. 

    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Input signal (single-channel).
    stoDelay : float
        [s] STO-induced delay.
    fs : int or float
        [samples/s] Sampling frequency.
    
    Returns
    -------
    xsto : [N x 1] np.ndarray (float)
        Output signal (STO-disturbed).
    """

    sto = int(np.floor(stoDelay * fs))

    if sto > 0:
        xsto = np.concatenate((x[sto:], np.zeros(sto)))
    else:
        xsto = x

    return xsto


def resample_for_sro(x, baseFs, SROppm):
    """Resamples a vector given an SRO and a base sampling frequency.

    Parameters
    ----------
    x : [N x 1] np.ndarray
        Signal to be resampled.
    baseFs : float or int
        Base sampling frequency [samples/s].
    SROppm : float
        SRO [ppm].

    Returns
    -------
    xResamp : [N x 1] np.ndarray
        Resampled signal
    t : [N x 1] np.ndarray
        Corresponding resampled time stamp vector.
    fsSRO : float
        Re-sampled signal's sampling frequency [Hz].
    """
    fsSRO = baseFs * (1 + SROppm / 1e6)
    if baseFs != fsSRO:
        xResamp = resampy.core.resample(x, baseFs, fsSRO)
    else:
        xResamp = copy(x)

    # tOriginal = np.arange(len(x)) / baseFs
    # numSamplesPostResamp = int(np.floor(fsSRO / baseFs * len(x)))
    # xResamp, t = sig.resample(x, num=numSamplesPostResamp, t=tOriginal)
    # t = np.arange(0, numSamplesPostResamp) * (tOriginal[1] - tOriginal[0]) * x.shape[0] / float(numSamplesPostResamp) + tOriginal[0]    # based on line 3116 in `scipy.signal.resample`

    t = np.arange(len(xResamp)) / fsSRO

    if len(xResamp) >= len(x):
        xResamp = xResamp[:len(x)]
        t = t[:len(x)]
    else:
        # Append zeros
        xResamp = np.concatenate((xResamp, np.zeros(len(x) - len(xResamp))))
        # Extend time stamps vector
        dt = t[1] - t[0]
        tadd = np.linspace(t[-1]+dt, t[-1]+dt*(len(x) - len(xResamp)), len(x) - len(xResamp))
        t = np.concatenate((t, tadd))

    return xResamp, t, fsSRO


def apply_sro_sto(sigs, baseFs, sensorToNodeTags, SROsppm, STOinducedDelays, plotit=False):
    """Applies sampling rate offsets (SROs) and sampling time offsets (STOs) to signals.

    Parameters
    ----------
    sigs : [N x Ns] np.ndarray
        Signals onto which to apply SROs (<Ns> sensors with <N> samples each).
    baseFs : float
        Base sampling frequency [samples/s]
    sensorToNodeTags : [Ns x 1] np.ndarray
        Tags linking each sensor (channel) to a node (i.e. an SRO).
    SROsppm : [Nn x 1] np.ndarray or list (floats)
        SROs per node [ppm].
    STOinducedDelays : [Nn x 1] np.ndarray or list (floats)
        STO-induced delay per node [s].
    plotit : bool
        If True, plots a visualization of the applied signal changes.

    Returns
    -------
    sigsOut : [N x Ns] np.ndarray of floats
        Signals after SROs application.
    timeVectorOut : [N x Nn] np.ndarray of floats
        Corresponding sensor-specific time stamps vectors.
    fs : [Ns x 1] np.ndarray of floats
        Sensor-specific sampling frequency, after SRO application. 
    """
    # Extract useful variables
    numSamples = sigs.shape[0]
    numSensors = sigs.shape[-1]
    numNodes = len(np.unique(sensorToNodeTags))

    # Check / adapt SROs object size
    if len(SROsppm) != numNodes:
        if len(SROsppm) == 1:
            print(f'Applying the same SRO ({SROsppm} ppm) to all {numNodes} nodes.')
            SROsppm = np.repeat(SROsppm[0], numNodes)
        else:
            raise ValueError(f'An incorrect number of SRO values was provided ({len(SROsppm)} for {numNodes} nodes).')

    # Apply STOs / SROs
    sigsOut       = np.zeros((numSamples, numSensors))
    timeVectorOut = np.zeros((numSamples, numNodes))
    fs = np.zeros(numSensors)
    for idxSensor in range(numSensors):
        k = sensorToNodeTags[idxSensor] - 1     # corresponding node index
        # Apply SROs
        sigsOut[:, idxSensor], timeVectorOut[:, k], fs[idxSensor] = resample_for_sro(sigs[:, idxSensor], baseFs, SROsppm[k])
        # Apply STOs
        sigsOut[:, idxSensor] = apply_sto(sigsOut[:, idxSensor], STOinducedDelays[k], fs[idxSensor])

    # Plot
    if plotit:
        minimumObservableDrift = 1   # plot enough samples to observe drifts of at least that many samples on all signals
        smallestDriftFrequency = np.amin(SROsppm[SROsppm != 0]) / 1e6 * baseFs  # [samples/s]
        samplesToPlot = int(minimumObservableDrift / smallestDriftFrequency * baseFs)
        markerFormats = ['o','v','^','<','>','s','*','D']
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        for k in range(numNodes):
            allSensors = np.arange(numSensors)
            idxSensor = allSensors[sensorToNodeTags == (k + 1)]
            if isinstance(idxSensor, np.ndarray):
                idxSensor = idxSensor[0]
            markerline, _, _ = ax.stem(timeVectorOut[:samplesToPlot, k], sigsOut[:samplesToPlot, idxSensor],
                    linefmt=f'C{k}', markerfmt=f'C{k}{markerFormats[k % len(markerFormats)]}',
                    label=f'Node {k + 1} - $\\varepsilon={SROsppm[k]}$ ppm')
            markerline.set_markerfacecolor('none')
        ax.set(xlabel='$t$ [s]', title='SROs / STOs visualization')
        ax.grid()
        plt.tight_layout()
        plt.show()

    return sigsOut, timeVectorOut, fs


def evaluate_enhancement_outcome(sigs: classes.Signals, settings: classes.ProgramSettings, tStartForMetrics=1):
    """Wrapper for computing and storing evaluation metrics after signal enhancement.

    Parameters
    ----------
    sigs : Signals object
        The signals before and after enhancement.
    settings : ProgramSettings object
        The settings for the current run.
    tStartForMetrics : [Nn x 1] np.ndarray (float)
        Start instants (per node) for the computation of speech enhancement metrics.
        --> Avoiding metric bias due to first DANSE iterations where the filters have not converged yet.
    
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

    startIdx = np.zeros(numNodes, dtype=int)
    snr      = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Unweighted SNR
    fwSNRseg = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Frequency-weighted segmental SNR
    stoi     = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Short-Time Objective Intelligibility
    pesq     = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Perceptual Evaluation of Speech Quality
    startIdx     = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Perceptual Evaluation of Speech Quality
    tStart = time.perf_counter()    # time computation
    for idxNode in range(numNodes):
        trueIdxSensor = settings.referenceSensor + sum(numSensorsPerNode[:idxNode])

        # Derive starting sample for metrics computations
        startIdx[idxNode] = int(tStartForMetrics[idxNode] * sigs.fs[trueIdxSensor])
        print(f"""
        Node {idxNode+1}: computing speech enhancement metrics from the {startIdx[idxNode] + 1}'th sample on
        (t_start = {tStartForMetrics[idxNode]} s; avoid bias due to not-yet-converged filters in first DANSE iterations)...
        """)

        # Adapt local signal estimate input value to `get_metrics()` function
        if settings.computeLocalEstimate:
            localSig = sigs.desiredSigEstLocal[startIdx[idxNode]:, idxNode]
        else:
            localSig = []
        # Adapt centralized signal estimate input value to `get_metrics()` function
        if settings.computeCentralizedEstimate:
            centralizedSig = sigs.desiredSigEstCentralized[startIdx:, idxNode]
        else:
            centralizedSig = []
        
        print(f'Computing signal enhancement evaluation metrics for node {idxNode + 1}/{numNodes} (sensor {settings.referenceSensor + 1}/{numSensorsPerNode[idxNode]})...')
        out0, out1, out2, out3 = eval_enhancement.get_metrics(
                                    sigs.wetSpeech[startIdx[idxNode]:, trueIdxSensor],
                                    sigs.sensorSignals[startIdx[idxNode]:, trueIdxSensor],
                                    sigs.desiredSigEst[startIdx[idxNode]:, idxNode], 
                                    sigs.fs[trueIdxSensor],
                                    sigs.VAD[startIdx[idxNode]:],
                                    settings.dynamicMetricsParams,  # dynamic metrics computation parameters
                                    settings.gammafwSNRseg,
                                    settings.frameLenfwSNRseg,
                                    localSig,
                                    centralizedSig
                                    )

        snr[f'Node{idxNode + 1}'] = out0
        fwSNRseg[f'Node{idxNode + 1}'] = out1
        stoi[f'Node{idxNode + 1}'] = out2
        pesq[f'Node{idxNode + 1}'] = out3
    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    measures = classes.EnhancementMeasures(fwSNRseg=fwSNRseg,
                                            stoi=stoi,
                                            pesq=pesq,
                                            snr=snr)

    return measures, startIdx


def get_istft(X, fs, settings: classes.ProgramSettings):
    """Derives STFT-domain signals' time-domain representation
    given certain settings.

    Parameters
    ----------
    X : [Nf x Nt x C] np.ndarray (complex)
        STFT-domain signal(s).
    fs : [C x 1] np.ndarray of floats
        Sampling frequencies [samples/s].
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
                                    fs=fs[channel],
                                    window=settings.stftWin, 
                                    nperseg=settings.stftWinLength, 
                                    noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
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


def prep_for_ffts(signals: classes.Signals, asc: classes.AcousticScenario, settings: classes.ProgramSettings):
    """Zero-padding and signals length adaptation to ensure correct FFT/IFFT operation.
    Based on FFT implementation by `scipy.signal` module.

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
    y : np.ndarray of floats
        Prepped signals.
    t : np.ndarray of floats
        Corresponding time stamps.
    nadd : int
        Number of zeros added at the of signal after frame-extension (step 2) below).
    """

    frameSize = settings.stftWinLength
    Ns = settings.stftEffectiveFrameLen
    y = signals.sensorSignals

    # 1) Extend signal on both ends to ensure that the first frame is centred on t = 0 -- see <scipy.signal.stft>'s `boundary` argument (default: `zeros`)
    y = zero_ext(y, frameSize // 2, axis=0)
    # --- Also adapt timeInstants vector
    t = signals.timeStampsSROs
    dt = np.diff(t, axis=0)[0, :]   # delta t between each time instant for each node   # TODO what if clock jitter?
    tpre = np.zeros((frameSize // 2, asc.numNodes))
    tpost = np.zeros((frameSize // 2, asc.numNodes))
    for k in range(asc.numNodes):
        tpre[:, k] = np.linspace(start= - dt[k] * (frameSize // 2), stop=-dt[k], num=frameSize // 2)
        tpost[:, k] = np.linspace(start= t[-1, k] + dt[k], stop=t[-1, k] + dt[k] * (frameSize // 2), num=frameSize // 2)
    t = np.concatenate((tpre, t, tpost), axis=0)

    # 2) Zero-pad signal if necessary to include an integer number of frames in the signal
    nadd = 0
    if not (y.shape[0] - frameSize) % Ns == 0:
        nadd = (-(y.shape[0] - frameSize) % Ns) % frameSize  # see <scipy.signal.stft>'s `padded` argument (default: `True`)
        print(f'Padding {nadd} zeros to the signals in order to fit FFT size')
        y = np.concatenate((y, np.zeros([nadd, y.shape[-1]])), axis=0)
        # Adapt time vector too
        tzp = np.zeros((nadd, asc.numNodes))
        for k in range(asc.numNodes):
            tzp[:, k] = np.linspace(start=t[-1, k] + dt[k], stop=t[-1, k] + dt[k] * nadd, num=nadd)     # TODO what if clock jitter?
        t = np.concatenate((t, tzp), axis=0)
        if not (y.shape[0] - frameSize) % Ns == 0:   # double-check
            raise ValueError('There is a problem with the zero-padding...')

    return y, t, nadd


def launch_danse(signals: classes.Signals, asc: classes.AcousticScenario, settings: classes.ProgramSettings):
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
    desiredSigEst : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using full-observations vectors (also data coming from neighbors).
    desiredSigEstLocal : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using only local observations (not data coming from neighbors).
        -Note: if `settings.computeLocalEstimate == False`, then `desiredSigEstLocal` is output as an all-zeros array.
    sroData : danse_subfcns.SROdata object
        Data on SRO estimation / compensation (see danse_subfcns.sro_subfcns module for details).
    tStartForMetrics : [Nn x 1] np.ndarray (float)
        Start instants (per node) for the computation of speech enhancement metrics.
        --> Avoiding metric bias due to first DANSE iterations where the filters have not converged yet.
    """
    # Prepare signals for Fourier transforms
    # -- Zero-padding and signals length adaptation to ensure correct FFT/IFFT operation.
    # -- Based on FFT implementation by `scipy.signal` module.
    y, t, nadd = prep_for_ffts(signals, asc, settings)

    # DANSE it up
    if settings.danseUpdating == 'sequential':
        raise ValueError('NOT YET IMPLEMENTED: conversion to time domain before output in sequential DANSE (see how it is done in `danse_simultaneous()`)')
        desiredSigEst_STFT = danse_scripts.danse_sequential(y, asc, settings, signals.VAD)
    elif settings.danseUpdating == 'simultaneous':
        desiredSigEst, desiredSigEstLocal, sroData, tStartForMetrics = danse_scripts.danse_simultaneous(
            y, asc, settings, signals.VAD, t, signals.masterClockNodeIdx)
    else:
        raise ValueError(f'`danseUpdating` setting unknown value: "{settings.danseUpdating}". Accepted values: {{"sequential", "simultaneous"}}.')

    # Discard pre-DANSE added samples (for Fourier transform processing, see first section of this function)
    desiredSigEst = desiredSigEst[settings.stftWinLength // 2:-(settings.stftWinLength // 2 + nadd)]
    desiredSigEstLocal = desiredSigEstLocal[settings.stftWinLength // 2:-(settings.stftWinLength // 2 + nadd)]
    # desiredSigEstCentralized = desiredSigEstCentralized[settings.stftWinLength // 2:-(settings.stftWinLength // 2 + nadd)]
    
    return desiredSigEst, desiredSigEstLocal, sroData, tStartForMetrics


def whiten(sig, vad=[]):
    """
    Renders a sequence zero-mean and unit-variance.
    
    Parameters
    ----------
    sig : [N x 1] np.ndarray (real floats)
        Non-white input sequence.
    vad : [N x 1] np.ndarray (binary)
        Corresponding oracle Voice Activity Detector.

    Returns
    -------
    sig_out : [N x 1] np.ndarray
        Whitened input.
    """
    
    if vad == []:
        sig_out = (sig - np.mean(sig)) / np.std(sig)
    else:
        sig_out = (sig - np.mean(sig)) / np.std(sig[vad == 1])

    return sig_out


def pre_process_signal(rawSignal, desiredDuration, originalFs, targetFs):
    """Truncates/extends, resamples, centers, and scales a signal to match a target.
    Computes VAD estimate before whitening. 

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
    VADenergyFactor : float or int
        VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor).
    VADwinLength : float
        VAD window duration [s].
    vadGiven : [N x 1] np.ndarray (binary float)
        Pre-computed VAD. If not `[]`, `VADenergyFactor` and `VADwinLength` arguments are ignored.

    Returns
    -------
    sig_out : [N_out x 1] np.ndarray
        Processed signal.
    """

    signalLength = int(desiredDuration * targetFs)   # desired signal length [samples]
    if originalFs != targetFs:
        # Resample signal so that its sampling frequency matches the target
        rawSignal = resampy.resample(rawSignal, originalFs, targetFs)
        # rawSignal = sig.resample(rawSignal, signalLength) 

    while len(rawSignal) < signalLength:
        sig_out = np.concatenate([rawSignal, rawSignal])             # extend too short signals
    if len(rawSignal) > signalLength:
        sig_out = rawSignal[:int(desiredDuration * targetFs)]  # truncate too long signals

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
    if not hasattr(asc, 'nodeLinks'):
        asc.__post_init__()

    # Detect conflicts
    if asc.numDesiredSources > len(settings.desiredSignalFile):
        raise ValueError(f'{settings.desiredSignalFile} "desired" signal files provided while {asc.numDesiredSources} are needed.')
    if asc.numNoiseSources > len(settings.noiseSignalFile):
        raise ValueError(f'{settings.noiseSignalFile} "noise" signal files provided while {asc.numNoiseSources} are needed.')
    if asc.numNodes != len(settings.asynchronicity.SROsppm):
        if (np.array(settings.asynchronicity.SROsppm) == 0).all():
            settings.asynchronicity.SROsppm = np.zeros(asc.numNodes)
        else:
            raise ValueError('Number of nodes does not match number of given non-zero SRO values.')
    if asc.numNodes != len(settings.asynchronicity.STOinducedDelays):
        if (np.array(settings.asynchronicity.STOinducedDelays) == 0).all():
            settings.asynchronicity.STOinducedDelays = np.zeros(asc.numNodes)
        else:
            raise ValueError('Number of nodes does not match number of given non-zero STO values.')

    # Adapt sampling frequency
    if asc.samplingFreq != settings.samplingFrequency:
        # Resample RIRs
        for ii in range(asc.rirDesiredToSensors.shape[1]):
            for jj in range(asc.rirDesiredToSensors.shape[2]):
                resampled = resampy.resample(asc.rirDesiredToSensors[:, ii, jj], asc.samplingFreq, settings.samplingFrequency)
                if ii == 0 and jj == 0:
                    rirDesiredToSensors_resampled = np.zeros((resampled.shape[0], asc.rirDesiredToSensors.shape[1], asc.rirDesiredToSensors.shape[2]))
                rirDesiredToSensors_resampled[:, ii, jj] = resampled
            for jj in range(asc.rirNoiseToSensors.shape[2]):
                resampled = resampy.resample(asc.rirNoiseToSensors[:, ii, jj], asc.samplingFreq, settings.samplingFrequency)
                if ii == 0 and jj == 0:
                    rirNoiseToSensors_resampled = np.zeros((resampled.shape[0], asc.rirNoiseToSensors.shape[1], asc.rirNoiseToSensors.shape[2]))
                rirNoiseToSensors_resampled[:, ii, jj] = resampled
        asc.rirDesiredToSensors = rirDesiredToSensors_resampled
        if asc.rirNoiseToSensors.shape[2] > 0:  # account for noiseless scenario
            asc.rirNoiseToSensors = rirNoiseToSensors_resampled
        # Modify ASC object parameter
        asc.samplingFreq = settings.samplingFrequency

    # Desired signal length [samples]
    signalLength = int(settings.signalDuration * asc.samplingFreq) 

    # Load + pre-process dry desired signals and build wet desired signals
    dryDesiredSignals = np.zeros((signalLength, asc.numDesiredSources))
    wetDesiredSignals = np.zeros((signalLength, asc.numDesiredSources, asc.numSensors))
    oVADsourceSpecific = np.zeros((signalLength, asc.numDesiredSources))
    for ii in range(asc.numDesiredSources):
        # Load signal
        rawSignal, fsRawSignal = sf.read(settings.desiredSignalFile[ii])

        # Pre-process (resample, truncate, whiten)
        dryDesiredSignals[:, ii] = pre_process_signal(rawSignal,
                                                    settings.signalDuration,
                                                    fsRawSignal,
                                                    asc.samplingFreq)

        # Convolve with RIRs to create wet signals - TO GET THE VAD
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]

        # Voice Activity Detection (pre-truncation/resampling)
        thrsVAD = np.amax(wetDesiredSignals[:, ii, 0] ** 2) / settings.VADenergyFactor
        oVADsourceSpecific[:, ii], _ = VAD.oracleVAD(wetDesiredSignals[:, ii, 0], settings.VADwinLength, thrsVAD, asc.samplingFreq)

        # Whiten dry signal 
        dryDesiredSignals[:, ii] = whiten(dryDesiredSignals[:, ii], oVADsourceSpecific[:, ii])

        # Convolve with RIRs to create wet signals - For actual use
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]


    # Get VAD consensus
    oVADsourceSpecific = np.sum(oVADsourceSpecific, axis=1)
    oVAD = np.zeros_like(oVADsourceSpecific)
    oVAD[oVADsourceSpecific == asc.numDesiredSources] = 1   # only set global VAD = 1 when all sources are active


    # Load + pre-process dry noise signals and build wet noise signals
    dryNoiseSignals = np.zeros((signalLength, asc.numNoiseSources))
    wetNoiseSignals = np.zeros((signalLength, asc.numNoiseSources, asc.numSensors))
    for ii in range(asc.numNoiseSources):

        rawSignal, fsRawSignal = sf.read(settings.noiseSignalFile[ii])
        tmp = pre_process_signal(rawSignal,
                                    settings.signalDuration,
                                    fsRawSignal,
                                    asc.samplingFreq)

        # Whiten signal 
        tmp = whiten(tmp, oVAD)

        # Set SNR
        dryNoiseSignals[:, ii] = 10 ** (-settings.baseSNR / 20) * tmp

        # Convolve with RIRs to create wet signals
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryNoiseSignals[:, ii], asc.rirNoiseToSensors[:, jj, ii])
            wetNoiseSignals[:, ii, jj] = tmp[:signalLength]

    # Build speech-only and noise-only signals
    wetNoise = np.sum(wetNoiseSignals, axis=1)      # sum all noise sources at each sensor
    wetSpeech = np.sum(wetDesiredSignals, axis=1)   # sum all speech sources at each sensor
    wetNoise_norm = wetNoise / np.amax(np.abs(wetNoise + wetSpeech))    # Normalize
    wetSpeech_norm = wetSpeech / np.amax(np.abs(wetNoise + wetSpeech))  # Normalize

    # --- Apply STOs / SROs ---
    wetNoise_norm, _, _ = apply_sro_sto(wetNoise_norm, asc.samplingFreq, asc.sensorToNodeTags,
                                settings.asynchronicity.SROsppm, settings.asynchronicity.STOinducedDelays)
    wetSpeech_norm, timeStampsSROs, fsSROs = apply_sro_sto(wetSpeech_norm, asc.samplingFreq, asc.sensorToNodeTags,
                                settings.asynchronicity.SROsppm, settings.asynchronicity.STOinducedDelays)
    # Set reference node (for master clock) based on SRO values
    masterClockNodeIdx = np.where(np.array(settings.asynchronicity.SROsppm) == 0)[0][0]
    # ------------------

    # Build sensor signals
    sensorSignals = wetSpeech_norm + wetNoise_norm

    # Add self-noise to microphones
    rng = np.random.default_rng(settings.randSeed)
    for k in range(sensorSignals.shape[-1]):
        selfnoise = 10 ** (settings.selfnoiseSNR / 20) * np.amax(np.abs(sensorSignals[:, k])) * whiten(rng.uniform(-1, 1, (signalLength,)))
        sensorSignals[:, k] += selfnoise
    
    # Build output class object
    signals = classes.Signals(dryNoiseSources=dryNoiseSignals,
                                drySpeechSources=dryDesiredSignals,
                                wetIndivNoiseSources=wetNoiseSignals,
                                wetIndivSpeechSources=wetDesiredSignals,
                                wetNoise=wetNoise_norm,
                                wetSpeech=wetSpeech_norm,
                                sensorSignals=sensorSignals,
                                VAD=oVAD,
                                sensorToNodeTags=asc.sensorToNodeTags,
                                fs=fsSROs,
                                referenceSensor=settings.referenceSensor,
                                timeStampsSROs=timeStampsSROs,
                                masterClockNodeIdx=masterClockNodeIdx
                                )
                                
    # Check validity of chosen reference sensor
    if (signals.nSensorPerNode < settings.referenceSensor + 1).any():
        conflictIdx = signals.nSensorPerNode[signals.nSensorPerNode < settings.referenceSensor + 1]
        raise ValueError(f'The reference sensor index chosen ({settings.referenceSensor}) conflicts with the number of sensors in node(s) {conflictIdx}.')

    return signals, asc
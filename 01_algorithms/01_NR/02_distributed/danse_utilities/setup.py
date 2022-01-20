from multiprocessing.sharedctypes import Value
import os, sys, time
from xml.dom import ValidationErr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import scipy.signal as sig
from pathlib import Path, PurePath
from . import classes           # <-- classes for DANSE
from . import danse_scripts     # <-- scripts for DANSE
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import *
from mySTFT.calc_STFT import calcISTFT
import VAD
from metrics import eval_enhancement


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
    mySignals, preEnhancSNRs, asc = generate_signals(settings)
    # Convert all DANSE input signals to the STFT domain
    mySignals.get_all_stfts(asc.samplingFreq, settings.stftWinLength, settings.stftEffectiveFrameLen)

    # Apply SROs
    ###############TODO

    # DANSE
    mySignals.desiredSigEst_STFT = danse(mySignals.sensorSignals_STFT, asc, settings, mySignals.VAD)
    
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


def evaluate_enhancement_outcome(sigs: classes.Signals, settings: classes.ProgramSettings):
    """Wrapper for computing and storing evaluation metrics after signal enhancement.
    Parameters
    ----------
    sigs : Signals object
        The signals before and after enhancement.
    settings : ProgramSettings object
        The settings for the current run.

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

    snr      = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Unweighted SNR
    fwSNRseg = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Frequency-weighted segmental SNR
    sisnr    = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Speech-Intelligibility-weighted SNR
    stoi     = dict([(key, []) for key in [f'Node{n + 1}' for n in range(numNodes)]])  # Short-Time Objective Intelligibility
    tStart = time.perf_counter()    # time computation
    for idxNode in range(numNodes):
        for idxSensor in range(numSensorsPerNode[idxNode]):
            trueIdxSensor = sum(numSensorsPerNode[:idxNode]) + idxSensor
            print(f'Computing signal enhancement evaluation metrics for node {idxNode + 1}/{numNodes} (sensor {idxSensor + 1}/{numSensorsPerNode[idxNode]})...')
            out0, out1, out2, out3 = eval_enhancement.get_metrics(
                                    sigs.wetSpeech[:, trueIdxSensor],
                                    sigs.desiredSigEst[:, idxNode], 
                                    sigs.fs,
                                    sigs.VAD,
                                    settings.gammafwSNRseg,
                                    settings.frameLenfwSNRseg
                                    )
            snr[f'Node{idxNode + 1}'].append(out0)
            fwSNRseg[f'Node{idxNode + 1}'].append(out1)
            sisnr[f'Node{idxNode + 1}'].append(out2)
            stoi[f'Node{idxNode + 1}'].append(out3)
    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    measures = classes.EnhancementMeasures(fwSNRseg=fwSNRseg,
                                            sisnr=sisnr,
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
    
    # mySignal = calcISTFT(mySignal_STFT,
    #                     win=np.hanning(settings.stftWinLength), 
    #                     N_STFT=settings.stftWinLength, 
    #                     R_STFT=settings.stftEffectiveFrameLen, 
    #                     sides='onesided')
    # t = np.arange(mySignal.shape[0]) / fs

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

    targetLength = np.round(settings.signalDuration * fs)
    if x.shape[0] < targetLength:
        x = np.concatenate((x, np.full((targetLength - x.shape[0], x.shape[1]), np.finfo(float).eps)))
    elif x.shape[0] > targetLength:
        x = x[:targetLength, :]

    t = np.arange(x.shape[0]) / fs

    return x, t


def danse(y_STFT, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD):
    """Main wrapper for DANSE computations.
    Parameters
    ----------
    y_STFT : [Nf x Nt x Ns] np.ndarray (complex)
        The microphone signals in the STFT domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [N x 1] np.ndarray (bool /or/ int)
        Voice Activity Detector output.

    Returns
    -------
    desiredSigEst_STFT : [Nf x Nt x Nn] np.ndarry (complex)
        STFT representation of the desired signal at each of the Nn nodes. 
    """

    # Compute frame-wise VAD
    oVADframes = np.zeros((y_STFT.shape[1]), dtype=bool)
    for ii in range(y_STFT.shape[1]):
        VADinFrame = oVAD[ii * settings.stftEffectiveFrameLen : (ii + 1) * settings.stftEffectiveFrameLen - 1]
        nZeros = sum(VADinFrame == 0)
        oVADframes[ii] = nZeros <= settings.stftEffectiveFrameLen / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

    # DANSE it up
    desiredSigEst_STFT, wkk, gkmk, z = danse_scripts.danse_sequential(y_STFT, asc, settings, oVADframes)
    # desiredSigEst_STFT, wkk, gkmk = danse_scripts.danse_sequential_old(y_STFT, asc, settings, oVADframes)

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(8,4))
    # for ii in range(z.shape[-1]):
    #     ax = fig.add_subplot(2,z.shape[-1]+1,ii+1)
    #     ax.plot(np.abs(z[:,:,ii]).T)
    #     plt.title(f'z_-k({ii+1})')
    # ax = fig.add_subplot(2,z.shape[-1]+1,z.shape[-1]+1)
    # ax.plot(np.abs(y_STFT[:,:,0]).T)
    # plt.title('y')
    # #
    # ax = fig.add_subplot(2,z.shape[-1]+1,z.shape[-1]+1  + 1)
    # ax.plot(np.abs(wkk[0][:,:,0]))
    # plt.show()

    stop = 1
    
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

    signalLength = desiredDuration * targetFs   # desired signal length [samples]
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
    asc = load_acoustic_scenario(settings.acousticScenarioPath,
                                settings.plotAcousticScenario,
                                settings.acScenarioPlotExportPath)

    # Detect conflicts
    if asc.numDesiredSources > len(settings.desiredSignalFile):
        raise ValueError(f'{settings.desiredSignalFile} "desired" signal files provided while {asc.numDesiredSources} are needed.')
    if asc.numNoiseSources > len(settings.noiseSignalFile):
        raise ValueError(f'{settings.noiseSignalFile} "noise" signal files provided while {asc.numNoiseSources} are needed.')

    # Desired signal length [samples]
    signalLength = settings.signalDuration * asc.samplingFreq   

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

    # Build sensor signals
    wetNoise = np.sum(wetNoiseSignals, axis=1)
    wetSpeech = np.sum(wetDesiredSignals, axis=1)
    wetNoise_norm = wetNoise / np.amax(wetNoise + wetSpeech)    # Normalize
    wetSpeech_norm = wetSpeech / np.amax(wetNoise + wetSpeech)  # Normalize
    sensorSignals = wetSpeech_norm + wetNoise_norm

    # Time vector
    timeVector = np.arange(signalLength) / asc.samplingFreq

    # Calculate SNRs
    SNRs = np.zeros(asc.numSensors)
    for sensorIdx in range(asc.numSensors):
        SNRs[sensorIdx] = eval_enhancement.getSNR(sensorSignals[:, sensorIdx], oVAD)

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
                                referenceSensor=settings.referenceSensor)
                                
    # Check validity of chosen reference sensor
    if (signals.nSensorPerNode < settings.referenceSensor + 1).any():
        conflictIdx = signals.nSensorPerNode[signals.nSensorPerNode < settings.referenceSensor + 1]
        raise ValueError(f'The reference sensor index chosen ({settings.referenceSensor}) conflicts with the number of sensors in node(s) {conflictIdx}.')

    return signals, SNRs, asc


def load_acoustic_scenario(csvFilePath, plotScenario=False, figExportPath=''):
    """Reads, interprets, and organizes a CSV file
    containing an acoustic scenario and returns its 
    contents.
    Parameters
    ----------
    csvFilePath : str
        Path to CSV file.   
    plotScenario : bool
        If true, plot visualization of acoustic scenario.       
    figExportPath : str
        Path to directory where to export the figure (only used if plotScenario is True).        

    Returns
    -------
    acousticScenario : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    """
    # Check whether path exists
    if not os.path.isfile(csvFilePath):
        raise ValueError(f'The path provided\n\t\t"{csvFilePath}"\ndoes not correspond to an existing file.')

    # Load dataframe
    acousticScenario_df = pd.read_csv(csvFilePath,index_col=0)

    # Count sources and receivers
    numSpeechSources = sum('Source' in s for s in acousticScenario_df.index)       # number of speech sources
    numNoiseSources  = sum('Noise' in s for s in acousticScenario_df.index)         # number of noise sources
    numSensors       = sum('Sensor' in s for s in acousticScenario_df.index)        # number of sensors

    # Extract single-valued data
    roomDimensions   = [f for f in acousticScenario_df.rd if not np.isnan(f)]               # room dimensions
    absCoeff         = [f for f in acousticScenario_df.alpha if not np.isnan(f)]            # absorption coefficient
    absCoeff         = absCoeff[1]
    samplingFreq     = [f for f in acousticScenario_df.Fs if not np.isnan(f)]               # sampling frequency
    samplingFreq     = int(samplingFreq[1])
    numNodes         = [f for f in acousticScenario_df.nNodes if not np.isnan(f)]           # number of nodes
    numNodes         = int(numNodes[1])
    distBtwSensors   = [f for f in acousticScenario_df.d_intersensor if not np.isnan(f)]    # inter-sensor distance
    distBtwSensors   = int(distBtwSensors[1])

    # Extract coordinates
    desiredSourceCoords = np.zeros((numSpeechSources,3))
    for ii in range(numSpeechSources):      # desired sources
        desiredSourceCoords[ii,0] = acousticScenario_df.x[f'Source {ii + 1}']
        desiredSourceCoords[ii,1] = acousticScenario_df.y[f'Source {ii + 1}']
        desiredSourceCoords[ii,2] = acousticScenario_df.z[f'Source {ii + 1}']
    sensorCoords = np.zeros((numSensors,3))
    for ii in range(numSensors):            # sensors
        sensorCoords[ii,0] = acousticScenario_df.x[f'Sensor {ii + 1}']
        sensorCoords[ii,1] = acousticScenario_df.y[f'Sensor {ii + 1}']
        sensorCoords[ii,2] = acousticScenario_df.z[f'Sensor {ii + 1}']
    noiseSourceCoords = np.zeros((numNoiseSources,3))
    for ii in range(numNoiseSources):       # noise sources
        noiseSourceCoords[ii,0] = acousticScenario_df.x[f'Noise {ii + 1}']
        noiseSourceCoords[ii,1] = acousticScenario_df.y[f'Noise {ii + 1}']
        noiseSourceCoords[ii,2] = acousticScenario_df.z[f'Noise {ii + 1}']

    # Prepare arrays to extract RIRs
    rirLength = int(1/numNodes * sum('h_sn' in s for s in acousticScenario_df.index))    # number of samples in one RIR
    rirDesiredToSensors = np.zeros((rirLength, numSensors, numSpeechSources))
    rirNoiseToSensors = np.zeros((rirLength, numSensors, numNoiseSources))

    # Identify dataframe columns containing RIR data
    idxDataColumns = [ii for ii in range(len(acousticScenario_df.columns)) if acousticScenario_df.columns[ii][:4] == 'Node']
    tagsSensors = np.zeros(numSensors, dtype=int)  # tags linking sensor to node

    # Start RIRs extraction from dataframe
    for idxEnum, idxColumn in enumerate(idxDataColumns):
        ref = acousticScenario_df.columns[idxColumn]
        nodeNumber = ref[4]
        sensorData = acousticScenario_df[ref]
        for jj in range(numSpeechSources):
            tmp = sensorData[f'h_sn{jj + 1}'].to_numpy()
            rirDesiredToSensors[:,idxEnum,jj] = tmp[~np.isnan(tmp)]    # source-to-node RIRs
        for jj in range(numNoiseSources):
            tmp = sensorData[f'h_nn{jj + 1}'].to_numpy()
            rirNoiseToSensors[:,idxEnum,jj] = tmp[~np.isnan(tmp)]    # noise-to-node RIRs
        tagsSensors[idxEnum] = nodeNumber

    acousticScenario = classes.AcousticScenario(rirDesiredToSensors, rirNoiseToSensors,
                                                desiredSourceCoords, sensorCoords, tagsSensors, noiseSourceCoords,
                                                roomDimensions, absCoeff,
                                                samplingFreq, numNodes, distBtwSensors)

    # Make a plot of the acoustic scenario
    if plotScenario:
        fig = acousticScenario.plot()
        fig.suptitle(csvFilePath[csvFilePath.rfind('/', 0, csvFilePath.rfind('/')) + 1:-4])
        plt.tight_layout()

    return acousticScenario


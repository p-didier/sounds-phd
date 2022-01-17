from argparse import _ActionsContainer
from copyreg import constructor
import os, sys
from . import classes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import soundfile as sf
import scipy.signal as sig
sys.path.append(os.path.join(os.path.expanduser('~'), 'py/sounds-phd/_general_fcts'))
from plotting.twodim import *

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
        return self

def runExperiment(settings: classes.ProgramSettings):
    """Wrapper function to run a DANSE-related experiment given certain settings.
    Parameters
    ----------
    settings : classes.ProgramSettings object
        The settings for the current run.

    Returns
    -------
    results : classes.Results object
        The experiment results.
    """

    # Generate base signals (and extract acoustic scenario)
    signals, acousticScenario = generate_signals(settings)

    # Apply SROs
    ###############TODO

    # DANSE
    results = danse(signals, acousticScenario, settings)

    # # Transform into a class
    # results = classes.Results(results)    

    return results

def danse(y, asc: AcousticScenario, settings: classes.ProgramSettings):
    """Main wrapper for DANSE computations.
    Parameters
    ----------
    y : [N x Ns] np.ndarray
        The microphone signals
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : classes.ProgramSettings object
        The settings for the current run.

    Returns
    -------
    #TODO
    """

    # Convert signals 

    results = 1
    
    return results

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
    settings : classes.ProgramSettings object
        The settings for the current run.

    Returns
    -------
    micSignals : [N x Nsensors] np.ndarray
        Sensor signals. 
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
    for ii in range(asc.numDesiredSources):
        
        rawSignal, fsRawSignal = sf.read(settings.desiredSignalFile[ii])
        dryDesiredSignals[:, ii] = pre_process_signal(rawSignal, settings.signalDuration, fsRawSignal, asc.samplingFreq)

        # Convolve with RIRs to create wet signals
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]

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
    micSignals = np.sum(wetNoiseSignals, axis=1) + np.sum(wetDesiredSignals, axis=1)

    # Normalize
    micSignals /= np.amax(micSignals)

    return micSignals, asc


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

    acousticScenario = AcousticScenario(rirDesiredToSensors, rirNoiseToSensors,
                                        desiredSourceCoords, sensorCoords, tagsSensors, noiseSourceCoords,
                                        roomDimensions, absCoeff,
                                        samplingFreq, numNodes, distBtwSensors)

    # Plot
    if plotScenario:
        
        scatsize = 20

        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(121)
        plot_side_room(ax, roomDimensions[0:2], 
                    desiredSourceCoords[:, [0,1]], sensorCoords[:, [0,1]], 
                    noiseSourceCoords[:, [0,1]], scatsize)
        ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
        #
        ax = fig.add_subplot(122)
        plot_side_room(ax, roomDimensions[1:], 
                    desiredSourceCoords[:, [1,2]], sensorCoords[:, [1,2]], 
                    noiseSourceCoords[:, [1,2]], scatsize)
        ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')
        #
        fig.suptitle(csvFilePath[csvFilePath.rfind('/', 0, csvFilePath.rfind('/')) + 1:-4])
        fig.tight_layout()
        # set_axes_equal(ax)
        plt.show()

    return acousticScenario


def plot_side_room(ax, rd2D, rs, r, rn, scatsize):
    """Plots a 2-D room side, showing the positions of
    sources and nodes inside of it.
    Parameters
    ----------
    ax : Axes handle
        Axes handle to plot on.
    rd2D : [2 x 1] list
        2-D room dimensions [m].
    rs : [Ns x 2] np.ndarray
        Desired (speech) source(s) coordinates [m]. 
    r : [N x 2] np.ndarray
        Sensor(s) coordinates [m]. 
    rn : [Nn x 2] np.ndarray
        Noise source(s) coordinates [m]. 
    scatsize : float
        Scatter plot marker size.
    """
    
    plot_room2D(ax, rd2D)
    for ii in range(rs.shape[0]):
        ax.scatter(rs[ii,0],rs[ii,1],s=scatsize,c='blue',marker='d')
        ax.text(rs[ii,0],rs[ii,1],"D%i" % (ii+1))
    for ii in range(rn.shape[0]):
        ax.scatter(rn[ii,0],rn[ii,1],s=scatsize,c='red',marker='P')
        ax.text(rn[ii,0],rn[ii,1],"N%i" % (ii+1))
    for ii in range(r.shape[0]):
        ax.scatter(r[ii,0],r[ii,1],s=scatsize,c='green',marker='o')
        ax.text(r[ii,0],r[ii,1],"S%i" % (ii+1))
    ax.grid()
    ax.axis('equal')
    return None
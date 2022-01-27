import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from class_methods import dataclass_methods 
from plotting.twodim import plot_side_room


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
        
    @classmethod
    def load(cls, filename: str):
        return dataclass_methods.load(cls, filename)
    def save(self, filename: str):
        dataclass_methods.save(self, filename)

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


class micArrayAttributes:
    def __init__(self, Mk, arraygeom, mic_sep):
        self.Mk = Mk
        self.array = arraygeom
        self.mic_sep = mic_sep

@dataclass
class ASCProgramSettings:
    """Class for keeping track of global simulation settings"""
    roomDimBounds: list                             # [smallest, largest] room dimension possible [m]
    numScenarios: int = 1                           # number of AS to generate
    samplingFrequency: int = 16e3                   # sampling frequency [samples/s]
    rirLength: int = 2**12                          # RIR length [samples]
    numSpeechSources: int = 1                       # nr. of speech sources
    numNoiseSources: int = 1                        # nr. of noise sources
    numNodes: int = 3                               # nr. of nodes
    numSensorPerNode: np.ndarray = np.array([1])    # nr. of sensor per node
    revTime: float = 0.0                            # reverberation time [s]
    arrayGeometry: str = 'linear'                   # microphone array geometry (only used if numSensorPerNode > 1)
    sensorSeparation: float = 0.05                  # separation between sensor in array (only used if numSensorPerNode > 1)
    seed: int = 12345                               # seed for random generator
    
    @classmethod
    def load(cls, filename: str):
        return dataclass_methods.load(cls, filename)
    def save(self, filename: str):
        dataclass_methods.save(self, filename)
    
    def __post_init__(self):
        """Initial program settings checks"""
        if isinstance(self.numSensorPerNode, int):  # case where all nodes have the same number of sensors
            self.numSensorPerNode = np.full((self.numNodes,), self.numSensorPerNode)
        elif len(self.numSensorPerNode) == 1:       # case where all nodes have the same number of sensors
            self.numSensorPerNode = np.full((self.numNodes,), self.numSensorPerNode[0])
        elif len(self.numSensorPerNode) != self.numNodes:
            raise ValueError('Each node should have a number of nodes assigned to it.')
        # Reverberation time check
        nMinSamples = int(self.revTime * self.samplingFrequency)
        if self.rirLength < nMinSamples:
            print(f'Defined RIR length ({self.rirLength} samples) is too small for the defined reverberation time ({self.revTime} s).')
            print(f'Increasing RIR length to {nMinSamples} samples.')
            self.rirLength = nMinSamples
        return self

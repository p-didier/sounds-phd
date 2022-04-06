import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
if not any("_general_fcts" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/_general_fcts')
import class_methods.dataclass_methods as met
from plotting.twodim import plot_side_room


@dataclass
class AcousticScenario():
    """Class for keeping track of acoustic scenario parameters"""
    rirDesiredToSensors: np.ndarray = np.array([1])     # RIRs between desired sources and sensors
    rirNoiseToSensors: np.ndarray = np.array([1])       # RIRs between noise sources and sensors
    desiredSourceCoords: np.ndarray = np.array([1])     # Coordinates of desired sources
    sensorCoords: np.ndarray = np.array([1])            # Coordinates of sensors
    sensorToNodeTags: np.ndarray = np.array([1])        # Tags relating each sensor to its node
    noiseSourceCoords: np.ndarray = np.array([1])       # Coordinates of noise sources
    roomDimensions: np.ndarray = np.array([1])          # Room dimensions   
    absCoeff: float = 1.                                # Absorption coefficient
    samplingFreq: float = 16000.                        # Sampling frequency
    numNodes: int = 2                                   # Number of nodes in network
    distBtwSensors: float = 0.05                        # Distance btw. sensors at one node

    def __post_init__(self):
        self.numDesiredSources = self.desiredSourceCoords.shape[0]      # number of desired sources
        self.numSensors = self.sensorCoords.shape[0]                    # number of sensors
        self.numNoiseSources = self.noiseSourceCoords.shape[0]          # number of noise sources
        self.numSensorPerNode = np.unique(self.sensorToNodeTags, return_counts=True)[-1]    # number of sensors per node
        return self
    
    # Save and load
    def load(self, foldername: str):
        return met.load(self, foldername)
    def save(self, filename: str):
        met.save(self, filename)

    def plot(self):

        fig, (a0, a1) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]})
        plot_side_room(a0[0], self.roomDimensions[0:2], 
                    self.desiredSourceCoords[:, [0,1]], 
                    self.noiseSourceCoords[:, [0,1]], 
                    self.sensorCoords[:, [0,1]],
                    self.sensorToNodeTags,
                    dotted=self.absCoeff==1)
        a0[0].set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
        #
        plot_side_room(a0[1], self.roomDimensions[1:], 
                    self.desiredSourceCoords[:, [1,2]], 
                    self.noiseSourceCoords[:, [1,2]],
                    self.sensorCoords[:, [1,2]],
                    self.sensorToNodeTags,
                    dotted=self.absCoeff==1)
        a0[1].set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')
        # Add info
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        boxText = 'Node distances\n\n'
        for ii in range(self.numNodes):
            for jj in range(self.desiredSourceCoords.shape[0]):
                d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.desiredSourceCoords[jj,:]))
                boxText += f'{ii + 1}$\\to$D{jj + 1}={np.round(d, 2)}m\n'
            for jj in range(self.noiseSourceCoords.shape[0]):
                d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.noiseSourceCoords[jj,:]))
                boxText += f'{ii + 1}$\\to$N{jj + 1}={np.round(d, 2)}m\n'
            boxText += '\n'
        boxText = boxText[:-1]
        # Plot RIRs
        t = np.arange(self.rirDesiredToSensors.shape[0]) / self.samplingFreq
        ymax = np.amax([np.amax(self.rirDesiredToSensors[:, 0, 0]), np.amax(self.rirNoiseToSensors[:, 0, 0])])
        ymin = np.amin([np.amin(self.rirDesiredToSensors[:, 0, 0]), np.amin(self.rirNoiseToSensors[:, 0, 0])])
        a1[0].plot(t, self.rirDesiredToSensors[:, 0, 0], 'k')
        a1[0].grid()
        a1[0].set(xlabel='$t$ [s]', title=f'RIR node 1 - D1')
        a1[0].set_ylim([ymin, ymax])
        a1[1].plot(t, self.rirNoiseToSensors[:, 0, 0], 'k')
        a1[1].grid()
        a1[1].set(xlabel='$t$ [s]', title=f'RIR node 1 - N1')
        a1[1].set_ylim([ymin, ymax])
        # Add text boxes
        a1[1].text(1.1, 0.9, f'Abs. coeff.:\n$\\alpha$ = {np.round(self.absCoeff, 2)}', transform=a1[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        a0[1].text(1.1, 0.9, boxText, transform=a0[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
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
    specialCase: str = ''                           # flag for special cases: '' or 'none' --> no special case;
                                                    # 'allNodesInSamePosition': all nodes and sensors are placed at the same position in space.
    
    @classmethod
    def load(cls, filename: str):
        return met.load(cls, filename)
    def save(self, filename: str):
        met.save(self, filename)
    
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

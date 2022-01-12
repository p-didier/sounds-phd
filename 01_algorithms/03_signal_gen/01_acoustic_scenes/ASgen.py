# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import random
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as rot
from rimPypack.rimPy import rimPy
sys.path.append(os.path.join(os.path.expanduser('~'), 'py/sounds-phd/_general_fcts'))
from plotting.threedim import plot_room
from class_methods import dataclass_methods 


# Acoustic Scenario (AS) generation script.

def main():

    # Define settings
    sets = ProgramSettings(
        numScenarios = 1,               # Number of AS to generate
        samplingFrequency = 16e3,       # Sampling frequency [samples/s]
        rirLength = 2**12,              # RIR length [samples]
        roomDimBounds = [3,7],          # [Smallest, largest] room dimension possible [m]
        numSpeechSources = 1,           # nr. of speech sources
        numNoiseSources = 1,            # nr. of noise sources
        numNodes = 5,                   # nr. of nodes
        numSensorPerNode = 1,           # nr. of sensor per node
        arrayGeometry = 'linear',       # microphone array geometry (only used if numSensorPerNode > 1)
        sensorSeparation = 0.05,        # separation between sensor in array (only used if numSensorPerNode > 1)
        revTime = 0.0,                   # reverberation time [s]
        seed = 12345                    # seed for random generator
    )

    # Local booleans
    plotit = True        # If true, plots the AS

    # Prepare export
    expFolder = f"{os.getcwd()}/sounds-phd/02_data/01_acoustic_scenarios/J{sets.numNodes}Mk{sets.numSensorPerNode}_Ns{sets.numSpeechSources}_Nn{sets.numNoiseSources}"
    if not os.path.isdir(expFolder):   # check if subfolder exists
        os.mkdir(expFolder)   # if not, make directory
    nas = len(next(os.walk(expFolder))[2])   # count only files in export dir
    fname = f"{expFolder}/AS{nas}"  # file name
    if sets.revTime == 0:
        fname += '_anechoic'
    fname += '.csv'

    # Generate scenarios
    counter = 0 # counter the number of acoustic scenarios generated
    while counter < sets.numScenarios:
        
        # Generate RIRs
        h_ns, h_nn, rs, rn, r, rd = genAS(sets, plotit=plotit)
        
        # Prepare header for CSV export
        header = {'rd': pd.Series(np.squeeze(rd)),
                    'RT': sets.revTime,
                    'Fs': sets.samplingFrequency,
                    'nNodes': sets.numNodes, 
                    'd_intersensor': sets.sensorSeparation}
        #  
        export_data(h_ns, h_nn, header, rs, rn, r, fname)
        sets.save(f'{expFolder}/simulSettings')      # Save settings for potential re-run

        counter += 1

    print('\n\nAll done.')

    return h_ns, h_nn
        

class micArrayAttributes:
    def __init__(self, Mk, arraygeom, mic_sep):
        self.Mk = Mk
        self.array = arraygeom
        self.mic_sep = mic_sep

@dataclass
class ProgramSettings:
    """Class for keeping track of global simulation settings"""
    roomDimBounds: list                 # [Smallest, largest] room dimension possible [m]
    numScenarios: int = 1               # Number of AS to generate
    samplingFrequency: int = 16e3       # Sampling frequency [samples/s]
    rirLength: int = 2**12              # RIR length [samples]
    numSpeechSources: int = 1           # nr. of speech sources
    numNoiseSources: int = 1            # nr. of noise sources
    numNodes: int = 3                   # nr. of nodes
    numSensorPerNode: int = 1           # nr. of sensor per node
    revTime: float = 0.0                # reverberation time [s]
    arrayGeometry: str = 'linear'       # microphone array geometry (only used if numSensorPerNode > 1)
    sensorSeparation: float = 0.05      # separation between sensor in array (only used if numSensorPerNode > 1)
    seed: int = 12345                   # seed for random generator
    
    @classmethod
    def load(cls, filename: str):
        return dataclass_methods.load(cls, filename)
    def save(self, filename: str):
        dataclass_methods.save(self, filename)


def genAS(sets: ProgramSettings,plotit=False):
    """Computes the RIRs in a rectangular cavity where sensors, speech
    sources, and noise sources are present.

    Parameters
    ----------
    sets : classes.ProgramSettings object.
        The settings for the current run.
    plotit : bool.
        If true, plots the scenario in a figure.

    Returns
    -------
    rirSpeechToNodes : [RIR_l x J x Ns] array of complex floats
        RIRs between desired sources and sensors.  
    rirNoiseToNodes : [RIR_l x J x Nn] array of complex floats
        RIRs between noise sources and sensors.
    speechSourceCoords : [Ns x 3] array of real floats
        Speech source(s) coordinates [m].
    noiseSourceCoords : [Nn x 3] array of real floats
        Noise source(s) coordinates [m].
    nodesCoords : [J x 3] array of real floats
        Node(s) coordinates [m].
    roomDimensions : [3,] list of real floats
        Room dimensions [m].
    """

    # Create random generator
    rng = np.random.default_rng(sets.seed)

    # Create node array
    arrayAttrib = micArrayAttributes(Mk=sets.numSensorPerNode, 
                        arraygeom=sets.arrayGeometry, 
                        mic_sep=sets.sensorSeparation)

    # Room parameters
    roomDimensions = rng.uniform(sets.roomDimBounds[0], sets.roomDimBounds[1], size=(3,))    # Generate random room dimensions
    roomVolume = np.prod(roomDimensions)                                   # Room volume
    roomSurface = 2*(roomDimensions[0] * roomDimensions[1] +
                    roomDimensions[0] * roomDimensions[2] +
                    roomDimensions[1] * roomDimensions[2])   # Total room surface area
    if sets.revTime == 0:
        absorbCoeff = 1
    else:
        absorbCoeff = np.minimum(1, 0.161 * roomVolume / (sets.revTime * roomSurface))  # Absorption coefficient of the walls (Sabine's equation)
    
    # Random element positioning in 3-D space
    speechSourceCoords = np.multiply(rng.uniform(0, 1, (sets.numSpeechSources, 3)), roomDimensions)
    nodesCoords        = np.multiply(rng.uniform(0, 1, (sets.numNodes, 3)), roomDimensions)
    noiseSourceCoords  = np.multiply(rng.uniform(0, 1, (sets.numNoiseSources, 3)), roomDimensions)
    
    # Generate sensor arrays
    totalNumNodes = sets.numNodes * arrayAttrib.Mk
    sensorsCoords = np.zeros((totalNumNodes, 3))
    for ii in range(sets.numNodes):
        sensorsCoords[ii * arrayAttrib.Mk : (ii + 1) * arrayAttrib.Mk,:] = \
            generate_array_pos(nodesCoords[ii, :], arrayAttrib, rng)

    # If asked, show geometry on plot
    if plotit:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_room(ax, roomDimensions)
        ax.scatter(sensorsCoords[:,0],sensorsCoords[:,1],sensorsCoords[:,2], c='green')
        ax.scatter(speechSourceCoords[:,0], speechSourceCoords[:,1], speechSourceCoords[:,2], c='blue')
        ax.scatter(noiseSourceCoords[:,0], noiseSourceCoords[:,1], noiseSourceCoords[:,2], c='red')
        ax.grid()
        plt.show()

    # Walls reflection coefficient  
    reflectionCoeff = -1*np.sqrt(1 - absorbCoeff)   
    
    # Compute RIRs from speech source to sensors
    rirSpeechToNodes = np.zeros((sets.rirLength, totalNumNodes, sets.numSpeechSources))
    for ii in range(sets.numSpeechSources):
        print('Computing RIRs from speech source #%i at %i sensors' % (ii+1, totalNumNodes))
        rirSpeechToNodes[:,:,ii] = rimPy(sensorsCoords, speechSourceCoords[ii,:], 
        roomDimensions, reflectionCoeff, 
        sets.rirLength/sets.samplingFrequency, sets.samplingFrequency)
    # Compute RIRs from noise source to sensors
    rirNoiseToNodes = np.zeros((sets.rirLength, totalNumNodes, sets.numNoiseSources))
    for ii in range(sets.numNoiseSources):
        print('Computing RIRs from noise source #%i at %i sensors' % (ii+1, totalNumNodes))
        rirNoiseToNodes[:,:,ii] = rimPy(sensorsCoords, noiseSourceCoords[ii,:], 
        roomDimensions, reflectionCoeff, 
        sets.rirLength/sets.samplingFrequency, sets.samplingFrequency)
    
    return rirSpeechToNodes, rirNoiseToNodes, speechSourceCoords, noiseSourceCoords, sensorsCoords, roomDimensions

  
def export_data(h_sn, h_nn, header, rs, rn, r, fname):
    """Exports the necessary acoustic scenario data as CSV for 
    later use in other simulations.

    Parameters
    ----------
    h_sn : [RIR_l x J x Ns] array of complex floats
        RIRs between desired sources and sensors.  
    h_nn : [RIR_l x J x Nn] array of complex floats
        RIRs between noise sources and sensors.
    header : dictionary.
        Other important information to exploit the exported data.
    rs : [Ns x 3] array of real floats
        Speech source(s) coordinates [m].
    rn : [Nn x 3] array of real floats
        Noise source(s) coordinates [m].
    r : [J x 3] array of real floats
        Node(s) coordinates [m].
    fname : str.
        Name of file to be exported.
    """

    # Check if export folder exists
    os.path.realpath(fname)
    mydir = str(Path(fname).parent)
    if not os.path.isdir(mydir):
        os.mkdir(mydir)   # if not, make directory
        print(f'Directory "{mydir}" was created.')

    # Source-to-node RIRs
    data_df = pd.DataFrame()   # init dataframe
    for ii in range(h_sn.shape[-1]):
        # Build dataframe for current matrix slice
        data_df_curr = pd.DataFrame(np.squeeze(h_sn[:,:,ii]),\
            columns=['Sensor %i' % (idx+1) for idx in range(h_sn.shape[1])],\
            index=['h_sn %i' % (ii+1) for idx in range(h_sn.shape[0])])

        # Concatenate to global dataframe
        data_df = pd.concat([data_df,data_df_curr])

    # Noise-to-node RIRs
    for ii in range(h_nn.shape[-1]):
        # Build dataframe for current matrix slice
        data_df_curr = pd.DataFrame(np.squeeze(h_nn[:,:,ii]),\
            columns=['Sensor %i' % (idx+1) for idx in range(h_nn.shape[1])],\
            index=['h_nn %i' % (ii+1) for idx in range(h_nn.shape[0])])

        # Concatenate to global dataframe
        data_df = pd.concat([data_df,data_df_curr])
            
    # Build header dataframe
    header_df = pd.DataFrame(header)

    # Build coordinates dataframes
    rs_df = pd.DataFrame(rs,\
                index=['Source %i' % (idx+1) for idx in range(rs.shape[0])],\
                columns=['x','y','z'])
    rn_df = pd.DataFrame(rn,\
                index=['Noise %i' % (idx+1) for idx in range(rn.shape[0])],\
                columns=['x','y','z'])
    r_df = pd.DataFrame(r,\
                index=['Sensor %i' % (idx+1) for idx in range(r.shape[0])],\
                columns=['x','y','z'])

    # Concatenate header + coordinates with rest of data
    big_df = pd.concat([header_df,rs_df,rn_df,r_df,data_df])
    big_df.to_csv(fname)

    print('Data exported to CSV: "%s"' % Path(fname).name)
    print('Find it in folder: %s' % str(Path(fname).parent))


def generate_array_pos(nodeCoords, arrayAttrib: micArrayAttributes, randGenerator: np.random._generator.Generator, force2D=False):
    """Define node positions based on node position, number of nodes, and array type

    Parameters
    ----------
    nodeCoords : [J x 3] array of real floats.
        Nodes coordinates in 3-D space [m].
    arrayAttrib : micArrayAttributes object.
        Characteristics of the sensor array at each node.
    randGenerator : NumPy random generator.
        Random generator with pre-specified seed.
    force2D : bool.
        If true, projects the sensor coordinates on the z=0 plane.

    Returns
    -------
    sensorCoordsRotated : [(J*arrayAttrib.Mk) x 3] array of real floats.
        Sensor coordinates in 3-D space [m].
    """

    if arrayAttrib.array == 'linear':
        # 1D local geometry
        x = np.linspace(start=0, stop=arrayAttrib.Mk * arrayAttrib.mic_sep, num=arrayAttrib.Mk)
        # Center
        x -= np.mean(x)
        # Make 3D
        sensorCoords = np.zeros((arrayAttrib.Mk,3))
        sensorCoords[:,0] = x
        
        # Rotate in 3D through randomized rotation vector 
        rotvec = randGenerator.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        sensorCoordsRotated = np.zeros_like(sensorCoords)
        for ii in range(arrayAttrib.Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            sensorCoordsRotated[ii,:] = myrot.apply(sensorCoords[ii, :]) + nodeCoords
    else:
        raise ValueError('No sensor array geometry defined for array type "%s"' % arrayAttrib.array)

    return sensorCoordsRotated


if __name__ == '__main__':
    sys.exit(main())
# %%

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Acoustic Scenario (AS) generation script.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import os, sys, copy
from pathlib import Path, PurePath
from scipy.spatial.transform import Rotation as rot
from rimPypack.rimPy import rimPy
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from utilsASC.classes import *

# asc = ASCProgramSettings()
# asc = asc.load(r'U:\py\sounds-phd\02_data\01_acoustic_scenarios\tests\J2Mk[3, 1]_Ns1_Nn1\AS1_anechoic')

# Define settings
sets = ASCProgramSettings(
    numScenarios=10,                   # Number of AS to generate
    samplingFrequency=16e3,           # Sampling frequency [samples/s]
    rirLength=2**12,                  # RIR length [samples]
    roomDimBounds=[5,5],              # [Smallest, largest] room dimension possible [m]
    # roomDimBounds=[10,10],              # [Smallest, largest] room dimension possible [m]
    minDistFromWalls = 0.25,
    minDistBwNodes = 1,               # minimum distance between nodes [m]
    maxDistFromNoise = 1.5,           # maximum distance between nodes and noise source [m] (only if `numNoiseSources==1`)
    numSpeechSources=1,               # nr. of speech sources
    numNoiseSources=2,                # nr. of noise sources
    numNodes=4,                       # nr. of nodes
    numSensorPerNode=[1,3,2,5],               # nr. of sensor per node,
    # numSensorPerNode=[3,3],               # nr. of sensor per node,
    # numSensorPerNode=[1,1],               # nr. of sensor per node,
    # numSensorPerNode=[1,2,3],               # nr. of sensor per node,
    # numSensorPerNode=1,               # nr. of sensor per node,
    # arrayGeometry='linear',         # microphone array geometry (only used if numSensorPerNode > 1)
    # arrayGeometry='radius',           # microphone array geometry (only used if numSensorPerNode > 1)
    arrayGeometry='grid3d',           # microphone array geometry (only used if numSensorPerNode > 1)
    sensorSeparation=0.2,            # separation between sensor in array (only used if numSensorPerNode > 1) [m]
    revTime=0.15,                      # reverberation time [s]
    # specialCase='allNodesInSamePosition'    # special cases
)

basepath = f'{pathToRoot}/02_data/01_acoustic_scenarios/for_submissions/icassp2023'
# customASCname = 'testforTIDANSE'
customASCname = None
plotit = 1  
exportit = 1    # If True, export the ASC, settings, and figures.
globalSeed = int("".join([str(integer) for integer in sets.numSensorPerNode]))  # use number of sensors per node to create seed [python: https://www.adamsmith.haus/python/answers/how-to-convert-a-list-of-integers-into-a-single-integer-in-python]

def main(sets, basepath, globalSeed, plotit=True, exportit=True):
    """Main wrapper for acoustic scenarios generation.
    
    Parameters
    ----------
    sets : ASCProgramSettings object
        Settings for the generation of a specific acoustic scenario (ASC).
    basepath : str
        Base directory where to export the ASC data.
    globalSeed : int
        Global random generator seed (used to generate random sub-generators, one for each ASC)
    plotit: bool
        If True, plots the ASC in figure.
    exportit: bool
        If True, export the ASC, settings, and figures.
    """

    # Create random generator
    rngGlobal = np.random.default_rng(globalSeed)

    # Export folder
    if customASCname is None:
        Mks = '['
        for ii in range(len(sets.numSensorPerNode)):
            Mks += str(sets.numSensorPerNode[ii])
            if ii < len(sets.numSensorPerNode) - 1:
                Mks += '_'
        Mks += ']'
        expFolder = f"{basepath}/J{sets.numNodes}Mk{Mks}_Ns{sets.numSpeechSources}_"
        if sets.numNoiseSources > 0:
            expFolder += f'Nn{sets.numNoiseSources}'
        else:
            expFolder += 'noiseless'
    else:
        expFolder = f'{basepath}/{customASCname}'

    if not os.path.isdir(expFolder):   # check if subfolder exists
        os.mkdir(expFolder)   # if not, make directory

    # Generate scenarios
    counter = 0 # counter the number of acoustic scenarios generated
    while counter < sets.numScenarios:
        print(f'Generating ASC {counter + 1}/{sets.numScenarios}...')
        # Get sub-seed
        sets.seed = rngGlobal.integers(low=0, high=9999, size=1)
        # Generate RIRs
        asc = genAS(sets)
        
        # Folder name
        nas = len(next(os.walk(expFolder))[1])   # count only files in export dir
        foldername = f"{expFolder}/AS{nas + 1}"  # file name
        if sets.specialCase not in ['', 'none']:
            foldername += f'_{sets.specialCase}'
        if sets.revTime == 0:
            foldername += '_anechoic'
        else:
            foldername += f'_RT{int(sets.revTime * 1e3)}ms'
        # Export
        if exportit:
            asc.save(foldername)
            sets.save(foldername)
        # Plot
        if plotit:
            # Determine appropriate node radius
            fig = asc.plot(options=PlottingOptions(
                nodesColors='multi'
            ))
            # plt.show()
            if exportit:
                fig.savefig(f'{foldername}/schematic.pdf')
                fig.savefig(f'{foldername}/schematic.png')
                print('Acoustic scenario plotted and figure exported.')
            else:
                plt.show()
        counter += 1

    print('All done.')

    return None
        

def genAS(sets: ASCProgramSettings):
    """Computes the RIRs in a rectangular cavity where sensors, speech
    sources, and noise sources are present.

    Parameters
    ----------
    sets : ASCProgramSettings object.
        The settings for the current run.
    plotit : bool.
        If true, plots the scenario in a figure.

    Returns
    -------
    acousticScenario : AcousticScenario object
        Acoustic scenario (ASC).
    """

    # Create random generator
    rng = np.random.default_rng(sets.seed)

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
    speechSourceCoords = np.multiply(rng.uniform(0, 1, (sets.numSpeechSources, 3)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
    noiseSourceCoords  = np.multiply(rng.uniform(0, 1, (sets.numNoiseSources, 3)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
    if sets.specialCase == 'allNodesInSamePosition':
        # Special case: put all nodes and sensors at the same position
        nodesCoords = np.multiply(rng.uniform(0, 1, (1, 3)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
        nodesCoords = np.repeat(nodesCoords, sets.numNodes, axis=0)
        sets.sensorSeparation = 0.0
    else:
        nodesCoords = np.zeros((sets.numNodes, 3))
        if sets.numNoiseSources == 1:
            # Consider maximum distance from noise source
            for ii in range(sets.numNodes):
                distToNoiseSource = np.inf
                while distToNoiseSource > sets.maxDistFromNoise:
                    nodesCoords[ii, :] = np.multiply(rng.uniform(0, 1, (1, 3)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
                    distToNoiseSource = np.sqrt(np.sum((nodesCoords[ii, :] - noiseSourceCoords) ** 2))
        else:
            # Consider minimum distance between nodes
            for ii in range(sets.numNodes):
                curr = np.multiply(rng.uniform(0, 1, (3,)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
                if ii > 0:
                    dists = np.array([np.sqrt(np.sum((nodesCoords[jj, :] - curr) ** 2)) for jj in range(ii)])
                    while any(dists < sets.minDistBwNodes):
                        curr = np.multiply(rng.uniform(0, 1, (3,)), roomDimensions - 2 * sets.minDistFromWalls) + sets.minDistFromWalls
                        dists = np.array([np.sqrt(np.sum((nodesCoords[jj, :] - curr) ** 2)) for jj in range(ii)])
                nodesCoords[ii, :] = curr
    
    # Generate sensor arrays
    totalNumSensors = np.sum(sets.numSensorPerNode, dtype=int)
    sensorsCoords = np.zeros((totalNumSensors, 3))
    sensorNodeTags = np.zeros(totalNumSensors, dtype=int)     # tags linking each sensor to its node
    for ii in range(sets.numNodes):

        # Current node's sensors indices
        idxStart = np.sum(sets.numSensorPerNode[:ii], dtype=int)
        idxEnd = idxStart + sets.numSensorPerNode[ii]

        # Create node array
        arrayAttrib = micArrayAttributes(Mk=sets.numSensorPerNode[ii], 
                                        arraygeom=sets.arrayGeometry, 
                                        mic_sep=sets.sensorSeparation)
        # Derive coordinates
        sensorsCoords[idxStart : idxEnd,:] = generate_array_pos(nodesCoords[ii, :], arrayAttrib, rng)
        # Assign node tag
        sensorNodeTags[idxStart : idxEnd] = ii + 1

    # Walls reflection coefficient  
    reflectionCoeff = -1 * np.sqrt(1 - absorbCoeff)   
    
    # Compute RIRs from speech source to sensors
    rirSpeechToNodes = np.zeros((sets.rirLength, totalNumSensors, sets.numSpeechSources))
    for ii in range(sets.numSpeechSources):
        print('Computing RIRs from speech source #%i at %i sensors' % (ii+1, totalNumSensors))
        rirSpeechToNodes[:,:,ii] = rimPy(sensorsCoords, speechSourceCoords[ii,:], 
                                        roomDimensions, reflectionCoeff, 
                                        sets.rirLength/sets.samplingFrequency, 
                                        sets.samplingFrequency)
    # Compute RIRs from noise source to sensors
    rirNoiseToNodes = np.zeros((sets.rirLength, totalNumSensors, sets.numNoiseSources))
    for ii in range(sets.numNoiseSources):
        print('Computing RIRs from noise source #%i at %i sensors' % (ii+1, totalNumSensors))
        rirNoiseToNodes[:,:,ii] = rimPy(sensorsCoords, noiseSourceCoords[ii,:], 
                                        roomDimensions, reflectionCoeff, 
                                        sets.rirLength/sets.samplingFrequency, 
                                        sets.samplingFrequency)

    # Build output object
    acousticScenario = AcousticScenario(
        rirDesiredToSensors=rirSpeechToNodes,
        rirNoiseToSensors=rirNoiseToNodes,
        desiredSourceCoords=speechSourceCoords,
        sensorCoords=sensorsCoords,
        sensorToNodeTags=sensorNodeTags,
        noiseSourceCoords=noiseSourceCoords,
        roomDimensions=roomDimensions,
        absCoeff=absorbCoeff,
        samplingFreq=sets.samplingFrequency,
        numNodes=sets.numNodes,
        distBtwSensors=sets.sensorSeparation,
    )

    return acousticScenario


def generate_array_pos(nodeCoords, arrayAttrib: micArrayAttributes, randGenerator, force2D=False):
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
    sensorCoords : [(J*arrayAttrib.Mk) x 3] array of real floats.
        Sensor coordinates in 3-D space [m].
    """

    if arrayAttrib.array == 'linear':
        # 1D local geometry
        x = np.linspace(start=0, stop=arrayAttrib.Mk * arrayAttrib.mic_sep, num=arrayAttrib.Mk)
        # Center
        x -= np.mean(x)
        # Make 3D
        sensorCoordsBeforeRot = np.zeros((arrayAttrib.Mk,3))
        sensorCoordsBeforeRot[:,0] = x
        
        # Rotate in 3D through randomized rotation vector 
        rotvec = randGenerator.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        sensorCoords = np.zeros_like(sensorCoordsBeforeRot)
        for ii in range(arrayAttrib.Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            sensorCoords[ii,:] = myrot.apply(sensorCoordsBeforeRot[ii, :]) + nodeCoords
    elif arrayAttrib.array == 'radius':
        radius = arrayAttrib.mic_sep 
        sensorCoords = np.zeros((arrayAttrib.Mk,3))
        for ii in range(arrayAttrib.Mk):
            flag = False
            while not flag:
                r = randGenerator.uniform(low=0, high=radius, size=(3,))
                if np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) <= radius:
                    sensorCoords[ii, :] = r + nodeCoords - radius/2
                    flag = True
    elif arrayAttrib.array == 'grid3d':
        sensorCoords = np.zeros((arrayAttrib.Mk, 3))
        # Create grid
        d = arrayAttrib.mic_sep
        x_ = np.linspace(nodeCoords[0] - d, nodeCoords[0] + d, 3)
        y_ = np.linspace(nodeCoords[1] - d, nodeCoords[1] + d, 3)
        z_ = np.linspace(nodeCoords[2] - d, nodeCoords[2] + d, 3)
        x, y, z = np.meshgrid(x_, y_, z_)
        # Flatten coordinates
        coordFlat = np.zeros((np.prod(x.shape), 3))
        counter = 0
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    coordFlat[counter, :] = [x[ii,jj,kk], y[ii,jj,kk], z[ii,jj,kk]]
                    counter += 1
        # Base configuration ("atomic" -- see Word journal week 39, MON)
        idx = [13,4,22,10,16,14,12]
        sensorCoords[:np.amin([len(idx), arrayAttrib.Mk]), :] = coordFlat[idx[:np.amin([len(idx), arrayAttrib.Mk])], :]
        if len(idx) < arrayAttrib.Mk:
            allIdx = np.arange(coordFlat.shape[0])
            idxValid = [ii for ii in allIdx if ii not in idx]
            for ii in range(arrayAttrib.Mk - len(idx)):
                sensorCoords[np.amin([len(idx), arrayAttrib.Mk]) + ii, :] = coordFlat[idxValid[ii], :]
    else:
        raise ValueError('No sensor array geometry defined for array type "%s"' % arrayAttrib.array)

    return sensorCoords

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(sets, basepath, globalSeed, plotit, exportit))
# ------------------------------------------------------------------------------------
# %%

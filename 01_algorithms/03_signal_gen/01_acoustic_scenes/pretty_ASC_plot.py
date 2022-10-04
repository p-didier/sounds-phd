# Script to make an ASC look pretty and publishable (haha)

import sys, copy
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np

# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes/utilsASC')
from classes import AcousticScenario, PlottingOptions
sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import plot_side_room
from plotting.threedim import plot_room

rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 12})

BASEPATH = 'C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/02_data/01_acoustic_scenarios'

def main():

    # Give path
    pathToASC = f'{BASEPATH}/for_submissions/icassp2023/J4Mk[1_3_2_5]_Ns1_Nn2/AS15_RT150ms'

    fig = pretty_asc(pathToASC)
    plt.show()


def pretty_asc(path):

    # Load ASC
    asc = AcousticScenario()
    asc = asc.load(path)
    # Get options
    options = PlottingOptions(texts=False)

    # Determine appropriate node radius for ASC subplots
    nodeRadius = 0
    for k in range(asc.numNodes):
        allIndices = np.arange(asc.numSensors)
        sensorIndices = allIndices[asc.sensorToNodeTags == k + 1]
        curr = np.amax(asc.sensorCoords[sensorIndices, :] - np.mean(asc.sensorCoords[sensorIndices, :], axis=0))
        if curr > nodeRadius:
            nodeRadius = copy.copy(curr)

    fig = plt.figure()
    fig.set_size_inches(8.5, 3.5)
    
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('$x$', labelpad=-10)
    ax.set_ylabel('$y$', labelpad=-10)
    ax.set_zlabel('$z$', labelpad=-10)
    handles = []
    leglabs = []
    for ii in range(asc.numNodes):
        idx = asc.sensorToNodeTags == ii+1
        a = ax.scatter(asc.sensorCoords[idx, 0], asc.sensorCoords[idx, 1], asc.sensorCoords[idx, 2],
            color=f'C{ii}', s=30, edgecolors='k', alpha=1)
        handles.append(a)
        leglabs.append(f'Node {ii+1}')
    a = ax.scatter(asc.desiredSourceCoords[:, 0], asc.desiredSourceCoords[:, 1], asc.desiredSourceCoords[:, 2],
        color=['g' for _ in range(asc.desiredSourceCoords.shape[0])], edgecolors='k', alpha=1, marker='d', s=40)
    handles.append(a)
    leglabs.append('Desired source')
    a = ax.scatter(asc.noiseSourceCoords[:, 0], asc.noiseSourceCoords[:, 1], asc.noiseSourceCoords[:, 2],
        color=['r' for _ in range(asc.noiseSourceCoords.shape[0])], marker='P',edgecolors='k', alpha=1, s=40)
    handles.append(a)
    leglabs.append('Noise source')
    plot_room(ax, asc.roomDimensions)
    ax.view_init(elev=25., azim=40)
    ax.set_title('3D view')
    ax.legend(handles=handles,
        labels=leglabs,
        mode='expand',
        ncol=2,
        # bbox_to_anchor=(0, 0),
        loc="lower left"
    )  # <-- DONT BOTHER WITH THIS, DO LIKE A. BERTRAND IN HIS DANSE PAPER --> LEGEND IN THE CAPTION

    ax = fig.add_subplot(1, 3, 2)
    plot_side_room(ax, asc.roomDimensions[0:2], 
                asc.desiredSourceCoords[:, [0,1]], 
                asc.noiseSourceCoords[:, [0,1]], 
                asc.sensorCoords[:, [0,1]],
                asc.sensorToNodeTags,
                dotted=asc.absCoeff==1,
                options=options,
                showLegend=False,
                nodeRadius=nodeRadius)
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
    #
    ax = fig.add_subplot(1, 3, 3)
    plot_side_room(ax, asc.roomDimensions[1:], 
                asc.desiredSourceCoords[:, [1,2]], 
                asc.noiseSourceCoords[:, [1,2]],
                asc.sensorCoords[:, [1,2]],
                asc.sensorToNodeTags,
                dotted=asc.absCoeff==1,
                options=options,
                showLegend=False,
                nodeRadius=nodeRadius)
    ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')

    # Add legend


    plt.tight_layout()
        
    return fig


if __name__=='__main__':
    sys(exit(main()))
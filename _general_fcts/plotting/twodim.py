import matplotlib.pyplot as plt
import numpy as np

def plot_room2D(ax, rd):
    # Plots the edges of a rectangle in 2D on the axes <ax>.
    
    ax.plot([rd[0],0], [0,0], 'k')
    ax.plot([0,0], [0,rd[1]], 'k')
    ax.plot([rd[0],rd[0]], [0,rd[1]], 'k')
    ax.plot([0,rd[0]], [rd[1],rd[1]], 'k')

    return None


def plotSTFT(data):
    
    fig, ax = plt.subplots()
    ax.imshow(20*np.log10(np.abs(data)))
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.grid()
    # ax.set(xlabel='$t$ [s]')
    plt.show()

    return None
    

def plot_side_room(ax, rd2D, rs, rn, r, sensorToNodeTags, scatsize=20):
    """Plots a 2-D room side, showing the positions of
    sources and nodes inside of it.
    Parameters
    ----------
    ax : Axes handle
        Axes handle to plot on.
    rd2D : [2 x 1] list
        2-D room dimensions [m].
    rs : [Ns x 2] np.ndarray (real)
        Desired (speech) source(s) coordinates [m]. 
    rn : [Nn x 2] np.ndarray (real)
        Noise source(s) coordinates [m]. 
    r : [N x 2] np.ndarray (real)
        Sensor(s) coordinates [m]. 
    sensorToNodeTags : [N x 1] np.ndarray (int)
        Tags relating each sensor to a node number (>=1).
    scatsize : float
        Scatter plot marker size.
    """

    numNodes = len(np.unique(sensorToNodeTags))
    numSensors = len(sensorToNodeTags)
    
    plot_room2D(ax, rd2D)
    # Desired sources
    for idxSensor in range(rs.shape[0]):
        ax.scatter(rs[idxSensor,0], rs[idxSensor,1], s=scatsize,c='blue',marker='d')
        ax.text(rs[idxSensor,0], rs[idxSensor,1], "D%i" % (idxSensor+1))
    # Noise sources
    for idxSensor in range(rn.shape[0]):
        ax.scatter(rn[idxSensor,0], rn[idxSensor,1], s=scatsize,c='red',marker='P')
        ax.text(rn[idxSensor,0], rn[idxSensor,1], "N%i" % (idxSensor+1))
    # Nodes and sensors
    for idxNode in range(numNodes):
        allIndices = np.arange(numSensors)
        sensorIndices = allIndices[sensorToNodeTags == idxNode + 1]
        for idxSensor in sensorIndices:
            ax.scatter(r[idxSensor,0], r[idxSensor,1], s=scatsize,c='green',marker='o')
        # Draw circle around node
        radius = np.amax(r[sensorIndices, :] - np.mean(r[sensorIndices, :], axis=0))
        circ = plt.Circle((np.mean(r[sensorIndices,0]), np.mean(r[sensorIndices,1])),
                            radius * 2, color='k', fill=False)
        ax.add_patch(circ)
        # Add label
        ax.text(np.amax(r[sensorIndices,0]), np.amax(r[sensorIndices,1]), "Node %i" % (idxNode+1))
    ax.grid()
    ax.axis('equal')
    return None
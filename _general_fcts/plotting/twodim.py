import matplotlib.pyplot as plt
import numpy as np

def plot_room2D(ax, rd, dotted=False):
    """Plots the edges of a rectangle in 2D on the axes <ax>
    
    Parameters
    ----------
    ax : matplotlib Axes object
        Axes object onto which the rectangle should be plotted.
    rd : [3 x 1] (or [1 x 3], or [2 x 1], or [1 x 2]) np.ndarray or list of float
        Room dimensions [m].
    dotted : bool
        If true, use dotted lines. Else, use solid lines (default).
    """

    fmt = 'k'
    if dotted:
        fmt += '--'
    
    ax.plot([rd[0],0], [0,0], fmt)
    ax.plot([0,0], [0,rd[1]], fmt)
    ax.plot([rd[0],rd[0]], [0,rd[1]], fmt)
    ax.plot([0,rd[0]], [rd[1],rd[1]], fmt)

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
    

def plot_side_room(ax, rd2D, rs, rn, r, sensorToNodeTags,
                    options, scatsize=20, dotted=False, showLegend=True, nodeRadius=None):
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
    dotted : bool
        If true, use dotted lines. Else, use solid lines (default).
    """

    numNodes = len(np.unique(sensorToNodeTags))
    numSensors = len(sensorToNodeTags)
    
    plot_room2D(ax, rd2D, dotted)
    # Desired sources
    for idxSensor in range(rs.shape[0]):
        ax.scatter(rs[idxSensor,0], rs[idxSensor,1], s=scatsize,c='blue',marker='d')
        ax.text(rs[idxSensor,0], rs[idxSensor,1], "D%i" % (idxSensor+1))
    # Noise sources
    for idxSensor in range(rn.shape[0]):
        ax.scatter(rn[idxSensor,0], rn[idxSensor,1], s=scatsize,c='red',marker='P')
        ax.text(rn[idxSensor,0], rn[idxSensor,1], "N%i" % (idxSensor+1))
    # Nodes and sensors
    if options.nodesColors == 'multi':
        circHandles = []
        leg = []
    for idxNode in range(numNodes):
        allIndices = np.arange(numSensors)
        sensorIndices = allIndices[sensorToNodeTags == idxNode + 1]
        for idxSensor in sensorIndices:
            if options.nodesColors == 'multi':
                ax.scatter(r[idxSensor,0], r[idxSensor,1], s=scatsize,c=f'C{idxNode}',edgecolors='black',marker='o')
            else:
                ax.scatter(r[idxSensor,0], r[idxSensor,1], s=scatsize,c=options.nodesColors,edgecolors='black',marker='o')
        # Draw circle around node
        if nodeRadius is not None:
            radius = nodeRadius
        else:
            radius = np.amax(r[sensorIndices, :] - np.mean(r[sensorIndices, :], axis=0))
        if options.nodesColors == 'multi':
            circ = plt.Circle((np.mean(r[sensorIndices,0]), np.mean(r[sensorIndices,1])),
                                radius * 2, color=f'C{idxNode}', fill=False)
            circHandles.append(circ)
            leg.append(f'Node {idxNode + 1}')
        else:
            circ = plt.Circle((np.mean(r[sensorIndices,0]), np.mean(r[sensorIndices,1])),
                                radius * 2, color=options.nodesColors, fill=False)
            # Add label
            ax.text(np.amax(r[sensorIndices,0]), np.amax(r[sensorIndices,1]), "Node %i" % (idxNode+1))
        ax.add_patch(circ)
    ax.grid()
    ax.axis('equal')
    if showLegend and options.nodesColors == 'multi':
        nc = 1  # number of columbs in legend object
        if len(circHandles) >= 4:
            nc = 2
        ax.legend(circHandles, leg, loc='lower right', ncol=nc, mode='expand')
    return None
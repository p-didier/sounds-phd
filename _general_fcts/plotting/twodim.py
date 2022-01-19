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
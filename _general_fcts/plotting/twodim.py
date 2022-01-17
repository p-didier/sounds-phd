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
    
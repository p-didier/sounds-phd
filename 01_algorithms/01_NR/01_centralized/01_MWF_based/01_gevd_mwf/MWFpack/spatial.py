import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys,os
import scipy.signal as sig
from numba import njit
from scipy.signal.ltisys import dbode
#
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from plotting.twodim import plot_room2D, plotSTFT
from plotting.threedim import set_axes_equal, plot_room
from plotting.general import add_colorbar
from utilities.terminal import loop_progress
from mySTFT.calc_STFT import calcSTFT
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\03_signal_gen\\01_acoustic_scenes')))
from rimPypack.rimPy import rimPy
from ASgen import generate_array_pos


def gridify(rd, targetSources, gridres=0.5, plotgrid=False):
    # Gridifies a cuboidal space for MWF output visualization.

    # Init
    flag2D = False
    # Gridify room or room-slice
    x_ = np.linspace(0,rd[0],num=int(np.round(rd[0]/gridres)))
    y_ = np.linspace(0,rd[1],num=int(np.round(rd[1]/gridres)))
    if targetSources is not None:
        if len(targetSources.shape) == 2:
            z_ = np.unique(targetSources[:,-1])  # Set the z-coordinates of the target sources as slice heights
        else:
            z_ = np.array([targetSources[-1]])
            flag2D = True
        z_ = z_[:,np.newaxis]
        print('%.1f x %.1f x %.1f m^3 room gridified in %i slices every %.1f m,\nresulting in %i possible source locations.' %\
             (rd[0],rd[1],rd[2],len(z_),gridres,len(x_)*len(y_)*len(z_)))
    else:
        z_ = np.linspace(0,rd[2],num=int(np.round(rd[2]/gridres)))
        print('%.1f x %.1f x %.1f m^3 room gridified every %.1f m,\nresulting in %i possible source locations across the 3D space.' %\
             (rd[0],rd[1],rd[2],gridres,len(x_)*len(y_)*len(z_)))
    xx,yy,zz = np.meshgrid(x_, y_, z_, indexing='ij')

    if plotgrid:
        fig = plt.figure()
        if flag2D:
            ax = fig.add_subplot(111)
            ax.scatter(xx,yy,s=5.0,c='blue')
            ax.set(title='Room grid (2-D, $z$=%.2f m) - $\\delta$=%.2f m' % (z_[0],gridres))
            plot_room2D(ax,rd)
            ax.axis('equal')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xx,yy,zz,s=5.0,c='blue')
            ax.set(zlabel='$z$ [m]',title='Room grid (3-D) - $\\delta$=%.2f m' % gridres)
            set_axes_equal(ax)
            plot_room(ax,rd)
        ax.grid()
        ax.set(xlabel='$x$ [m]',ylabel='$y$ [m]')
        plt.show()

    return xx,yy,zz


def getmicsig(r0,r,x,Fs,rd,alpha,rir_dur):

    # Room's reflection coefficient
    refCoeff = -1*np.sqrt(1 - alpha)
    # Get RIRs
    h = rimPy(r, r0, rd, refCoeff, rir_dur, Fs)
    # Normalize individual RIRs
    h /= np.sum(h, axis=0)
    # Generate sensors signal 
    y = sig.fftconvolve(x, h, axes=0)

    return y, h

@njit
def mfb(h):
    # Computes Matched Filter Beamformer (MFB).
    # Following equation 2.72 in Randall Ali's PhD thesis.
    w = h @ np.linalg.pinv(h.conj().T @ h)
    return w


def applymwf(w,x):
    # Applies a multichannel Wiener filter to a multichannel signal
    # in the frequency-domain.
    y = np.zeros(x.shape[0], dtype=complex)
    for kp in range(w.shape[0]):
    
        # normfact = x[kp,:].conj().T @ x[kp,:]
        normfact = 1
        y[kp] = w[kp,:].conj().T @ (x[kp,:] / normfact)
    return y

# ----------- PLOTTING FUNCTIONS -----------
def plotspatialresp(xx,yy,zz,data,targetSource,micpos,dBscale=False,exportit=False,exportname='',freqs=[]):

    if dBscale:
        data = 20*np.log10(data)

    # PLOT
    fig = plt.figure(figsize=(5,4), constrained_layout=True)
    if len(data.shape) == 4:
        # Set the colorbar limits
        vmax = np.nanmean(np.nanmax(data, axis=-1))
        vmin = np.nanmean(np.nanmin(data, axis=-1))/3
        # There is a frequency-dependency
        for ii in range(data.shape[3]):
            print('Plotting GIF frame for frequency %.2f Hz (%i/%i)...' % (freqs[ii], ii+1, data.shape[3]))
            ax = fig.add_subplot(111)
            # Plot energy spatial contour map
            mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0,ii], vmin=vmin, vmax=vmax, levels=15)
            # Show target source as dot
            ax.scatter(targetSource[0],targetSource[1],s=70,marker='x',c='red',edgecolors='black',alpha=0.5)
            # Show microphones as dots
            ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
            # 
            # titstr = '$\\mathrm{E}_{l,k}\\left\{\\left| \\mathbf{w}(\\kappa)^H\\mathbf{h}(\\kappa) \\right|^2\\right\}$ - %i Hz' % (freqs[ii])
            titstr = '$\\bar{e}(\mathbf{r}_i) - %i Hz$' % (freqs[ii])
            if dBscale:
                titstr += ' [dB]'
            ax.set(title=titstr, xlabel='$x$ [m]', ylabel='$y$ [m]') # title
            ax.axis('equal')
            fmt = '%.1e'
            if dBscale:
                fmt = '%.1f'
            fig.colorbar(
                ScalarMappable(norm=mapp.norm, cmap=mapp.cmap),
                ticks=range(int(np.round(vmin)), int(np.round(vmax)), int((vmax - vmin)/10))
                )
            plt.savefig('%s_f%i.png' % (exportname, ii+1))#, bbox_inches='tight')

            fig.clear()
        print('All GIF frames saved as PNGs, names "%s".' % exportname)
    else:
        ax = fig.add_subplot(111)
        # Plot energy spatial contour map
        mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0], levels=15)
        # Show target source as dot
        ax.scatter(targetSource[0],targetSource[1],s=70,marker='x',c='red',edgecolors='black',alpha=0.5)
        # Show microphones as dots
        ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
        # 
        # titstr = '$\\mathrm{E}_{\\kappa,l,k}\\left\{\\left| \\mathbf{w}^H\\mathbf{h} \\right|^2\\right\}$'
        titstr = '$\\bar{e}(\mathbf{r}_i)$'
        if dBscale:
            titstr += ' [dB]'
        ax.set(title=titstr, xlabel='$x$ [m]', ylabel='$y$ [m]') # title
        ax.axis('equal')
        fmt = '%.1e'
        if dBscale:
            fmt = '%.1f'
        plt.colorbar(mapp, ax=ax, format=fmt)
    
        # Export or show
        if exportit:
            plt.savefig('%s.png' % exportname, bbox_inches='tight')
            plt.savefig('%s.pdf' % exportname, bbox_inches='tight')
        else:
            plt.show()
        print('Average energy plot exported, "%s".' % exportname)

    stop = 1

    return None


def main(refidx):

    # User inputs
    rd = np.array([6,6,5])                                  # room dimensions
    revTime = 0.2                                           # reverberation time in room (0 if anechoic)
    revTime = 0.0                                           # reverberation time in room (0 if anechoic)
    z_slice = 3                                             # height of 2D-slice (where all nodes and sources are)
    targetSource = np.random.uniform(0,1,size=(3,)) * rd    # target source coordinates
    targetSource[-1] = z_slice              
    Nr = 5                                                 # number of receivers 
    # arraytype = 'random'    # If "random" - Generate <Nr> random receiver positions across available volume
    arraytype = 'fixedrandom'    # If "fixedrandom" - Use pre-generated 5 random receiver positions
    # arraytype = 'compact'   # If "compact" - Generate single linear array of <Nr> microphones, randomly placed in room
    arraytype = 'fixedcompact'   # If "fixedcompact" - Use pre-generated single linear array of 5 microphones
    Fs = 16e3                           # sampling frequency
    gres = 0.1                          # spatial grid resolution
    # gres = 2                          # spatial grid resolution
    #
    makeGIF = 1     # If True, export a series of PNGs (per freq. bin) for GIF-making
    exportit = 1    # If False, do not export any figure

    # Generate microphone positions
    if arraytype == 'compact':
        arraycentroid = np.random.uniform(0,1,size=(3,)) * rd
        arraycentroid[-1] = z_slice              
        micpos = generate_array_pos(arraycentroid, Nr, 'linear', 0.05, force2D=True)
    elif arraytype == 'random':
        micpos = np.random.uniform(0,1,size=(Nr,3)) * rd        # receiver coordinates
        micpos[:,-1] = z_slice  
    elif arraytype == 'fixedrandom':
        micpos = np.array([[2.90100361, 1.31403583, 3.        ],
                            [1.118861  , 0.90872411, 3.        ],
                            [3.60949543, 3.94110555, 3.        ],
                            [3.09656351, 3.98553171, 3.        ],
                            [3.76354577, 0.24801863, 3.        ]])
        targetSource = np.array([1.59439374, 1.30800814, 3.        ])
    elif arraytype == 'fixedcompact':
        micpos = np.array([[4.62104008, 2.56325556, 3],
                            [4.64143235, 2.61825619, 3],
                            [4.66182462, 2.67325683, 3.        ],
                            [4.68221689, 2.72825747, 3.],
                            [4.70260915, 2.78325811, 3.]])
        targetSource = np.array([4.41080322, 5.33609293, 3.        ])

    # RIR duration
    rir_dur = 2**10/Fs                      # RIR duration
        
    # Room's reflection coefficient
    alpha = 0.161*np.prod(rd)/(revTime * 2 *(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2]))  # room absorption coefficient
    if alpha > 1:
        alpha = 1
    refCoeff = -1*np.sqrt(1 - alpha)

    # Get beamforming filter for target source
    RIRs = rimPy(micpos, targetSource, rd, refCoeff, rir_dur, Fs)      # Get RIRs to target source
    RTFs = np.fft.fft(RIRs, axis=0)
    # Only consider positive frequencies
    RTFs = RTFs[:int(np.round(RTFs.shape[0] / 2)), :]
    # Compute Matched Filter Beamformer h/(h^H*h)
    w_target = mfb(RTFs)

    # Gridify room
    xx,yy,zz = gridify(rd, targetSource, gridres=gres, plotgrid=False)

    # Plot acoustic scenario + grid
    # plot_AS(rd, micpos, targetSource, ASref=arraytype, xx=xx,yy=yy,zz=zz, exportit=1)

    en_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],int(rir_dur*Fs / 2)))     # per-freq. energy
    en    = np.zeros_like(xx)                        # freq.-averaged energy
    for ix in range(xx.shape[0]):
        for iy in range(xx.shape[1]):
            for iz in range(xx.shape[2]):
                r0 = [xx[ix,iy,iz], yy[ix,iy,iz], zz[ix,iy,iz]]     # Source location (grid point)
                # Monitor loop
                progress_percent = loop_progress([ix,iy,iz], xx.shape)
                if progress_percent % 10 == 0:
                    print('Computing filter output energy for source at (%.2f,%.2f,%.2f) in room [%.2f %%]...' %\
                        (r0[0],r0[1],r0[2],progress_percent))
                
                RIRs = rimPy(micpos, r0, rd, refCoeff, rir_dur, Fs) # Get RIRs
                # RIRs /= np.amax(RIRs, axis=0)                       # ``Max`` scaling
                # RIRs /= np.sum(np.abs(RIRs), axis=0)                # 1-norm scaling
                RIRs /= np.sqrt(np.sum(np.abs(RIRs)**2, axis=0))      # Euclidean norm scaling

                            
                # fig = plt.figure(figsize=(4,4), constrained_layout=True)
                # ax = fig.add_subplot(111)
                # ax.plot(np.arange(len(RIRs))/Fs, RIRs)
                # ax.grid()
                # ax.set(xlabel='$t$ [s]')
                # plt.legend(['Sensor %i' % (s+1) for s in range(RIRs.shape[1])])
                # fname = '%s\\RIRs.pdf' % ('01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\RIRsnorms')
                # plt.savefig(fname)
                # plt.show()

                RTFs = np.fft.fft(RIRs, axis=0)                     # Get RTFs
                freqs = Fs * np.fft.fftfreq(len(RTFs), d=1.0)       # ...and the corresponding frequency vector
                # Only consider positive frequencies
                RTFs = RTFs[:int(np.round(RTFs.shape[0] / 2)), :]
                freqs = freqs[:int(np.round(len(freqs) / 2))]

                y = applymwf(w_target, RTFs)                        # Apply target filter to RIRs

                # fig, ax = plt.subplots(2,1)
                # ax[0].plot(freqs, 20*np.log10(np.abs(RTFs)))
                # ax[0].grid()
                # ax[1].plot(freqs, 20*np.log10(np.abs(y)))
                # ax[1].grid()
                # plt.show()

                en_pf[ix,iy,iz,:] = np.abs(y)**2                    # Get output energy per freq.   
                en[ix,iy,iz] = np.mean(en_pf[ix,iy,iz,:], axis=0)   # Get avg. normalized output energy     


    # Plotting
    foldername = 'C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\MFB'
    fname = '%s\\spatialvisu_MFB_%i' % (foldername,refidx)
    plotspatialresp(xx,yy,zz,en,targetSource,micpos,dBscale=1,exportit=exportit,exportname=fname)
    if makeGIF:
        plotspatialresp(xx,yy,zz,en_pf,targetSource,micpos,dBscale=1,exportit=exportit,exportname=fname,freqs=freqs)

    return 0


def plot_AS(rd,r,rs,ASref='',xx=[],yy=[],zz=[],exportit=0):
    fig = plt.figure(figsize=(4,4), constrained_layout=True)
    ax = fig.add_subplot(111)
    if xx is not []:
        # Plot grid
        phgrid = ax.scatter(xx,yy,zz,c='lightblue',alpha=0.5)
    # Show target source as dot
    ph1 = ax.scatter(rs[0],rs[1],c='red',edgecolors='black')
    # Show microphones as dots
    ph2 = ax.scatter(r[:,0],r[:,1],c='cyan',edgecolors='black')
    # 
    ax.grid()
    plt.legend([ph1, ph2, phgrid],['Target source', 'Sensors', 'Source grid $\{\mathbf{r}_i\}_{i=1}^N$'],framealpha=1,loc='upper left')
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]') # title
    ax.axis('equal')
    ax.set_xlim(0, rd[0])
    ax.set_ylim(0, rd[1])
    if exportit:
        fname = '%s\\AS_%s.png' % ('01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\ASs', ASref)
        plt.savefig(fname)
        fname = '%s\\AS_%s.pdf' % ('01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\ASs', ASref)
        plt.savefig(fname)
    plt.show()

    return None


if __name__ == '__main__':
    Nfigs = 1
    for refidx in range(Nfigs):
        print('\n\nRunning for figure %i/%i...\n' % (refidx+1,Nfigs))
        main(refidx + 1)
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
    w = np.zeros_like(h)
    for ii in range(h.shape[0]):
        hcurr = h[ii,:,:]
        w[ii,:,:] = hcurr @ np.linalg.pinv(hcurr.conj().T @ hcurr)
    return w


def applymwf(w,x):
    # Applies a multichannel Wiener filter to a multichannel signal
    # in the STFT-domain.

    y = np.zeros_like(x)
    for kp in range(w.shape[0]):
        for l in range(w.shape[1]):
            y[kp,l,:] = w[kp,l,:].conj().T @ x[kp,l,:]

    return y


# @njit
def energy(x):
    # Computes total "energy" of (multi-channel) signal(s) in STFT-domain. 

    if len(x.shape) == 3:   # multi-channel case
        e = np.zeros(x.shape[2])
        e_pf = np.zeros((x.shape[2],x.shape[0]))
        for ii in range(x.shape[2]):
            # Average over frequency bins
            eav = np.mean(np.abs(x[:,:,ii])**2, axis=0)
            e[ii] = np.amax(eav)   # get max value across time frames
            e_pf[ii,:] = np.amax(np.abs(x[:,:,ii])**2, axis=1)  # per frequency
    else:
        e = np.amax(np.mean(np.abs(x)**2, axis=0))
        e_pf = np.amax(np.abs(x)**2, axis=1)      # per frequency
    return e, e_pf


# ----------- PLOTTING FUNCTIONS -----------
def plotspatialresp(xx,yy,zz,data,targetSource,micpos,dBscale=False,exportit=False,exportname='',freqs=[]):

    if dBscale:
        data = 20*np.log10(data)

    # PLOT
    fig = plt.figure(figsize=(5,4), constrained_layout=True)
    if len(data.shape) == 5:
        # There is a frequency-dependency
        for ii in range(data.shape[-1]):
            ax = fig.add_subplot(111)
            # Plot energy spatial contour map
            mapp = ax.contourf(xx[:,:,0], yy[:,:,0], np.mean(data[:,:,0,:,ii], axis=2), vmin=-100., vmax=0.)
            # Show target source as dot
            ax.scatter(targetSource[0],targetSource[1],c='red',edgecolors='black')
            # Show microphones as dots
            ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
            # 
            titstr = '$\\mathrm{E}_{l,k}\\left\{\\left| \\mathbf{w}_\\mathrm{MFB})(\\kappa)^H\\mathbf{h}(\\kappa) \\right|^2\\right\}$ - %i Hz' % (freqs[ii])
            if dBscale:
                titstr += ' [dB]'
            ax.set(title=titstr, xlabel='$x$ [m]', ylabel='$y$ [m]') # title
            ax.axis('equal')
            fmt = '%.1e'
            if dBscale:
                fmt = '%.1f'
            fig.colorbar(
                ScalarMappable(norm=mapp.norm, cmap=mapp.cmap),
                ticks=range(-100, 0, 10)
                )
            plt.savefig('%s_f%i.png' % (exportname, ii+1))#, bbox_inches='tight')

            fig.clear()
    else:
        ax = fig.add_subplot(111)
        # Plot energy spatial contour map
        mapp = ax.contourf(xx[:,:,0], yy[:,:,0], np.mean(data[:,:,0,:], axis=2))
        # Show target source as dot
        ax.scatter(targetSource[0],targetSource[1],c='red',edgecolors='black')
        # Show microphones as dots
        ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
        # 
        titstr = '$\\mathrm{E}_{\\kappa,l,k}\\left\{\\left| \\mathbf{w}_\\mathrm{MFB}^H\\mathbf{h} \\right|^2\\right\}$'
        if dBscale:
            titstr += ' [dB]'
        ax.set(title=titstr, xlabel='$x$ [m]', ylabel='$y$ [m]') # title
        ax.axis('equal')
        fmt = '%.1e'
        if dBscale:
            fmt = '%.1f'
        plt.colorbar(mapp, ax=ax, format=fmt)
    #
    # Export or show
    if exportit:
        plt.savefig('%s.png' % exportname, bbox_inches='tight')
        plt.savefig('%s.pdf' % exportname, bbox_inches='tight')
    else:
        plt.show()

    stop = 1

    return None


def main(refidx):

    # User inputs
    rd = np.array([6,6,5])                                  # room dimensions
    # alpha = 0.99                                            # room absorption coefficient
    alpha = 1                                               # room absorption coefficient
    z_slice = 3                                             # height of 2D-slice (where all nodes and sources are)
    targetSource = np.random.uniform(0,1,size=(3,)) * rd    # target source coordinates
    targetSource[-1] = z_slice              
    Nr = 5                                                 # number of receivers 
    # arraytype = 'random'    # If "random" - Generate <Nr> random receiver positions across available volume
    arraytype = 'fixedrandom'    # If "fixedrandom" - Use pre-generated 5 random receiver positions
    # arraytype = 'compact'   # If "compact" - Generate single linear array of <Nr> microphones, randomly placed in room
    # arraytype = 'fixedcompact'   # If "fixedcompact" - Use pre-generated single linear array of 5 microphones
    Fs = 16e3                           # sampling frequency
    gres = 0.1                          # spatial grid resolution
    # gres = 2                          # spatial grid resolution

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
    # elif arraytype == 'fixedcompact':


    # RIR duration
    # c0 = 343        # speed of sound
    # maxt = 1/c0 * np.amax(np.linalg.norm(micpos - targetSource, axis=1))   # sound travel time corresponding to greatest source-receiver distance
    # rir_dur = 2 * maxt                      # RIR duration
    rir_dur = 2**10/Fs                      # RIR duration
    # print('Greatest mic-source distance: %.2f m' % (maxt*c0))

    # STFT
    # L_fft = np.amin([2**9, 2**(np.floor(np.log2(rir_dur * Fs)) - 1)])    # Time frame length [samples]
    L_fft = 2**9    # Time frame length [samples]
    R_fft = L_fft/2                         # Inter-frame overlap length [samples]
    win = np.sqrt(np.hanning(L_fft))        # STFT time window
    print('Corresponding RIR duration: %.2f s (%i samples)' % (rir_dur, rir_dur*Fs))
    print('Chosen FFT frame size: %i samples (%i frames/RIR)' % (R_fft, np.floor(rir_dur*Fs/R_fft)))
        
    # Room's reflection coefficient
    refCoeff = -1*np.sqrt(1 - alpha)

    # Get beamforming filter for target source
    RIRs = rimPy(micpos, targetSource, rd, refCoeff, rir_dur, Fs)      # Get RIRs to target source
    RTFs = np.fft.fft(RIRs, axis=0)
    # h_STFT = calcSTFT(h, Fs, win, L_fft, R_fft, 'onesided')[0]      # Go to STFT domain
    w_target = mfb(h_STFT)                                                 # Get MFB in STFT domain for target source

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.arange(len(RIRs))/Fs, RIRs)
    # ax[0].grid()
    # ax[1].plot(20*np.log10(np.abs(RTFs[:int(len(RTFs)/2)])))
    # ax[1].grid()
    # plt.show()

    # Gridify room
    xx,yy,zz = gridify(rd, targetSource, gridres=gres, plotgrid=False)

    en = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],micpos.shape[0]))            # total TF-averaged energy
    e_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],micpos.shape[0],int(R_fft + 1)))    # t-averaged energy per frequency line
    for ix in range(xx.shape[0]):
        for iy in range(xx.shape[1]):
            for iz in range(xx.shape[2]):
                r0 = [xx[ix,iy,iz], yy[ix,iy,iz], zz[ix,iy,iz]]     # Source location (grid point)
                # Monitor loop
                progress_percent = loop_progress([ix,iy,iz], xx.shape)
                if progress_percent % 10 == 0:
                    print('Computing filter output energy for source at (%.2f,%.2f,%.2f) in room [%.2f %%]...' %\
                        (r0[0],r0[1],r0[2],progress_percent))
                
                h = rimPy(micpos, r0, rd, refCoeff, rir_dur, Fs)            # Get RIRs
                h /= np.sum(np.abs(h), axis=0)                                      # Normalize individual RIRs
                h_STFT, f = calcSTFT(h, Fs, win, L_fft, R_fft, 'onesided')  # Go to STFT domain
                
                y = applymwf(w_target, h_STFT)          # Apply target filter to RIRs
                
                a, b = energy(y)         # Get output energy
                en[ix,iy,iz,:], e_pf[ix,iy,iz,:,:] = a, b

    # Plotting
    foldername = 'C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\03_for_20211021meeting\\02_spatial_visu\\MFB'
    fname = '%s\\spatialvisu_MFB_%i' % (foldername,refidx)
    plotspatialresp(xx,yy,zz,en,targetSource,micpos,dBscale=1,exportit=1,exportname=fname)
    plotspatialresp(xx,yy,zz,e_pf,targetSource,micpos,dBscale=1,exportit=0,exportname=fname,freqs=f)

    return 0


if __name__ == '__main__':
    Nfigs = 10
    for refidx in range(Nfigs):
        print('\n\nRunning for figure %i/%i...\n' % (refidx+1,Nfigs))
        main(refidx + 1)
from numba.core.errors import ForbiddenConstruct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import sys,os
from numba import njit
from numpy.core.numeric import zeros_like
from numpy.ma import multiply
import scipy
import scipy.fft
#
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from plotting.threedim import plot_room
from utilities.terminal import loop_progress
from general.frequency import get_closest
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\03_signal_gen\\01_acoustic_scenes')))
from rimPypack.rimPy import rimPy
from ASgen import generate_array_pos


def gridify(rd, z_, gridres=0.5, plotgrid=False):
    # Gridifies a 2D space for MWF output visualization.

    # Do not compute more slices than necessary
    z_ = np.unique(z_)

    # Gridify room or room-slice
    x_ = np.linspace(0,rd[0],num=int(np.round(rd[0]/gridres)))
    y_ = np.linspace(0,rd[1],num=int(np.round(rd[1]/gridres)))
    print('%.1f x %.1f m^2 space gridified in %i slices every %.1f m,\nresulting in %i possible source locations.' %\
            (rd[0],rd[1],len(z_),gridres,len(x_)*len(y_)*len(z_)))
    xx,yy,zz = np.meshgrid(x_, y_, z_, indexing='ij')

    if plotgrid:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xx,yy,zz,s=5.0,c='blue')
        ax.set(title='Room grid - $\\delta$=%.2f m' % (gridres))
        plot_room(ax,rd)
        ax.axis('equal')
        ax.grid()
        ax.set(xlabel='$x$ [m]',ylabel='$y$ [m]',zlabel='$z$ [m]')
        plt.show()

    return xx,yy,zz


@njit
def mfb(h):
    # Computes Matched Filter Beamformer (MFB).
    # Following equation 2.72 in Randall Ali's PhD thesis.
    w = zeros_like(h)
    for ii in range(h.shape[0]):
        # w = h @ np.linalg.pinv(h.conj().T @ h)
        w[ii,:] = h[ii,:] / (h[ii,:].conj().T @ h[ii,:])
    return w


def mvdr(h, hn):
    # Computes MVDR filter.
    w = np.zeros_like(h)
    for ii in range(h.shape[0]):
        Rnn = np.outer(hn[ii,:], hn[ii,:].conj())
        # Rnn = np.identity(h.shape[1])    # <-- uncomment to get back to MFB
        num = np.linalg.inv(Rnn) @ h[ii,:].T
        den = h[ii,:].conj() @ np.linalg.inv(Rnn) @ h[ii,:].T
        w[ii,:] = num / den
    return w

# @njit
def applymwf(w,x):
    # Applies a multichannel Wiener filter to a multichannel signal
    # in the frequency-domain.

    # Check input dimensions
    if len(w.shape) == 2:
        y = np.zeros(x.shape[0], dtype=np.complex128)
        for kp in range(w.shape[0]):
            y[kp] = applymwf_jitted(w[kp,:], x[kp,:])
    elif len(w.shape) == 3:
        y = np.zeros_like(x, dtype=np.complex128)
        for kp in range(w.shape[0]):
            y[kp,:] = applymwf_jitted(w[kp,:,:], x[kp,:])
        stop = 1
    return y

@njit
def applymwf_jitted(a,b):
    return a.conj().T @ b


def getenergy(xx,yy,zz,rir_dur,Fs,micpos,rd,refCoeff,w_target,bins_of_interest=None):
    # Calculate energy over grid

    # Initialize arrays
    if len(w_target.shape) == 2:
        en = np.zeros_like(xx)                        # freq.-averaged energy
        if bins_of_interest is not None:
            # Check that optional input arguments are correctly dimensioned
            if len(bins_of_interest) != w_target.shap3e[0]:
                raise ValueError('The optional arg <bins_of_interest> must refer to as many bins as the filter provided.')
            en_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],len(bins_of_interest)))     # per-freq. energy
        else:
            en_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],int(rir_dur*Fs / 2)))     # per-freq. energy
    elif len(w_target.shape) == 3:
        en = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],micpos.shape[0]))                        # freq.-averaged energy
        if bins_of_interest is not None:
            # Check that optional input arguments are correctly dimensioned
            if len(bins_of_interest) != w_target.shape[0]:
                raise ValueError('The optional arg <bins_of_interest> must refer to as many bins as the filter provided.')
            en_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],len(bins_of_interest),micpos.shape[0]))     # per-freq. energy
        else:
            en_pf = np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],int(rir_dur*Fs / 2),micpos.shape[0]))     # per-freq. energy

    for ix in range(xx.shape[0]):
        for iy in range(xx.shape[1]):
            for iz in range(xx.shape[2]):
                r0 = [xx[ix,iy,iz], yy[ix,iy,iz], zz[ix,iy,iz]]     # Source location (grid point)
                # Monitor loop
                progress_percent = loop_progress([ix,iy,iz], xx.shape)
                print('Computing filter output energy for source at (%.2f,%.2f,%.2f) in room [%.2f %%]...' %\
                    (r0[0],r0[1],r0[2],progress_percent))
                
                RIRs = rimPy(micpos, r0, rd, refCoeff, rir_dur, Fs) # Get RIRs
                RIRs /= np.sqrt(np.sum(np.abs(RIRs)**2, axis=0))    # Euclidean norm scaling
                RTFs = np.fft.fft(RIRs, axis=0)                     # Get RTFs
                freqs = np.fft.fftfreq(len(RTFs), d=1/Fs)       # ...and the corresponding frequency vector
                # Only consider positive frequencies
                RTFs = RTFs[freqs >= 0, :]
                freqs = freqs[freqs >= 0]
                # Interpret inputs
                if bins_of_interest is not None:
                    indices_freq_bins = get_closest(freqs,bins_of_interest)
                    # print('Only taking into account freq.-bins from %i to %i Hz' % (freqs[indices_freq_bins[0]], np.abs(freqs[indices_freq_bins[-1]])))
                else:
                    indices_freq_bins = np.arange(len(freqs))  # take all bins into account
                    # print('Taking into account all frequency bins (up to %i Hz)' % freqs[-1])
                #
                y = applymwf(w_target, RTFs[indices_freq_bins, :])                        # Apply target filter to RIRs
                if len(w_target.shape) == 2:
                    en_pf[ix,iy,iz,:] = np.abs(y)**2                    # Get output energy per freq.   
                    en[ix,iy,iz] = np.mean(en_pf[ix,iy,iz,:], axis=0)   # Get avg. normalized output energy 
                elif len(w_target.shape) == 3:
                    en_pf[ix,iy,iz,:,:] = np.abs(y)**2                    # Get output energy per freq.   
                    en[ix,iy,iz,:] = np.mean(en_pf[ix,iy,iz,:,:], axis=0)   # Get avg. normalized output energy 
                    

    return en, en_pf, freqs[indices_freq_bins]

# ----------- PLOTTING FUNCTIONS -----------
def plotspatialresp(xx,yy,data,targetSource,micpos,dBscale=False,exportit=False,exportname='',\
    freqs=[],noiseSource=None,multichannel=False,noise_spatially_white=False,\
        stoi_imp=None, fwSNRseg_imp=None):

    # If asked, bring to dB-scale
    if dBscale:
        data = 20*np.log10(np.abs(data))

    # Multi-channel case
    if multichannel:
        maxlen = 5
        nChannels = data.shape[-1]  
        n_rows_plots = int(np.floor(nChannels / 3) + 1)
        n_cols_plots = int(np.ceil(nChannels / n_rows_plots))
    else:
        maxlen = 4

    # Labels and titles font size
    fts = 8

    # PLOT
    if multichannel:
        fig = plt.figure(figsize=(np.amax(xx)*n_cols_plots * 0.5, np.amax(yy)*n_rows_plots * 0.8 * 0.5))
    else:
        fig = plt.figure(figsize=(np.amax(xx), np.amax(yy)), constrained_layout=True)
    if len(data.shape) == maxlen:
        if multichannel:
            raise ValueError('NOT YET IMPLEMENTED')
        # Set the colorbar limits
        vmax = np.nanmean(np.nanmax(data, axis=3))
        vmin = np.nanmean(np.nanmin(data, axis=3))/3
        # There is a frequency-dependency
        for ii in range(data.shape[3]):
            print('Plotting GIF frame for frequency %.2f Hz (%i/%i)...' % (freqs[ii], ii+1, data.shape[3]))
            ax = fig.add_subplot(111)
            # Plot energy spatial contour map
            mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0,ii], vmin=vmin, vmax=vmax, levels=15)
            # Show microphones as dots
            ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
            # Show target source as dot
            if len(targetSource.shape) == 1:
                ax.scatter(targetSource[0],targetSource[1],s=70,marker='x',c='blue',edgecolors='black',alpha=0.5)
            else:
                for ii in range(targetSource.shape[0]):
                    ax.scatter(targetSource[ii,0],targetSource[ii,1],s=70,marker='x',c='blue',edgecolors='black',alpha=0.5)
            # Show noise sources, if given
            if noiseSource is not None:
                if len(noiseSource.shape) == 1:
                    ax.scatter(noiseSource[0],noiseSource[1],s=70,marker='x',c='red',edgecolors='black',alpha=0.5)
                else:
                    for ii in range(noiseSource.shape[0]):
                        ax.scatter(noiseSource[ii,0],noiseSource[ii,1],s=70,marker='x',c='red',edgecolors='black',alpha=0.5)
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
                ticks=range(int(np.round(vmin)), int(np.round(vmax)), int((vmax - vmin)/5))
                )
            plt.savefig('%s_f%i.png' % (exportname, ii+1))#, bbox_inches='tight')

            fig.clear()
        print('All GIF frames saved as PNGs, names "%s".' % exportname)
    else:
        if multichannel:
            ylim_min = np.nanmin(data[:,:,0,:])
            ylim_max = np.nanmax(data[:,:,0,:])
            for ii in range(nChannels):
                ax = fig.add_subplot(n_rows_plots,n_cols_plots,ii+1)

                # Plot energy spatial contour map
                # mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0,ii], levels=15, vmin=ylim_min, vmax=ylim_max)
                mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0,ii], levels=15)
                # Show microphones as dots
                for jj in range(nChannels):
                    if jj != ii:
                        ph_nodes = ax.scatter(micpos[jj,0],micpos[jj,1],c='cyan',edgecolors='black')
                    else:
                        ph_nodes = ax.scatter(micpos[jj,0],micpos[jj,1],c='blue',edgecolors='yellow')
                # Show target source as dot
                if len(targetSource.shape) == 1:
                    ph_speech = ax.scatter(targetSource[0],targetSource[1],s=70,marker='x',c='blue',alpha=0.5)
                else:
                    for jj in range(targetSource.shape[0]):
                        ph_speech = ax.scatter(targetSource[jj,0],targetSource[jj,1],s=70,marker='x',c='blue',alpha=0.5)
                        ax.text(targetSource[jj,0]+0.01,targetSource[jj,1]+0.01,'S%i' % (jj+1)) # add text to mark source position
                # Show noise sources, if given
                if noiseSource is not None and (noise_spatially_white == False or noise_spatially_white == 'combined'):
                    if len(noiseSource.shape) == 1:
                        ph_noise = ax.scatter(noiseSource[0],noiseSource[1],s=70,marker='x',c='red',alpha=0.5)
                    else:
                        for jj in range(noiseSource.shape[0]):
                            ph_noise = ax.scatter(noiseSource[jj,0],noiseSource[jj,1],s=70,marker='x',c='red',alpha=0.5)
                
                # Axes formatting (labels, limits, title, font sizes)
                titstr = '$\\bar{e}(\mathbf{r}_i)$ - Sensor %i' % (ii+1)
                if dBscale:
                    titstr += ' [dB]'
                ax.set_title(titstr, fontsize=fts)
                ax.set_ylabel('$y$ [m]', fontsize=fts)
                if ii+1 > (n_rows_plots - 1) * n_cols_plots:
                    ax.set_xlabel('$x$ [m]', fontsize=fts)
                ax.axis('equal')
                ax.set_xlim((0, np.amax(xx)))
                ax.set_ylim((0, np.amax(yy)))
                ax.tick_params(axis='x', labelsize=fts)
                ax.tick_params(axis='y', labelsize=fts)
                #
                # Colorbar
                fmt = '%.1e'
                if dBscale:
                    fmt = '%i'
                fig.colorbar(
                    ScalarMappable(norm=mapp.norm, cmap=mapp.cmap),
                    ticks=np.linspace(np.round(ylim_min), np.round(ylim_max), 10),
                    format=fmt
                    )

            if freqs is not []:
                if len(freqs) > 1:
                    suptit = 'Range: $f \in [%i,...,%i]$ Hz' % (np.amin(np.abs(freqs)), np.amax(np.abs(freqs)))
                else:
                    suptit = 'Single freq. bin -- $f = %i$ Hz' % (freqs[0])

                if fwSNRseg_imp is not None:
                    suptit += ';  $\Delta$fwSNRseg = %.1f' % (fwSNRseg_imp)
                if stoi_imp is not None:
                    for idx_ns in range(stoi_imp.shape[-1]):
                        suptit += ';  $\Delta$STOI(S%i) = %.3f' % (idx_ns+1, stoi_imp[idx_ns])
                plt.suptitle(suptit)
        else:
            ax = fig.add_subplot(111)
            # Plot energy spatial contour map
            mapp = ax.contourf(xx[:,:,0], yy[:,:,0], data[:,:,0], levels=15)
            # Show microphones as dots
            ax.scatter(micpos[:,0],micpos[:,1],c='cyan',edgecolors='black')
            # Show target source as dot
            if len(targetSource.shape) == 1:
                ax.scatter(targetSource[0],targetSource[1],s=70,marker='x',c='blue',alpha=0.5)
            else:
                for ii in range(targetSource.shape[0]):
                    ax.scatter(targetSource[ii,0],targetSource[ii,1],s=70,marker='x',c='blue',alpha=0.5)
            # Show noise sources, if given
            if noiseSource is not None:
                if len(noiseSource.shape) == 1:
                    ax.scatter(noiseSource[0],noiseSource[1],s=70,marker='x',c='red',alpha=0.5)
                else:
                    for ii in range(noiseSource.shape[0]):
                        ax.scatter(noiseSource[ii,0],noiseSource[ii,1],s=70,marker='x',c='red',alpha=0.5)
            # 
            titstr = '$\\bar{e}(\mathbf{r}_i)$'
            if dBscale:
                titstr += ' [dB]'
            ax.set(title=titstr, xlabel='$x$ [m]', ylabel='$y$ [m]') # title
            ax.axis('equal')
            ax.set_xlim((0, np.amax(xx)))
            ax.set_ylim((0, np.amax(yy)))
            fmt = '%.1e'
            if dBscale:
                fmt = '%i'
            plt.colorbar(mapp, ax=ax, format=fmt)
    
        # Export or show
        if exportit:
            # Check whether folder already exists
            if not os.path.isdir(exportname[:exportname.rfind('\\')]):
                os.mkdir(exportname[:exportname.rfind('\\')])
            plt.savefig('%s.png' % exportname, bbox_inches='tight')
            plt.savefig('%s.pdf' % exportname, bbox_inches='tight')
            print('Average energy plot exported, "%s".' % exportname)
        else:
            plt.show()

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
    noiseSource = np.random.uniform(0,1,size=(3,)) * rd    # noise source coordinates
    noiseSource[-1] = z_slice              
    # noiseSource = None      # Uncomment to assume spatially white noise
    Nr = 5                                                 # number of receivers 
    # arraytype = 'random'    # If "random" - Generate <Nr> random receiver positions across available volume
    arraytype = 'fixedrandom'    # If "fixedrandom" - Use pre-generated 5 random receiver positions
    # arraytype = 'compact'   # If "compact" - Generate single linear array of <Nr> microphones, randomly placed in room
    # arraytype = 'fixedcompact'   # If "fixedcompact" - Use pre-generated single linear array of 5 microphones
    Fs = 16e3                           # sampling frequency
    gres = 0.1                          # spatial grid resolution
    # gres = 2                          # spatial grid resolution
    #
    makeGIF = 0     # If True, export a series of PNGs (per freq. bin) for GIF-making
    exportit = 0    # If False, do not export any figure

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
        # targetSource = np.array([1.108861  , 0.91872411, 3.        ])
        if noiseSource is not None:
            noiseSource = np.array([1, 2, 3.        ])
    elif arraytype == 'fixedcompact':
        micpos = np.array([[4.62104008, 2.56325556, 3],
                            [4.64143235, 2.61825619, 3],
                            [4.66182462, 2.67325683, 3.        ],
                            [4.68221689, 2.72825747, 3.],
                            [4.70260915, 2.78325811, 3.]])
        targetSource = np.array([4.41080322, 5.33609293, 3.        ])
        if noiseSource is not None:
            noiseSource = np.array([1, 2, 3.        ])

    # RIR duration
    rir_dur = 2**10/Fs                      # RIR duration
        
    # Room's reflection coefficient
    alpha = 0.161*np.prod(rd)/(revTime * 2 *(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2]))  # room absorption coefficient
    if alpha > 1:
        alpha = 1
    refCoeff = -1*np.sqrt(1 - alpha)

    # Get beamforming filter for target source
    RIRs = rimPy(micpos, targetSource, rd, refCoeff, rir_dur, Fs)      # Get RIRs to target source
    # RIRs /= np.sqrt(np.sum(np.abs(RIRs)**2, axis=0))    # Euclidean norm scaling
    RTFs = np.fft.fft(RIRs, axis=0)
    RTFs = RTFs[:int(np.round(RTFs.shape[0] / 2)), :]   # Only consider positive frequencies
    #
    if noiseSource is not None:
        RIRs_n = rimPy(micpos, noiseSource, rd, refCoeff, rir_dur, Fs)      # Get RIRs to noise source
        # RIRs_n /= np.sqrt(np.sum(np.abs(RIRs_n)**2, axis=0))    # Euclidean norm scaling
        RTFs_n = np.fft.fft(RIRs_n, axis=0)
        RTFs_n = RTFs_n[:int(np.round(RTFs_n.shape[0] / 2)), :]   # Only consider positive frequencies
        # RTFs_n = np.random.uniform(size=RTFs_n.shape)
        # w_target = mvdr(RTFs, RTFs_n)   # Compute MVDR-like filter
        w_target = mvdr(RTFs_n, RTFs)   # Compute MVDR-like filter
    else:
        w_target = mfb(RTFs)    # Compute Matched Filter Beamformer h/(h^H*h)

    xx,yy,zz = gridify(rd, targetSource[-1], gridres=gres, plotgrid=False)  # Gridify room
    en, en_pf, freqs = getenergy(xx,yy,zz,rir_dur,Fs,micpos,rd,refCoeff,w_target)  # Get energy over grid

    # Plot
    foldername = 'C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\00_notformeetings\\01_MFB'
    fname = '%s\\spatialvisu_MFB_%i' % (foldername,refidx)
    plotspatialresp(xx,yy,en,targetSource,micpos,dBscale=1,exportit=exportit,exportname=fname,noiseSource=noiseSource)
    if makeGIF:
        plotspatialresp(xx,yy,en_pf,targetSource,micpos,dBscale=1,exportit=exportit,exportname=fname,freqs=freqs,noiseSource=noiseSource)

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
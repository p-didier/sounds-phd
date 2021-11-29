from posixpath import basename
import random
import numpy as np
import numpy.matlib
import os, sys
import random
import pandas as pd
import math
from sklearn import preprocessing
import soundfile as sf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from plotting.threedim import plot_room, set_axes_equal
from playsounds.playsounds import playthis

def load_AS(fname, path_to, plot_AS=None, plot_AS_dir=''):
    # load_AS -- Loads an Acoustic Scenario (AS) from a CSV file.
    # CSV file creation done with <ASgen.py>.
    #
    # >>> Inputs:
    # -fname [string] - Name of CSV file (w/ or w/out ".csv" extension).
    # -path_to [string] - Relative or absolute path to AS files directory.
    # -plot_AS [str or None] - Plot and export of the Acoustic Scenario's geometry as figure or GIF.
                            # -None: Do not plot.
                            # -'plot': Just plot, do not export.
                            # -'PNG': Plot and export as PNG.
                            # -'PDF': Plot and export as PDF.
                            # -'GIF': Plot and export for GIF making (many rotated PNGs).
    # -plot_AS_dir [str] - Directory where to export the plots (if asked).
    #  
    # >>> Outputs:
    # -h_sn [N*J*Ns float matrix, - ] - Source-to-node RIRs. 
    # -h_nn [N*J*Nn float matrix, - ] - Noise-to-node RIRs. 
    # -rs [Ns*3 float matrix, m] - Speech source positions.
    # -r [J*3 float matrix, m] - Node positions.
    # -rn [Nn*3 float matrix, m] - Noise source positions.
    # -rd [3*1 float vector, m] - Room dimensions.
    # -alpha [float, -] - Walls absorption coefficient.
    # -Fs [int, samples/s] - Sampling frequency.

    # (c) Paul Didier - 13-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Check if path exists
    if not os.path.isdir(path_to):
        raise ValueError('The path provided does not exist')

    # Check format of file name
    if fname[-4:] != '.csv':
        fname += '.csv'

    # Fetch CSV file
    if fname == '.csv':
        # Choose random file from <path_to>
        csv_files = [f for f in os.listdir(path_to) if os.path.splitext(f)[1] == '.csv']   # only select CSV files
        random.seed()
        myfile = path_to + '\\' + csv_files[random.randint(0,len(csv_files)-1)]
    else:
        myfile = path_to + '\\' + fname
    
    # Check file name validity
    if not os.path.isfile(myfile):
        raise ValueError('The file name specified is invalid')

    # Load dataframe
    AS = pd.read_csv(myfile,index_col=0)

    # Versioning check - new dataframe format as of 5/10/2021
    flag_old_df = False
    if not hasattr(AS, 'nNodes'):
        flag_old_df = True

    # Count sources and receivers
    Ns = sum('Source' in s for s in AS.index)       # number of speech sources
    Nn = sum('Noise' in s for s in AS.index)        # number of noise sources
    if flag_old_df:
        M = sum('Node' in s for s in AS.index)        # number of sensors
    else:
        M = sum('Sensor' in s for s in AS.index)        # number of sensors

    # Extract single-valued data
    rd              = [f for f in AS.rd if not math.isnan(f)]               # room dimensions
    alpha           = [f for f in AS.alpha if not math.isnan(f)]            # absorption coefficient
    alpha           = alpha[1]
    Fs              = [f for f in AS.Fs if not math.isnan(f)]               # sampling frequency
    Fs              = int(Fs[1])
    if flag_old_df:       # new dataframe format as of 5/10/2021
        nNodes = M
        d_intersensor = None
    else:                           # old dataframe format
        nNodes          = [f for f in AS.nNodes if not math.isnan(f)]           # number of nodes
        nNodes          = int(nNodes[1])
        d_intersensor   = [f for f in AS.d_intersensor if not math.isnan(f)]    # inter-sensor distance
        d_intersensor  = int(d_intersensor[1])

    # Extract coordinates
    rs = np.zeros((Ns,3))
    for ii in range(Ns):
        rs[ii,0] = AS.x['Source ' + str(ii + 1)]
        rs[ii,1] = AS.y['Source ' + str(ii + 1)]
        rs[ii,2] = AS.z['Source ' + str(ii + 1)]

    r = np.zeros((M,3))
    if flag_old_df:
        for ii in range(M):
            r[ii,0] = AS.x['Node ' + str(ii + 1)]
            r[ii,1] = AS.y['Node ' + str(ii + 1)]
            r[ii,2] = AS.z['Node ' + str(ii + 1)]
    else:
        for ii in range(M):
            r[ii,0] = AS.x['Sensor ' + str(ii + 1)]
            r[ii,1] = AS.y['Sensor ' + str(ii + 1)]
            r[ii,2] = AS.z['Sensor ' + str(ii + 1)]

    rn = np.zeros((Nn,3))
    for ii in range(Nn):
        rn[ii,0] = AS.x['Noise ' + str(ii + 1)]
        rn[ii,1] = AS.y['Noise ' + str(ii + 1)]
        rn[ii,2] = AS.z['Noise ' + str(ii + 1)]

    # Extract RIRs
    RIR_length = int(1/Ns*sum('h_sn' in s for s in AS.index))    # number of samples in one RIR
    h_sn = np.zeros((RIR_length,M,Ns))
    h_nn = np.zeros((RIR_length,M,Nn))
    for ii in range(M):
        if flag_old_df:
            ASchunk = AS['Node ' + str(ii+1)]
        else:
            ASchunk = AS['Sensor ' + str(ii+1)]
        for jj in range(Ns):
            h_sn[:,ii,jj] = ASchunk['h_sn ' + str(jj+1)].to_numpy()    # source-to-node RIRs
        for jj in range(Nn):
            h_nn[:,ii,jj] = ASchunk['h_nn ' + str(jj+1)].to_numpy()    # noise-to-node RIRs

    # Reference text for the AS
    reftxt = fname.replace('\\','_')[:-4]

    # Plot
    if plot_AS is not None:
        scatsize = 20
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)
        plot_room(ax, rd)
        for ii in range(Ns):
            p1 = ax.scatter(rs[ii,0],rs[ii,1],rs[ii,2],s=scatsize,c='blue',marker='d')
            ax.text(rs[ii,0],rs[ii,1],rs[ii,2],"D%i" % (ii+1))
        for ii in range(Nn):
            p2 = ax.scatter(rn[ii,0],rn[ii,1],rn[ii,2],s=scatsize,c='red',marker='P')
            ax.text(rn[ii,0],rn[ii,1],rn[ii,2],"N%i" % (ii+1))
        for ii in range(M):
            p3 = ax.scatter(r[ii,0],r[ii,1],r[ii,2],s=scatsize,c='green',marker='o')
            ax.text(r[ii,0],r[ii,1],r[ii,2],"S%i" % (ii+1))
        # titl = 'Acoustic scenario geometry ($\\alpha$ = %.2f)' % alpha
        titl = ''
        ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', zlabel='$z$ [m]',
            title=titl)
        ax.grid()
        # Check if 2-D
        if all(x==r[0,-1] for x in np.concatenate((r[:,-1],rs[:,-1],rn[:,-1]))):
        # if (r[:,-1] == rs[:,-1]).all() and (r[:,-1] == rn[:,-1]).all():
            ax.view_init(90,0)
            ax.set_zticks([])
            ax.set(zlabel='')
        set_axes_equal(ax)
        exportname = 'AS_%s' % (reftxt)
        if plot_AS_dir != '':
            exportname = '%s\\%s' % (plot_AS_dir,exportname)
        if plot_AS == 'PDF':
            ax.legend([p1,p2,p3],['Speech source', 'Noise source', 'Sensor'], loc=1)
            plt.savefig('%s.pdf' % exportname)
        elif plot_AS == 'PNG':
            ax.legend([p1,p2,p3],['Speech source', 'Noise source', 'Sensor'], loc=1)
            plt.savefig('%s.png' % exportname)
        elif plot_AS == 'GIF':
            ax.set(title='')
            angles = np.linspace(0, 360, 50)
            for idx, angle in enumerate(angles):
                ax.view_init(20, angle)
                plt.savefig('%s_%i.png' % (exportname, int(idx)), bbox_inches='tight')
        elif plot_AS == 'plot':
            plt.show()

    return h_sn,h_nn,rs,r,rn,rd,alpha,Fs,nNodes,d_intersensor,reftxt


def load_speech(fname, datasetsPath='C:\\Users\\u0137935\\Dropbox\\BELGIUM\\KU Leuven\\SOUNDS_PhD\\02_research\\03_simulations\\99_datasets\\01_signals'):
    # load_speech -- Loads a speech signal from database (or path) <d>.d

    if type(fname) == list:
        fname = fname[0]

    if any(substring in fname for substring in ['\\','/']):
        # Load directly
        d, Fs = sf.read(fname)
    else:
        # Load from database (pick at random)

        # Find library directory
        dirs = os.listdir(datasetsPath)
        for currdir in dirs:
            if fname in currdir.lower():   # case-insensitive check
                dataset = currdir
                break
            elif currdir == dirs[-1]:
                raise ValueError('The specified speech library was not found')

        # Look for suitable speech file in library
        foundit = False
        while not foundit:
            dir_files = os.listdir(datasetsPath + '\\' + dataset)
            # Randomly shuffle the directories
            dir_files = sorted(dir_files, key=lambda _: random.randint(0, 1))
            for currsubdir in dir_files:
                ext = os.path.splitext(currsubdir)[1]   # get extension
                if ext != '' and any(substring in ext for substring in ['.wav','.flac','.mp3']): # check whether current folder contains audio files
                    idx = random.randint(0,len(dir_files)-1)
                    ext = os.path.splitext(dir_files[idx])
                    while not any(substring in ext for substring in ['.wav','.flac','.mp3']):
                        idx = random.randint(0,len(dir_files)-1)
                        ext = os.path.splitext(dir_files[idx])
                    
                    d, Fs = sf.read(datasetsPath + '\\' + dataset + '\\' + dir_files[idx])
                    foundit = True
                    print('Loaded speech file "%s" from data set "%s"' % (dir_files[idx],dataset))
                    break
                elif os.path.isdir(datasetsPath + '\\' + dataset + '\\' + currsubdir):
                    dataset = dataset + '\\' + currsubdir
                    break
            stop = 1

    return d, Fs


def sig_gen(path_to,speech_lib,Tmax,noise_type,baseSNR,pauseDur=0,pauseSpace=float('inf'),\
    ASref='',speech='',noise_in='',plotAS=None, plot_AS_dir='',ms='overlap',SNR_in_uncorr=30,\
        prewhiten=False,spatially_white_noise=False):
    # sig_gen -- Generates microphone signals based on a given acoustic scenario
    # and dry speech/noise data.
    #
    # >>> Inputs:
    # -path_to [str] - Path to folder containing acoustic scenarios files. 
    # -speech_lib [str] - Path or keyword corresponding to speech database to be used.
    # -Tmax [float, s] - Duration of output signal.
    # -noise_type [str] - Type of background noise. Possible values: 'white'.
    # -baseSNR [float, dB] - Dry speech-to-dry noise SNR.
    # -pauseDur [float, s] - Duration of forced pauses in speech.
    # -pauseSpace [float, s] - Duration of forced-pause-free segments in-between forced pauses.
    # -ASref [str] - If not '', path to specific acoustic scenario (AS) to be used. Else, random AS is chosen. 
    # -speech [str /or/ list of str] - Path(s) to specific speech file(s) to be used.
    # -noise [str /or/ list of str] - Path(s) to specific noise file(s) to be used.
    # -plot_AS [str or None] - Plot and export of the Acoustic Scenario's geometry as figure or GIF.
                            # -None: Do not plot.
                            # -'plot': Just plot, do not export.
                            # -'PNG': Plot and export as PNG.
                            # -'PDF': Plot and export as PDF.
                            # -'GIF': Plot and export for GIF making (many rotated PNGs).
    # -plot_AS_dir [str] - Directory where to export the plots (if asked).
    # -ms [str] - Option for multi-speakers speech signal generation:
                #   -'overlap': the speakers may speak simultaneously.
                #   -'distinct': the speakers may never speak simultaneously.
    # -SNR_in_uncorr [float, dB] - Input SNR for microphone self-noise.
    # -prewhiten [bool] - If true, whiten the wet clean speech signal mixtures.
    # -spatially_white_noise [bool or 'combined'] - Option for use of spatially-white noise:
                #   -false, do not assume any spatially white noise and use specific noise source positions.
                #   -true, assume spatially white noise and ignore entirely specific noise source positions.
                #   -'combined', use specific noise source positions and add some spatially white noise.
    
    # Load acoustic scenario
    h_sn,h_nn,rs,r,rn,rd,alpha,Fs,nNodes,d_intersensor,reftxt = load_AS(ASref, path_to, plotAS, plot_AS_dir)
    print('Loading acoustic scenario: %.1fx%.1fx%.1f = %.1f m^3; alpha = %.2f...' % (rd[0],rd[1],rd[2],np.prod(rd),alpha))
    if alpha == 1:
        print('--> Fully anechoic scenario')
    print('\n')

    # Quick plot
    # fig = plt.figure(figsize=(4,4), constrained_layout=True)
    # for ii in range(h_sn.shape[1]):
    #     ax = fig.add_subplot(h_sn.shape[1],1,ii+1)
    #     ax.plot(np.arange(h_sn.shape[0])/Fs,h_sn[:,ii,0])
    #     ax.grid()
    #     ax.set(yticklabels=[])
    #     plt.ylabel('S%i'%(ii+1))
    #     if ii < h_sn.shape[1]-1:
    #         ax.set(xticklabels=[])
    # plt.xlabel('$t$ [s]') 
    # plt.suptitle('Sensor-specific RIRs')
    # plt.show()
    # fig.savefig("Python_RIRs.png")

    # Extract useful values
    Ns = h_sn.shape[-1]             # number of speech sources
    Nn = h_nn.shape[-1]             # number of noise sources
    M = h_sn.shape[1]               # number of sensors
    nmax = int(np.floor(Fs*Tmax))   # max number of samples in speech signal

    # Check inputs
    if speech != '':
        speech_lib = speech
    if type(speech_lib) != list:
        speech_lib = [speech_lib]

    if len(speech_lib) == 1 or len(speech_lib) < Ns:
        print('Using speech library "%s"' % speech_lib)
        speech_lib = [speech_lib for ii in range(Ns)]   # make sure there is one value of <speech_lib> per source
    if len(speech_lib) > Ns:
        print('Too many speech files references were provided. Using first %i.' % Ns)
        speech_lib = speech_lib[:Ns]

    if noise_in != '' and len(noise_in) < Nn:
        raise ValueError('Not enough specific noise files provided to cover %i noise sources' % Nn)
    if len(noise_in) > Nn:
        print('Too many noise files references were provided. Using first %i.' % Nn)
        noise_in = noise_in[:Nn]

    # Load DRY speech signal(s)
    d = np.zeros((nmax,Ns))
    for ii in range(Ns):
        d_curr, Fs2 = load_speech(speech_lib[ii])
        # Check that sampling frequencies match btw. speech and RIR
        if Fs != Fs2:
            raise ValueError('The sampling rates of the speech signals and the RIR do not match')
        d_curr = preprocessing.scale(d_curr)   # normalize (mean 0 + unit variance)
        if len(d_curr) > nmax:
            d_curr = d_curr[:nmax]
        else:
            # Loop dry signal if needed to read the desired duration <Tmax>
            flagrep = False
            while not flagrep:
                if nmax - len(d_curr) > len(d_curr):
                    d_add = d_curr
                else:
                    d_add = d_curr[:nmax-len(d_curr)]
                    flagrep = True
                d_curr = np.concatenate([d_curr,d_add])

        d[:,ii] = d_curr

    # Deal with user-input speech characteristics: forced pauses in speech + multi-speakers scheme
    d = add_pauses(d, pauseDur, pauseSpace, Fs, ms)
    
    # Generate DRY noise signals
    noise = np.zeros((nmax, Nn))
    if noise_in == '':
        # Generate new noise
        for ii in range(Nn):
            if noise_type == 'white':
                noisecurr = np.random.normal(0,1,nmax)
            else:
                print('<sig_gen>: WARNING - Other options for noise than "white" have not been implemented yet. Will use white noise.')
                noisecurr = np.random.normal(0,1,nmax)
            noisecurr = preprocessing.scale(noisecurr)   # normalize
            noise[:,ii] = 10**(-baseSNR/20)*noisecurr    # ensure desired SNR w.r.t. to speech sound sources
    else:
        for ii in range(Nn):
            noisecurr, Fsn = sf.read(noise_in[ii])
            if Fsn != Fs:
                raise ValueError('The noise file provided has a mismatching sampling frequency')
            # Adapt length (trim or repeat noise file if needed)
            if len(noisecurr) > nmax:
                noisecurr = noisecurr[:nmax]
            else:
                while len(noisecurr) < nmax:
                    noisecurr = np.concatenate((noisecurr, noisecurr))
                    if len(noisecurr) > nmax:
                        noisecurr = noisecurr[:nmax]
                        break
            # noisecurr = preprocessing.scale(noisecurr)   # normalize
            # Apply base SNR to noise
            sp_pow = np.amax(np.var(d, axis=0))
            sf_uncorr = np.sqrt(sp_pow/10**(baseSNR/10))            # Scale factor for desired SNR for uncorr noise, assume initial std. dev of uncorr noise = 1s
            noise[:,ii] = sf_uncorr*noisecurr    # ensure desired SNR w.r.t. to speech sound sources
            
    # Generate no-noise WET sensor signals ("desired signals" -- convolution w/ RIRs only)
    ds = np.zeros((nmax,M))
    ds_single_sources = np.zeros((nmax,M,Ns))
    for k in range(M):
        d_k = np.zeros(nmax)
        for ii in range(Ns):
            d_kk = scipy.signal.fftconvolve(d[:,ii], np.squeeze(h_sn[:,k,ii]))
            d_k += d_kk[:nmax]
            # Extract single-source wet no-noise mic signal
            ds_single_sources[:,k,ii] = d_kk[:nmax]

        ds[:,k] = d_k

    # If asked, "whiten" WET sensor signals
    if prewhiten:
        Ds = np.fft.fft(ds, axis=0)
        psd = np.abs(Ds)
        ds = np.real(np.fft.ifft(Ds / psd, axis=0))

    # Generate no-speech WET sensor noise (convolution w/ RIRs and/or spatially white noise)
    ny = np.zeros((nmax,M))
    if spatially_white_noise is True:
        for k in range(M):
            ny[:,k] = np.random.uniform(low=-1,high=1,size=nmax) * noise[:,0]
            h_nn = np.nan   # Spatially white noise ONLY -- no localized noise source
            rn = np.nan     # Spatially white noise ONLY -- no localized noise source
    elif spatially_white_noise == 'combined':
        for k in range(M):
            n_k = np.zeros(nmax)
            for ii in range(Nn):
                n_kk = scipy.signal.fftconvolve(noise[:,ii], np.squeeze(h_nn[:,k,ii]))
                n_k += n_kk[:nmax]
            ny[:,k] = n_k
            # Add spatially white noise
            ny[:,k] += np.amax(n_k) * np.random.uniform(low=-1,high=1,size=nmax) * noise[:,0]   
    else:
        for k in range(M):
            n_k = np.zeros(nmax)
            for ii in range(Nn):
                n_kk = scipy.signal.fftconvolve(noise[:,ii], np.squeeze(h_nn[:,k,ii]))
                n_k += n_kk[:nmax]
            ny[:,k] = n_k
    
    # Generate microphone signals (speech + noise)
    y = ds + ny
    y_single_sources = np.zeros_like(ds_single_sources)
    for ii in range(Ns):
        y_single_sources[:,:,ii] = ds_single_sources[:,:,ii] + ny

    # Add self-noise
    y = add_selfnoise(y,SNR_in_uncorr)

    # Time vector
    t = np.arange(nmax)/Fs

    return y,ds,d,ny,y_single_sources,t,Fs,nNodes,rd,r,rs,rn,alpha,reftxt


def add_selfnoise(y, SNR):
    # Adds microphone self-noise to microphone signals.

    nchannels = 1
    if len(y.shape) == 2:
        if y.shape[0] < y.shape[1]:
            y = np.transpose(y)
        nchannels = np.amin(y.shape[1])

    y_sn = np.copy(y)
    for ii in range(nchannels):
        sp_pow = np.var(y[:,ii])
        sf_uncorr = np.sqrt(sp_pow/10**(SNR/10))            # Scale factor for desired SNR for uncorr noise, assume initial std. dev of uncorr noise = 1
        n_uncorr = sf_uncorr*(2 * np.random.uniform(size=(y.shape[0],)) - 1)      # Uncorrelated noise to be added
        y_sn[:,ii] += n_uncorr

    return y_sn


def add_pauses(s,pauseDur,pauseSpace,Fs,ms='overlap'):
    # Adds pauses to speech signal <s> in a way that preserves natural pauses.
    # 
    # >>> Inputs:
    # -s [L*Ns float matrix, -] - Input speech signal(s).
    # -pauseDur [float, s] - Duration of forced pauses in speech.
    # -pauseSpace [float, s] - Duration of forced-pause-free segments in-between forced pauses.
    # -Fs [int, samples/s] - Sampling frequency.
    # -ms [str] - Option for multi-speakers speech signal generation:
                #   -'overlap': the speakers may speak simultaneously.
                #   -'distinct': the speakers may never speak simultaneously.
    #
    # >>> Outputs:
    # -s_wp [L*Ns float matrix, -] - Output signal(s), with appropriate forced pauses.
    
    # Extract parameters from inputs
    Ns = s.shape[1]                 # Number of speech sources
    Ns_p = int(Fs*pauseDur)         # Number of samples per in-between-speech pause
    Ns_ibp = int(Fs*pauseSpace)     # Minimum number of samples between two consecutive pauses
    flagOverlap = Ns == 1 or ms == 'overlap'

    # Segment signals
    segments = []
    for ii in range(Ns):
        if pauseDur > 0:
            # Detect natural pauses in file
            pauses = detect_pauses(s[:,ii], Fs)
            # Cut-up files into speech segments at least <Ns_ibp> samples long
            pauses_sep = [pauses[0]]
            for p in pauses:
                if p - pauses_sep[-1] >= Ns_ibp:
                    pauses_sep.append(p)        # Extract pause indices which ensures at least <Ns_ipb> samples in-between them
            currsegments = [s[int(pauses_sep[jj-1]):int(pauses_sep[jj]), ii] for jj in range(1,len(pauses_sep))]
            currsegments.append(s[int(pauses_sep[-1]):, ii])
            segments.append(currsegments)
        else:
            segments.append([s[:,ii]])

    # Build output signals
    s_wp = np.zeros_like(s)
    idx_end = 0
    idx_start = 0
    while idx_start < s.shape[0] and idx_end < s.shape[0]:
        mysegs = []
        maxlenseg = 0
        for ii in range(Ns):   # for each speech source
            # Select the first segment available
            mysegs.append(segments[ii][0])
            # Memorize the maximum segment length
            if len(mysegs[-1]) > maxlenseg:
                maxlenseg = len(mysegs[-1])     
            # Delete the segment
            segments[ii].pop(0)

        # Update the ending filling-in index
        idx_end += maxlenseg
        
        # Build output signals
        if flagOverlap:
            for ii in range(Ns):
                addition = np.concatenate((mysegs[ii], np.zeros(maxlenseg - len(mysegs[ii]))))
                if idx_end > s.shape[0]:
                    addition = addition[:(s.shape[0] - idx_start)]
                s_wp[idx_start:idx_end, ii] = addition
            # Increment indices for next iteration
            idx_start += maxlenseg + Ns_p
            idx_end += Ns_p
        else:
            for ii in range(Ns):
                idx_end = idx_start + len(mysegs[ii])
                addition = mysegs[ii]
                if idx_end > s.shape[0]:
                    addition = addition[:(s.shape[0] - idx_start)]
                s_wp[idx_start:idx_end, ii] = addition
                idx_start += len(addition) + Ns_p

    return s_wp


def detect_pauses(s, Fs):
    # Returns indices at the center of natural pauses in speech signal <s>.

    # Parameters
    tw = 20e-3      # window duration
    Np = int(tw*Fs)      # window length
    nw = int(np.floor(len(s)/Np))   # number of window needed to cover the whole signal
    thrs = np.amax(s**2)/2000   # Energy threshold detecting a pause in speech

    # Label windows as containing speech (False) or pause (True)
    flags = [np.mean((s[ii*Np:(ii+1)*Np])**2) <= thrs for ii in range(nw)]

    # Detect beginning and ends of pause segments
    idx_begin = [ii for ii in range(1,nw) if not flags[ii-1] and flags[ii]]
    idx_end = [ii for ii in range(nw-1) if flags[ii] and not flags[ii+1]]
    if flags[0]:
        idx_begin.insert(0,0)
    if idx_end[-1] < nw and flags[-1]:
        idx_end.append(nw)

    # Mark middles of pause segments
    pause_idx = np.round((np.array(idx_end) - np.array(idx_begin))/2) + np.array(idx_begin)
    pause_idx = np.round(pause_idx * Np + Np/2)     # convert back to original sampling rate indices

    return pause_idx


# def plot_data_indep_BF(W,freqs,rd,alpha,r,Fs,win,L,R,rs,rn):

#     spatial_visu_MWF(W,freqs,rd,alpha,r,Fs,win,L,R,targetSources=rs,noiseSources=rn)

#     return None
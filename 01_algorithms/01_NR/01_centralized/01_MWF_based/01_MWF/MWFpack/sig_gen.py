import random
import numpy as np
import os
import random
import pandas as pd
import math
from sklearn import preprocessing
import soundfile as sf
import time
import scipy.signal

def load_AS(fname, path_to):
    # load_AS -- Loads an Acoustic Scenario (AS) from a CSV file.
    # CSV file creation done with <ASgen.py>.
    #
    # >>> Inputs:
    # -fname [string] - Name of CSV file (w/ or w/out ".csv" extension).
    # -path_to [string] - Relative or absolute path to AS files directory.
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
    if fname[-5:-1] != '.csv':
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

    # Extract data
    rd = [f for f in AS.rd if not math.isnan(f)]            # room dimensions
    alpha = [f for f in AS.alpha if not math.isnan(f)]      # absorption coefficient
    alpha = alpha[1]
    Fs = [f for f in AS.Fs if not math.isnan(f)]            # sampling frequency
    Fs = Fs[1]

    # Count sources and receivers
    Ns = sum('Source' in s for s in AS.index)
    J = sum('Node' in s for s in AS.index)
    Nn = sum('Noise' in s for s in AS.index)

    # Extract coordinates
    rs = np.zeros((Ns,3))
    for ii in range(Ns):
        rs[ii,0] = AS.x['Source ' + str(ii + 1)]
        rs[ii,1] = AS.y['Source ' + str(ii + 1)]
        rs[ii,2] = AS.z['Source ' + str(ii + 1)]

    r = np.zeros((J,3))
    for ii in range(J):
        r[ii,0] = AS.x['Node ' + str(ii + 1)]
        r[ii,1] = AS.y['Node ' + str(ii + 1)]
        r[ii,2] = AS.z['Node ' + str(ii + 1)]

    rn = np.zeros((Nn,3))
    for ii in range(Nn):
        rn[ii,0] = AS.x['Noise ' + str(ii + 1)]
        rn[ii,1] = AS.y['Noise ' + str(ii + 1)]
        rn[ii,2] = AS.z['Noise ' + str(ii + 1)]

    # Extract RIRs
    RIR_length = int(1/Ns*sum('h_sn' in s for s in AS.index))    # number of samples in one RIR
    h_sn = np.zeros((RIR_length,J,Ns))
    h_nn = np.zeros((RIR_length,J,Nn))
    for ii in range(J):
        ASchunk = AS['Node ' + str(ii+1)]
        for jj in range(Ns):
            h_sn[:,ii,jj] = ASchunk['h_sn ' + str(jj+1)].to_numpy()    # source-to-node RIRs
        for jj in range(Nn):
            h_nn[:,ii,jj] = ASchunk['h_nn ' + str(jj+1)].to_numpy()    # noise-to-node RIRs

    return h_sn,h_nn,rs,r,rn,rd,alpha,Fs


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
                elif os.path.isdir(datasetsPath + '\\' + dataset + '\\' + currsubdir):
                    dataset = dataset + '\\' + currsubdir
                    break
            stop = 1

    return d, Fs

def sig_gen(path_to,speech_lib,Tmax,noise_type,baseSNR,pauseDur=0,pauseSpace=float('inf'),ASref='',speech=''):
    # sig_gen -- Generates microphone signals based on a given acoustic scenario
    # and dry speech/noise data.
    
    # Load acoustic scenario
    h_sn,h_nn,rs,r,rn,rd,alpha,Fs = load_AS(ASref, path_to)
    Ns = h_sn.shape[-1]            # number of speech sources
    Nn = h_sn.shape[-1]            # number of noise sources
    J = h_sn.shape[1]              # number of nodes
    nmax = int(np.floor(Fs*Tmax))    # max number of samples in speech signal

    # Check inputs
    if speech != '':
        speech_lib = speech
    if type(speech_lib) != list:
        speech_lib = [speech_lib]

    if len(speech_lib) == 1 or len(speech_lib) < Ns:
        speech_lib = [speech_lib for ii in range(Ns)]   # make sure there is one value of <speech_lib> per source
    if len(speech_lib) > Ns:
        print('Too many speech files references were provided. Using first %i.' % Ns)
        speech_lib = speech_lib[:Ns]

    # Load speech signal(s)
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

    # Add pauses, if asked
    if pauseDur > 0:
        Ns_p = int(Fs*pauseDur)                  # number of samples per in-between-speech pause
        Ns_ibp = int(Fs*pauseSpace)              # number of samples between two consecutive pauses
        Np = int(np.ceil(nmax/(Ns_p + Ns_ibp)))  # number of pauses, given the speech signal length

        # Build signal with pauses
        # t0 = time.time()
        d_wp = np.zeros(d.shape)
        for ii in range(Np):
            # Indices of output vector to be filled during current iteration
            idx_ii = range(np.amin([nmax, ii*(Ns_p + Ns_ibp)]), np.amin([nmax, (ii+1)*(Ns_p + Ns_ibp)]))
            # Indices of input vector to be used during current iteration
            idx_d = range(np.amin([nmax, ii*Ns_ibp]), np.amin([nmax, (ii+1)*Ns_ibp]))

            chunk = np.concatenate((d[idx_d,:], np.zeros((Ns_p,d.shape[1]))))
            d_wp[idx_ii, :] = chunk[:len(idx_ii),:]
        # t1 = time.time()
        # print('Time elapsed: %f s' % (t1-t0))

        # t0 = time.time()
        # d_wp2 = np.zeros(d.shape)
        # for ii in range(d_wp.shape[1]):

        #     d_wp2[:,ii] = [0 if (jj/Fs)/(pauseDur + pauseSpace) % 1 >= 0.5 else d_wp[jj,ii]\
        #          for jj in range(d_wp2.shape[0])]
        # t1 = time.time()
        # print('Time elapsed: %f s' % (t1-t0))

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(d_wp[:,0])
        # ax.plot(d_wp2[:,0]+10)
        # plt.show()

        d = d_wp
    

    # Generate noise signals
    noise = np.zeros((nmax, Nn))
    for ii in range(Nn):
        if noise_type == 'white':
            noisecurr = np.random.normal(0,1,nmax)
        else:
            print('<sig_gen>: WARNING - Other options for noise than "white" have not been implemented yet. Will use white noise.')
            noisecurr = np.random.normal(0,1,nmax)
        noisecurr = preprocessing.scale(noisecurr)   # normalize
        noise[:,ii] = 10**(-baseSNR/20)*noisecurr    # ensure desired SNR w.r.t. to speech sound sources

    # Generate no-noise sensor signals ("desired signals" -- convolution w/ RIRs only)
    ds = np.zeros((nmax,J))
    for k in range(J):
        d_k = np.zeros(nmax)
        for ii in range(Ns):
            # d_kk = np.convolve(d[:,ii], np.squeeze(h_sn[:,k,ii]))
            d_kk = scipy.signal.fftconvolve(d[:,ii], np.squeeze(h_sn[:,k,ii]))
            d_k += d_kk[:nmax]
        ds[:,k] = d_k

    # Generate no-speech sensor noise (convolution w/ RIRs only)
    ny = np.zeros((nmax,J))
    for k in range(J):
        n_k = np.zeros(nmax)
        for ii in range(Ns):
            # n_kk = np.convolve(noise[:,ii], np.squeeze(h_nn[:,k,ii]))
            n_kk = scipy.signal.fftconvolve(noise[:,ii], np.squeeze(h_nn[:,k,ii]))
            n_k += n_kk[:nmax]
        ny[:,k] = n_k
    
    # Generate microphone signals (speech + noise)
    y = ds + ny

    # Time vector
    t = np.arange(nmax)/Fs

    return y,ds,ny,t,Fs
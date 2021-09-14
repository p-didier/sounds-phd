#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import random
from rimPy import rimPy

# Acoustic Scenario (AS) generation script.

def main():
    nAS = 10        # Number of AS to generate
    Fs = 16e3       # Sampling frequency [samples/s]
    RIR_l = 2**12   # RIR length [samples]
    minRd = 3       # Smallest room dimension possible [m]
    maxRd = 7       # Largest room dimension possible [m]
    Ns = 1          # nr. of speech sources
    Nn = 1          # nr. of noise sources
    J = 5           # nr. of nodes

    T60max = 1.5*RIR_l/Fs   # Largest possible T60
    T60min = 0.4*RIR_l/Fs   # Smallest possible T60

    counter = 0
    while counter < nAS:

        rd = np.random.uniform(low=minRd, high=maxRd, size=(3,))    # Generate random room dimensions
        T60 = random.uniform(T60min, T60max)
        # T60 = np.random.uniform(low=T60min, high=T60max, size=(1,)) # Generate random reverberation time
        V = np.prod(rd)                                   # Room volume
        S = 2*(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2])   # Total room surface area
        alpha = np.minimum(1, 0.161*V/(T60*S))                # Absorption coefficient of the walls
        
        # Call function
        h_ns, h_nn, rs, rn, r = genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,1)
        
        # Export
        expfolder = "C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\02_data\\01_acoustic_scenarios"
        nas = len(next(os.walk(expfolder))[2])   # count only files in export dir
        fname =  "%s\\AS%i_J%i_Ns%i_Nn%i.csv" % (expfolder,nas,J,Ns,Nn)
        header = {'rd': pd.Series(np.squeeze(rd)), 'alpha': alpha, 'Fs': Fs}
        export_data(h_ns, h_nn, header, rs, rn, r, fname)

        counter += 1

    print('\n\nAll done.')

    return h_ns, h_nn
        

def genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,export=True):
    # genAS -- Computes the RIRs in a rectangular cavity where sensors, speech
    # sources, and noise sources are present.
    # 
    # >>> Inputs:
    # -rd [3*1 (or 1*3) float array, m] - Room dimensions [x,y,z].
    # -J [int, -] - # of sensors.
    # -Ns [int, -] - # of desired sources.
    # -Nn [int, -] - # of noise sources.
    # -alpha [float, -] - Walls absorption coefficient (norm. inc.).
    # -RIR_l [int, #samples] - Length of RIRs to produce.
    # -Fs [float, Hz] - Sampling frequency.
    # -export [bool] - If true, export AS (RIRs + parameters) as .MAT.
    # >>> Outputs:
    # -h_sn [RIR_l*J*Ns (complex) float 3-D array, -] - RIRs between desired sources and sensors.  
    # -h_nn [RIR_l*J*Nn (complex) float 3-D array, -] - RIRs between noise sources and sensors.  

    # (c) Paul Didier - 08-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------
    
    # Random element positioning in 3-D space
    rs = np.multiply(np.random.rand(Ns,3),rd)   # speech sources positions
    r  = np.multiply(np.random.rand(J,3),rd)    # nodes positions
    rn = np.multiply(np.random.rand(Nn,3),rd)   # noise sources positions
    
    # ------------------ FIXED ARBITRARY ------------------
    # rs = np.array([[1,2,3],])   # speech sources positions
    # r = np.array([[0.1,0.1,0.1],])  # speech sources positions
    # alpha = 0.5
    # rd = np.array([5,6,7])
    # -----------------------------------------------------
    
    R = -1*np.sqrt(1 - alpha)   # Walls reflection coefficient
    
    h_sn = np.zeros((RIR_l, J, Ns))
    for ii in range(Ns):
        print('Computing RIRs from speech source #', ii+1, '...')
        h_sn[:,:,ii] = rimPy(r, rs[ii,:], rd, R, RIR_l/Fs, Fs)
    
    h_nn = np.zeros((RIR_l, J, Nn))
    for ii in range(Nn):
        print('Computing RIRs from noise source #', ii+1, '...')
        h_nn[:,:,ii] = rimPy(r, rn[ii,:], rd, R, RIR_l/Fs, Fs)
    
    return h_sn, h_nn, rs, rn, r
    
def export_data(h_sn, h_nn, header, rs, rn, r, fname):

    # Check if export folder exists
    mydir = os.path.dirname(fname)
    if not os.path.isdir(mydir):
        os.mkdir(mydir)   # if not, make directory
        print('Direction "%s" was created.' % mydir)

    # Source-to-node RIRs
    data_df = pd.DataFrame()   # init dataframe
    for ii in range(h_sn.shape[-1]):
        # Build dataframe for current matrix slice
        data_df_curr = pd.DataFrame(np.squeeze(h_sn[:,:,ii]),\
            columns=['Node %i' % (idx+1) for idx in range(h_sn.shape[1])],\
            index=['h_sn %i' % (ii+1) for idx in range(h_sn.shape[0])])

        # Concatenate to global dataframe
        data_df = pd.concat([data_df,data_df_curr])

    # Noise-to-node RIRs
    for ii in range(h_nn.shape[-1]):
        # Build dataframe for current matrix slice
        data_df_curr = pd.DataFrame(np.squeeze(h_nn[:,:,ii]),\
            columns=['Node %i' % (idx+1) for idx in range(h_nn.shape[1])],\
            index=['h_nn %i' % (ii+1) for idx in range(h_nn.shape[0])])

        # Concatenate to global dataframe
        data_df = pd.concat([data_df,data_df_curr])
            
    # Build header dataframe
    header_df = pd.DataFrame(header)

    # Build coordinates dataframes
    rs_df = pd.DataFrame(rs,\
                index=['Source %i' % (idx+1) for idx in range(rs.shape[0])],\
                columns=['x','y','z'])
    rn_df = pd.DataFrame(rn,\
                index=['Noise %i' % (idx+1) for idx in range(rn.shape[0])],\
                columns=['x','y','z'])
    r_df = pd.DataFrame(r,\
                index=['Node %i' % (idx+1) for idx in range(r.shape[0])],\
                columns=['x','y','z'])

    # Concatenate header + coordinates with rest of data
    big_df = pd.concat([header_df,rs_df,rn_df,r_df,data_df])
    big_df.to_csv(fname)

    print('Data exported to CSV: "%s"' % os.path.basename(fname))


main()
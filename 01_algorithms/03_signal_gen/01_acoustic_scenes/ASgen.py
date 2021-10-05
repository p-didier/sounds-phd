#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import random
from rimPy import rimPy

# Acoustic Scenario (AS) generation script.

def main(gen_specific_AS=False):
    nAS = 10        # Number of AS to generate
    Fs = 16e3       # Sampling frequency [samples/s]
    RIR_l = 2**12   # RIR length [samples]
    minRd = 3       # Smallest room dimension possible [m]
    maxRd = 7       # Largest room dimension possible [m]
    Ns = 2          # nr. of speech sources
    Nn = 3          # nr. of noise sources
    J = 5           # nr. of nodes

    T60max = 1.5*RIR_l/Fs   # Largest possible T60
    T60min = 0.4*RIR_l/Fs   # Smallest possible T60

    if gen_specific_AS:
        nAS = 1

    counter = 0
    while counter < nAS:
        
        if gen_specific_AS:
            rd = get_fixed_values()[0]
            # T60 = 0.25
            T60 = 0
        else:
            rd = np.random.uniform(low=minRd, high=maxRd, size=(3,))    # Generate random room dimensions
            T60 = random.uniform(T60min, T60max)
        # T60 = np.random.uniform(low=T60min, high=T60max, size=(1,)) # Generate random reverberation time
        V = np.prod(rd)                                   # Room volume
        S = 2*(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2])   # Total room surface area
        alpha = np.minimum(1, 0.161*V/(T60*S))                # Absorption coefficient of the walls
        
        # Call function
        if gen_specific_AS:
            h_ns, h_nn, rs, rn, r = genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,1,random_coords=False)
        else:
            h_ns, h_nn, rs, rn, r = genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,1,random_coords=True)

        # Export
        expfolder = "C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\02_data\\01_acoustic_scenarios"
        if gen_specific_AS:
            fname = '%s\\J%i_Ns%i_Nn%i\\testAS' % (expfolder,J,Ns,Nn)
            if alpha == 1:
                fname += '_anechoic'
            fname += '.csv'
        else:
            expfolder += '\\J%i_Ns%i_Nn%i' % (J,Ns,Nn)
            if not os.path.isdir(expfolder):   # check if subfolder exists
                os.mkdir(expfolder)   # if not, make directory
            nas = len(next(os.walk(expfolder))[2])   # count only files in export dir
            fname =  "%s\\AS%i.csv" % (expfolder,nas)
        header = {'rd': pd.Series(np.squeeze(rd)), 'alpha': alpha, 'Fs': Fs}
        export_data(h_ns, h_nn, header, rs, rn, r, fname)

        counter += 1

    print('\n\nAll done.')

    return h_ns, h_nn
        

def genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,export=True,random_coords=True):
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
    if random_coords:
        rs = np.multiply(np.random.rand(Ns,3),rd)   # speech sources positions
        r  = np.multiply(np.random.rand(J,3),rd)    # nodes positions
        rn = np.multiply(np.random.rand(Nn,3),rd)   # noise sources positions
    else:
        rd, r, rs, rn = get_fixed_values()
        r = r[:J,:]
        rs = rs[:Ns,:]
        rn = rn[:Nn,:]
    
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

def get_fixed_values():

    rd = np.array([6.74663681, 6.47443158, 5.29141806])
    rs = np.array([[3.80871696, 3.47644605, 0.13365643],
       [2.8506613 , 2.24521067, 4.7211736 ],
       [0.53659517, 2.58889843, 3.93477618],
       [1.15229464, 2.00410181, 4.44485973],
       [4.1974679 , 2.43351876, 1.97641962],
       [1.57881229, 5.03781146, 2.65542962],
       [6.65379346, 4.44628479, 0.43348834],
       [4.14120661, 3.94366795, 2.36864132],
       [1.76631629, 6.15191657, 2.40936931],
       [0.42973443, 4.18359708, 0.15160593]])
    r = np.array([[2.48886545, 4.34982861, 2.87315296],
       [5.24655834, 5.66887148, 4.35400199],
       [0.12285002, 4.59571634, 3.47397411],
       [2.78390363, 1.49391502, 0.84512655],
       [1.1217436 , 6.07061552, 0.86066287],
       [3.17113765, 5.94627376, 2.11941889],
       [0.00946956, 1.35901537, 4.40715656],
       [4.25251161, 4.83032505, 4.59872196],
       [2.18592129, 2.37155164, 2.28711986],
       [3.81597077, 4.05754139, 1.48683971]])
    rn = np.array([[3.17100024, 4.40443333, 2.16296961],
       [3.60656018, 3.79377093, 1.76541558],
       [0.25901077, 4.614144  , 2.83299575],
       [2.73518121, 0.30138353, 0.77814191],
       [5.74734719, 3.85861567, 2.75872101],
       [5.18246482, 5.12011156, 3.70950443],
       [2.67892363, 3.30854761, 3.59350624],
       [1.23639543, 5.74947558, 2.96543862],
       [1.68529516, 3.24728082, 4.82935064],
       [0.04326686, 4.47009184, 3.15532972]])

    return rd, r, rs, rn

main(gen_specific_AS=1)
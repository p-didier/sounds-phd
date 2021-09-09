#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os 

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
        T60 = np.random.uniform(low=T60min, high=T60max, size=(1,)) # Generate random reverberation time
        V = np.prod(rd)                                   # Room volume
        S = 2*(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2])   # Total room surface area
        alpha = np.minimum(1, 0.161*V/(T60*S))                # Absorption coefficient of the walls
        
        # Call function
        h_ns, h_nn = genAS(rd,J,Ns,Nn,alpha,RIR_l,Fs,1)
        

        # Export
        expfolder = "C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\02_data\\01_acoustic_scenarios"
        nas = len(next(os.walk(expfolder))[2])   # count only files in export dir
        fname =  "%s\\AS%i_J%i_Ns%i_Nn%i.txt" % (expfolder,nas,J,Ns,Nn)
        export_data(h_ns, fname)

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
    
    return h_sn, h_nn


def rimPy(mic_pos, source_pos, room_dim, beta, rir_length, Fs, rand_dist=0, Tw=None, Fc=None, c=343):
    # RIMPY  Randomized Image Method, Python implementation
    #
    #    This script generates the impulse response of the (randomised) image
    #    method as proposed in De Sena et al. "On the modeling of 
    #    rectangular geometries in  room acoustic simulations." IEEE/ACM 
    #    Transactions on Audio, Speech and Language Processing (TASLP) 23.4 
    #    (2015): 774-786.
    #    It can also generate the response of the standard image method,
    #    if needed. The script uses fractional delays as proposed by 
    #    Peterson, ``Simulating the response of multiple microphones to a 
    #    single acoustic source in a reverberant room,'' JASA, 1986.
    #  
    #    RIMPY(mic_pos, source_pos, room_dim, beta, rir_length) gives the room
    #    impulse response where:
    #    - mic_pos is the 3xM array with the position of M omnidirectional
    #    microphones [meters]. 
    #    - source_pos is the 3x1 array with the position of the omni sound
    #    source in [meters].
    #    - room_dim is the 3x1 array with the dimensions of the room [meters].
    #    - beta is the 2x3 array with the reflection coefficient of the walls,
    #    in this order: [x1, y1, z1; x2, y2, z2], with e.g. x1 and x2 are the 
    #    reflection coefficient of the surface orthogonal to the x-axis and 
    #    with the subscript 1 referring to walls adjacent to the coordinate 
    #    origin, and the subscript 2 referring to the opposite wall. In the
    #    anechoic case beta=np.zeros((2,3)). 
    #    - rir_length is the length of the RIR [seconds]
    #    - Fs is the sampling frequency [Hz]
    #  
    #    RIMPY(mic_pos, source_pos, room_dim, beta, rir_length, rand_dist, Tw, Fc, c)
    #    - rand_dist is the random distance added to the position of the 
    #    image sources in [meters] (rand_dist=0 for the standard image method; 
    #    default is 0 cm)
    #    - Tw is the length of the low-pass filter in [seconds] (default is 40
    #    samples, i.e. Tw=40/Fs)
    #    - Fc is the cut-off frequency of the fractional delay filter in [Hz]
    #    (default is Fc=0.9*(Fs/2))
    #    - c is the speed of sound in m/s (default is c=343)
    #    
    #    If you use this code, please cite De Sena et al. "On the modeling of 
    #    rectangular geometries in  room acoustic simulations." IEEE/ACM 
    #    Transactions on Audio, Speech and Language Processing (TASLP) 23.4 
    #    (2015): 774-786.
    # 
    #    Author: Paul Didier (paul.didier AT kuleuven DOT be).
    
    # Default input arguments
    if Tw is None:
        Tw = 40/Fs
    if Fc is None:
        Fc = 0.9*Fs/2
        
    # Ensure that <beta> is an array
    if not(isinstance(beta, list)):
        beta = np.ones((2,3))*beta
    
    # Input args. checks
    if mic_pos.ndim == 2:
        if mic_pos.shape[1] != 3:
            mic_pos = np.transpose(mic_pos)
    if source_pos.ndim == 2:
        if source_pos.shape[1] != 3:
            source_pos = np.transpose(source_pos)
    if room_dim.ndim == 2:
        if room_dim.shape[1] != 3:
            room_dim = np.transpose(room_dim)
    if rir_length <= 0:
        raise ValueError("Arg. <rir_length> must be strictly positive")
    if rand_dist < 0:
        raise ValueError("Arg. <rand_dist> must be positive")
    if Tw < 0:
        raise ValueError("Arg. <Tw> must be positive")
    if Fc < 0:
        raise ValueError("Arg. <Fc> must be positive")
    if c < 0:
        raise ValueError("Arg. <c> must be positive")
    
    # Number of microphone positions
    M = mic_pos.shape[0]
    
    npts = int(np.ceil(rir_length*Fs))
    
    h = np.zeros((npts, M))
    ps = perm([0,1], [0,1], [0,1])   # all binary numbers between 000 and 111
    orr = np.ceil(np.divide(rir_length*c,room_dim*2))
    rs = perm(range(-int(orr[0]),int(orr[0])+1),               range(-int(orr[1]),int(orr[1])+1),               range(-int(orr[2]),int(orr[2])+1))
    num_permutations = rs.shape[0]
    
    for ii in range(num_permutations):
        r = rs[ii,:]
        
        for jj in range(8):
            
            p = ps[jj,:]
            part1 = np.multiply(1 - 2*p, source_pos + 2*np.multiply(r, room_dim))
            part2 = rand_dist*(2*np.random.rand(1,3) - np.ones((1,3)))
            image_pos = part1 + part2
            
            for m in range(M):
                
                d = np.linalg.norm(image_pos - mic_pos[m,:])
                
                if np.round(d/c*Fs) >= 1 and np.round(d/c*Fs) <= npts:

                    am = np.multiply(np.power(beta[0,:], np.abs(r + p)), np.power(beta[1,:], np.abs(r)))
                    if Tw == 0:
                        n_integer = int(np.round(d/c*Fs))
                        h[n_integer, m] = h[n_integer, m] + np.prod(am)/(4*math.pi*d)
                    else:

                        n = np.array(range(int(np.maximum(np.ceil(Fs*(d/c - Tw/2)), 1)),                                 int(np.minimum(np.floor(Fs*(d/c + Tw/2)), npts - 1))))
                        t = n/Fs - d/c
                        s = np.multiply(1 + np.cos(2*math.pi*t/Tw), np.sinc(2*Fc*t)/2)
                        
                        h[n, m] = h[n, m] + s*np.prod(am)/(4*math.pi*d)    # Build RIR
    return h

def perm(a,b,c):
    s = [a,b,c]
    return np.array(list(itertools.product(*s)))
    
def export_data(data, fname):

    # Check if export folder exists
    mydir = os.path.dirname(fname)
    if not os.path.isdir(mydir):
        os.mkdir(mydir)   # if not, make directory
        print('Direction "%s" was created.' % mydir)

    # Data dimensionality
    n = data.ndim

    if n == 3:
        # Reshape 3D array into 2D array
        data_reshaped = data.reshape(data.shape[0], -1)

    # Save 2D matrix to text
    np.savetxt(fname, data_reshaped)

    print('Data exported to TXT format as 2D matrix: "%s"' % os.path.basename(fname))

    
def load_data(fname, original_dims):

    # Load 2D matrix from text
    loaded_arr = np.loadtxt(fname)

    if len([el for el in original_dims if el > 1]) == 3:
        loaded_arr = loaded_arr.reshape(
            loaded_arr.shape[0], loaded_arr.shape[1] // original_dims[2], original_dims[2])

    print('Data imported from TXT format')

    return loaded_arr


main()
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import random
import itertools
# from more_itertools import powerset
from rimPypack.rimPy import rimPy
from scipy.spatial.transform import Rotation as rot
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from plotting.threedim import plot_room

# 2-dimensional Acoustic Scenario (AS) generation script.
# --
# Generates a random 2-D layout of sensors, noise sources, and target sources.
# Exports the sets of RIRs corresponding to each possible combinations of 
# noise/target sources activations (e.g. source 1 active + noise 1 active, source 1 active + noise 2 active, ...).

class Node:
    def __init__(self, Mk, arraygeom, mic_sep):
        self.Mk = Mk
        self.array = arraygeom
        self.mic_sep = mic_sep


def main():

    nAS = 1                    # Number of AS to generate
    Fs = 16e3                   # Sampling frequency [samples/s]
    RIR_l = 2**12               # RIR length [samples]
    minRd = 3                   # Smallest room dimension possible [m]
    maxRd = 7                   # Largest room dimension possible [m]
    #
    Ns = 2                      # nr. of speech sources
    Nn = 3                      # nr. of noise sources
    #
    nNodes = 5                  # nr. of nodes
    Mk = 1
    node = Node(Mk, 'linear', 0.05)
    #
    minimum_distance_btw_objects = 0.75  # Minimum distance between objects (nodes, sources) [m] (if None: no constraint imposed)
    #
    T60fixed = 0            # Set to None to draw T60 randomly from uniform dist. between <T60min> and <T60max>
    T60max = 1.5*RIR_l/Fs   # Largest possible T60
    T60min = 0.4*RIR_l/Fs   # Smallest possible T60
    #
    plotit = 1           # If true, plots the AS
    expfolder = "C:\\Users\\u0137935\\source\\repos\\PaulESAT\\sounds-phd\\02_data\\01_acoustic_scenarios\\2D"

    counter = 0
    while counter < nAS:
        
        rd = np.random.uniform(low=minRd, high=maxRd, size=(3,))    # Generate random room dimensions
        if T60fixed is None:
            T60 = random.uniform(T60min, T60max)
        else:
            T60 = T60fixed
        # T60 = np.random.uniform(low=T60min, high=T60max, size=(1,)) # Generate random reverberation time
        V = np.prod(rd)                                   # Room volume
        S = 2*(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2])   # Total room surface area
        alpha = np.minimum(1, 0.161*V/(T60*S))                # Absorption coefficient of the walls
                
        # Call function
        h_ns, h_nn, rs, rn, r = genAS2D(rd,nNodes,node,Ns,Nn,alpha,RIR_l,Fs,plotit=plotit,mdbo=minimum_distance_btw_objects)

        expfolder += '\\J%iMk%i_Ns%i_Nn%i' % (nNodes,Mk,Ns,Nn)
        if not os.path.isdir(expfolder):   # check if subfolder exists
            os.mkdir(expfolder)   # if not, make directory
        # Loop over all possible combinations of active noise source and active target source
        combs_Ns = get_combinations(Ns)
        combs_Nn = get_combinations(Nn)
        nas = len(next(os.walk(expfolder))[2])   # count only files in export dir
        for combNs in combs_Ns:
            for combNn in combs_Nn:
                h_ns_curr = h_ns[:,:,combNs-1]   # <-1> necessary for starting indexing at 0
                h_nn_curr = h_nn[:,:,combNn-1]
                rs_curr = rs[combNs-1,:]
                rn_curr = rn[combNn-1,:]
                # Export
                fname =  "%s\\AS%i" % (expfolder,nas)
                if alpha == 1:
                    fname += '_anechoic'
                if nNodes == 1 and Mk > 1:
                    fname += '_array'
                fname += '_s%s' % (''.join([str(item) for item in combNs]))
                fname += '_n%s' % (''.join([str(item) for item in combNn]))
                fname += '.csv'
                #
                header = {'rd': pd.Series(np.squeeze(rd)), 'alpha': alpha, 'Fs': Fs,\
                    'nNodes': nNodes, 'd_intersensor': node.mic_sep}
                #  
                export_data(h_ns_curr, h_nn_curr, header, rs_curr, rn_curr, r, fname)

        counter += 1

    print('\n\nAll done.')

    return h_ns, h_nn
        

def genAS2D(rd,J,node,Ns,Nn,alpha,RIR_l,Fs,plotit=False,mdbo=None):
    # genAS2D -- Computes the RIRs in a rectangular cavity where sensors, speech
    # sources, and noise sources are present. 2-D case.
    # 
    # >>> Inputs:
    # -rd [3*1 (or 1*3) float array, m] - Room dimensions [x,y,z].
    # -J [int, -] - # of nodes.
    # -node [<Node> class object] - Node object containing # of sensors, array type, and inter-sensor distance.
    # -Ns [int, -] - # of desired sources.
    # -Nn [int, -] - # of noise sources.
    # -alpha [float, -] - Walls absorption coefficient (norm. inc.).
    # -RIR_l [int, #samples] - Length of RIRs to produce.
    # -Fs [float, Hz] - Sampling frequency.
    # -plotit [bool] - If true, plots the AS.
    # -mdbo [float, m] - Minimum distance between objects (nodes, sources). If None, no constraint imposed.
    #
    # >>> Outputs:
    # -h_sn [RIR_l*J*Ns (complex) float 3-D array, -] - RIRs between desired sources and sensors.  
    # -h_nn [RIR_l*J*Nn (complex) float 3-D array, -] - RIRs between noise sources and sensors.  

    # (c) Paul Didier - 29-Oct-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------
    
    # Random element positioning in 3-D space
    if mdbo is None:
        rs = np.multiply(np.random.rand(Ns,3),rd)   # speech sources positions
        r  = np.multiply(np.random.rand(J,3),rd)    # nodes positions
        rn = np.multiply(np.random.rand(Nn,3),rd)   # noise sources positions
    else:
        r = ensure_min_dist(J,np.array([]),mdbo,rd,z=rd[-1]/2)
        rs = ensure_min_dist(Ns,r,mdbo,rd,z=rd[-1]/2)
        rn = ensure_min_dist(Nn,np.concatenate((r,rs),axis=0),mdbo,rd,z=rd[-1]/2)

    # Generate sensor arrays (or set nodes as sensors if Mk = 1)
    M = J*node.Mk
    r_sensors = np.zeros((M,3))
    for ii in range(J):
        r_sensors[ii*node.Mk:(ii+1)*node.Mk,:] = generate_array_pos(r[ii,:], node.Mk, node.array, node.mic_sep)

    # If asked, show AS on plot
    if plotit:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_room(ax, rd)
        ax.scatter(r_sensors[:,0],r_sensors[:,1],r_sensors[:,2], c='green')
        ax.scatter(rs[:,0],rs[:,1],rs[:,2], c='blue')
        ax.scatter(rn[:,0],rn[:,1],rn[:,2], c='red')
        ax.grid()
        ax.view_init(90,0)
        plt.show()

    # Walls reflection coefficient  
    R = -1*np.sqrt(1 - alpha)   
    
    # Compute RIRs from speech source to sensors
    h_sn = np.zeros((RIR_l, M, Ns))
    for ii in range(Ns):
        print('Computing RIRs from speech source #%i at %i sensors' % (ii+1, M))
        h_sn[:,:,ii] = rimPy(r_sensors, rs[ii,:], rd, R, RIR_l/Fs, Fs)
    
    # Compute RIRs from noise source to sensors
    h_nn = np.zeros((RIR_l, M, Nn))
    for ii in range(Nn):
        print('Computing RIRs from noise source #%i at %i sensors' % (ii+1, M))
        h_nn[:,:,ii] = rimPy(r_sensors, rn[ii,:], rd, R, RIR_l/Fs, Fs)
    
    return h_sn, h_nn, rs, rn, r_sensors


def ensure_min_dist(N,to_be_avoided,mdbo,rd,z=None):
    # Returns <N> random positions within the <rd>-dimensioned room,
    # ensuring a minimum distance <mdbo> between all points
    # contained in <to_be_avoided>.
    r = np.zeros((N,3))
    counter = 0
    counter_attempts = 0
    while counter < N:
        # print('Attempt #%i (%i/%i points determined)' % (counter_attempts+1, counter, N))
        r_curr = np.random.rand(3) * (rd - mdbo) + mdbo/2
        if z is not None:
            r_curr[-1] = z      # impose z-coordinate if given
        flag_ok = True
        for jj in range(to_be_avoided.shape[0]):
            if np.sqrt(np.sum((to_be_avoided[jj,:] - r_curr)**2)) < mdbo:
                flag_ok = False
                break
        if flag_ok:
            r[counter,:] = r_curr
            counter += 1
            if len(to_be_avoided) > 0:
                to_be_avoided = np.concatenate((to_be_avoided, r_curr[np.newaxis,:]))
            else:
                to_be_avoided = r_curr
                to_be_avoided = to_be_avoided[np.newaxis,:]
        counter_attempts += 1
        if counter_attempts > 20e3:
            raise ValueError('The distance-between-objects constraint is too restrictive, needs relaxation.')
    return r

  
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
            columns=['Sensor %i' % (idx+1) for idx in range(h_sn.shape[1])],\
            index=['h_sn %i' % (ii+1) for idx in range(h_sn.shape[0])])

        # Concatenate to global dataframe
        data_df = pd.concat([data_df,data_df_curr])

    # Noise-to-node RIRs
    for ii in range(h_nn.shape[-1]):
        # Build dataframe for current matrix slice
        data_df_curr = pd.DataFrame(np.squeeze(h_nn[:,:,ii]),\
            columns=['Sensor %i' % (idx+1) for idx in range(h_nn.shape[1])],\
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
                index=['Sensor %i' % (idx+1) for idx in range(r.shape[0])],\
                columns=['x','y','z'])

    # Concatenate header + coordinates with rest of data
    big_df = pd.concat([header_df,rs_df,rn_df,r_df,data_df])
    big_df.to_csv(fname)

    print('Data exported to CSV: "%s"' % os.path.basename(fname))
    print('Find it in folder: %s' % str(Path(fname).parent))


def generate_array_pos(r, Mk, array_type, min_d, force2D=False):
    # Define node positions based on node position, number of nodes, and array type

    if array_type == 'linear':
        # 1D local geometry
        x = np.linspace(start=0, stop=Mk*min_d, num=Mk)
        # Center
        x -= np.mean(x)
        # Make 3D
        r_sensors = np.zeros((Mk,3))
        r_sensors[:,0] = x
        
        # Rotate in 3D through randomized rotation vector 
        rotvec = np.random.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        r_sensors_rot = np.zeros_like(r_sensors)
        for ii in range(Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            r_sensors_rot[ii,:] = myrot.apply(r_sensors[ii,:]) + r
    else:
        raise ValueError('No sensor array geometry defined for array type "%s"' % array_type)

    return r_sensors_rot


def get_fixed_values():

    rd = np.array([6,5,4])
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

def get_combinations(N):
    # Returns a list of all possibles (min. 1-long, max. N-long) 
    # combinations of numbers from 1 to N.
    indices = np.arange(1,N+1)
    combs = []
    for ii in indices:
        oc = itertools.combinations(indices, ii)
        for c in oc:
            combs.append(np.array(c))
    return np.array(combs)


if __name__ == '__main__':
    sys.exit(main())
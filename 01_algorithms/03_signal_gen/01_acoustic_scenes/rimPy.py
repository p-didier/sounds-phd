import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt

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
    if isinstance(mic_pos, list):
        mic_pos = np.array(mic_pos)
    if isinstance(source_pos, list):
        source_pos = np.array(source_pos)
    if isinstance(room_dim, list):
        room_dim = np.array(room_dim)
    
    if mic_pos.ndim == 2:
        if mic_pos.shape[1] != 3:
            mic_pos = np.transpose(mic_pos)
    elif len(mic_pos) < 3:
        raise ValueError("Arg. <mic_pos> must have at 3 elements per row")
    if len(source_pos) < 3:
        raise ValueError("Arg. <source_pos> must have 3 elements")
    if len(room_dim) < 3:
        raise ValueError("Arg. <room_dim> must have 3 elements")
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

    # Check that room dimensions are not too small
    if any(np.linalg.norm(mic_pos,None,axis=1) > np.linalg.norm(room_dim)):
        raise ValueError("Some microphones are located outside the room")
    if np.linalg.norm(source_pos) > np.linalg.norm(room_dim):
        raise ValueError("Some sources are located outside the room")
    
    npts = int(np.ceil(rir_length*Fs))
    
    h = np.zeros((npts, M))
    ps = perm([0,1], [0,1], [0,1])   # all binary numbers between 000 and 111
    orr = np.ceil(np.divide(rir_length*c,room_dim*2))
    rs = perm(range(-int(orr[0]),int(orr[0])+1), \
              range(-int(orr[1]),int(orr[1])+1), \
              range(-int(orr[2]),int(orr[2])+1))
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

                        n = np.array(range(int(np.maximum(np.ceil(Fs*(d/c - Tw/2)), 1)),\
                                 int(np.minimum(np.floor(Fs*(d/c + Tw/2)), npts - 1))))
                        t = n/Fs - d/c
                        s = np.multiply(1 + np.cos(2*math.pi*t/Tw), np.sinc(2*Fc*t)/2)
                        
                        h[n, m] = h[n, m] + s*np.prod(am)/(4*math.pi*d)    # Build RIR
    return h

def perm(a,b,c):
    s = [a,b,c]
    return np.array(list(itertools.product(*s)))
    
def main():
    
    alpha = 0.5
    mic_pos = [[0.1,0.1,0.1],]
    source_pos = [1,2,3]
    room_dim = [5,6,7]
    rir_length = 2**11
    Fs = 16e3
    
    beta = -np.sqrt(1 - alpha)
    
    h = rimPy(mic_pos, source_pos, room_dim, beta, rir_length/Fs, Fs, rand_dist=0, Tw=None, Fc=None, c=343)
        
    fig, ax = plt.subplots()
    ax.plot(h)
    plt.show()


# -- RUN
main()
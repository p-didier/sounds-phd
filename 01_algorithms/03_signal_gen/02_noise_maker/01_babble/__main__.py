import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\MWFpack')))
from sig_gen import load_speech

class BabbleParams:
    def __init__(self, T, N, db, room, Vb, rd):
        self.T = T      # Signal duration
        self.N = N      # nr. of babblers
        self.db = db    # database for speech sigs.
        self.room = room    # room type
        self.Vb = Vb        # room volume bounds
        self.rd = rd        # room dims.
        
    def bprint(self):
        print('Babble parameters: \nGenerating a %.1f s noise' % (self.T))
        if self.room == 'random':
            substr = 'volume bounds: %.1f m^3 <= V <= %.1f m^3' % (self.Vb[0],self.Vb[1])
        elif self.room == 'specific':
            substr = 'dimensions: %.1f x %.1f x %.1f (V = %.1f m^3)' % (self.rd[0],self.rd[1],self.rd[2],np.prod(self.rd))
        print('%i babblers in a %s room with %s' % (self.N, self.room, substr))
        return None

def load_babbler(database, duration):

    babbler, Fs = load_speech(database)
    babblerout = np.copy(babbler)
    while len(babblerout)/Fs < duration:
        babblerout = np.concatenate((babblerout, babbler))
        if len(babblerout)/Fs > duration:
            babblerout = babblerout[:int(duration*Fs)]

    return babblerout

def generate_babble(bp=BabbleParams):

    # Get RIR
    ...

    for ii in range(bp.N):
        babbler = load_babbler(bp.db, bp.T)
        # Convolve with RIR
        if ii == 1:
            mybabble = babbler
        else:
            mybabble += babbler

    return None

def main():
    # Parameters
    T = 15                  # Signal duration [s]
    Ntalkers = 50           # Number of talkers in babble
    dataBase = 'libri'      # Reference for database where to fetch dry speech signals 
    room = 'random'         # Reference for type of room to use to generate RIRs 
                            #   -- if == 'random': generates a randomized shoebox-room within the volume bounds <Vbounds>
                            #   -- if == 'specific': generates a specific shoebox-room with the dimensions <rd>
    Vbounds = [70, 150]     # Room volume bounds [low, high] [m] -- only used if <room> == 'random'
    rd_specific = [6,5,7]   # Specific room dimensions [x,y,z] [m] -- only used if <room> == 'specific'

    # Generate babble object
    bp = BabbleParams(T, Ntalkers, dataBase, room, Vbounds, rd_specific)
    bp.bprint()     # inform user

    generate_babble(bp)

if __name__== "__main__" :
    main()
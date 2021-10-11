import numpy as np
import sys, os
import scipy.signal
from pathlib import Path
from sklearn import preprocessing
import scipy.io.wavfile
# Custom imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\MWFpack')))
from sig_gen import load_speech
currdir = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(currdir.parent.parent, '01_acoustic_scenes\\rimPypack')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from rimPy import rimPy
from playsounds.playsounds import playthis


class BabbleParams:
    def __init__(self, T, N, db, room, Vb, rd, T60):
        self.T = T      # Signal duration
        self.N = N      # nr. of babblers
        self.db = db    # database for speech sigs.
        self.room = room    # room type
        self.Vb = Vb        # room volume bounds
        self.rd = rd        # room dims.
        self.T60 = T60      # reverb. time
        
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

    babblerout = preprocessing.scale(babblerout)   # normalize

    return babblerout,Fs


def generate_babble(bp=BabbleParams):

    # ---------- Room ----------
    # Dimensions
    if bp.room == 'random':
        rd = np.random.uniform(np.amin(bp.Vb)**(1/3), np.amax(bp.Vb)**(1/3), size=(3,))
        print('Room generated: V = %.1f x %.1f x %.1f = %.1f m^3' % (rd[0], rd[1], rd[2], np.prod(rd)))
    elif bp.room == 'specific':
        rd = bp.rd
    # Define geometry
    mic_pos = (rd / 2).T
    mic_pos = mic_pos[:, np.newaxis]
    babbler_pos = np.random.uniform(0,1,size=(bp.N,3)) * rd
    # Define reflection coefficient
    Vol = np.prod(rd)                                   # Room volume
    Surf = 2*(rd[0]*rd[1] + rd[0]*rd[2] + rd[1]*rd[2])   # Total room surface area
    alpha = np.minimum(1, 0.161*Vol/(bp.T60*Surf))                # Absorption coefficient of the walls
    RefCoeff = -1*np.sqrt(1 - alpha)
    # Define RIR length
    rir_dur = np.amax([0.5 * bp.T60, 0.25])

    print('Generating babble...')
    for ii in range(bp.N):
        print('... Babbler #%i/%i ...' % (ii+1,bp.N))
        babbler,Fs = load_babbler(bp.db, bp.T)
        babbler = babbler[:, np.newaxis]
        # Get RIR
        rir = rimPy(mic_pos, babbler_pos[ii,:], rd, RefCoeff, rir_dur, Fs)
        if ii == 0:
            print('RIR length: %i samples' % (rir_dur*Fs))
        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # ax.plot(rir)
        # Convolve with RIR 
        babbler = scipy.signal.fftconvolve(babbler, rir)
        babbler = babbler[:int(bp.T*Fs), :]
        # Build babble
        if ii == 0:
            mybabble = babbler
        else:
            mybabble += babbler

    # Normalize
    mybabble = preprocessing.scale(mybabble)   # normalize

    if 0:
        playthis(mybabble, Fs)

    return mybabble, Fs


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
    T60 = 0                 # Reverberation time in room [s]
    # Export
    exportflag = True       # Only exports signal as .wav if <exportflag>.
    exportfolder = '%s\\02_data\\02_signals\\02_noise\\babble' % os.getcwd()
    exportfname = 'babble1'

    # Generate babble object
    bp = BabbleParams(T, Ntalkers, dataBase, room, Vbounds, rd_specific, T60)
    bp.bprint()     # inform user

    mybabble, Fs = generate_babble(bp)

    # Export
    if exportflag:
        if '.' in exportfname:
            exportfname = exportfname[:-4]
        if not os.path.isdir(exportfolder):
            os.mkdir(exportfolder)
        scipy.io.wavfile.write('%s\\%s.wav' % (exportfolder,exportfname),Fs,mybabble)

    return 0

if __name__== "__main__" :
    sys.exit(main())
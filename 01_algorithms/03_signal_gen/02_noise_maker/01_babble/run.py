import numpy as np
import sys, os
import scipy.signal
from pathlib import Path, PurePath
from sklearn import preprocessing
import scipy.io.wavfile
import soundfile as sf
# Custom imports
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("01_centralized" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/01_centralized/01_MWF_based/01_GEVD_MWF\MWFpack')
from sig_gen import load_speech
if not any("rimPypack" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes/rimPypack')
from rimPy import rimPy
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
from playsounds.playsounds import playthis
## vvv Useful for debugging
import matplotlib.pyplot as plt


def main():
    # Parameters
    nBabbles = 10   # number of babbles to generate
    Ttot = 20               # Signal duration [s]
    Ntalkers = 20           # Number of talkers in babble
    dataBase = 'libri'      # Reference for database where to fetch dry speech signals 
    room = 'random'         # Reference for type of room to use to generate RIRs 
                            #   -- if == 'random': generates a randomized shoebox-room within the volume bounds <Vbounds>
                            #   -- if == 'specific': generates a specific shoebox-room with the dimensions <rd>
    Vbounds = [70, 150]     # Room volume bounds [low, high] [m] -- only used if <room> == 'random'
    rd_specific = [6,5,7]   # Specific room dimensions [x,y,z] [m] -- only used if <room> == 'specific'
    t60 = 0                 # Reverberation time in room [s]
    # Export
    exportflag = True       # Only exports signal as .wav if <exportflag>.
    exportfolder = f'{pathToRoot}/02_data/00_raw_signals/02_noise/babble'

    # Prepare generation
    bp = BabbleParams(
        Ttot, Ntalkers, dataBase,
        room, Vbounds, rd_specific, t60
    )
    bp.bprint()     # inform user
    for ii in range(nBabbles):
        print(f'----- BABBLE {ii+1}/{nBabbles} -----')
        # Generate a babble
        mybabble, Fs = generate_babble(bp)
        # Export
        if exportflag:
            if not Path(exportfolder).is_dir():
                Path(exportfolder).mkdir()
            sf.write(_get_export_name(exportfolder), mybabble, Fs)

    return 0


def _get_export_name(folder):
    """Defines .wav file export name."""
    if folder[-1] in ['/', '\\']:
        folder = folder[:-1]
    nFiles = len([f for f in os.listdir(folder) if f[-4:]=='.wav'])
    return f'{folder}/babble{nFiles + 1}.wav'


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
    """Load a babbler (one talker)"""
    babbler, Fs = load_speech(database)
    babblerout = np.copy(babbler)
    while len(babblerout)/Fs < duration:
        babblerout = np.concatenate((babblerout, babbler))
        if len(babblerout)/Fs > duration:
            babblerout = babblerout[:int(duration*Fs)]

    babblerout = preprocessing.scale(babblerout)   # normalize

    return babblerout,Fs


def generate_babble(bp: BabbleParams):
    """Generate babble noise from parameters"""
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
        babbler,Fs = load_babbler(bp.db, bp.T * 1.5)
        # ^^^ `*1.5` to avoid problems of too short file when truncating
        babbler = babbler[:, np.newaxis]
        # Get RIR
        rir = rimPy(mic_pos, babbler_pos[ii,:], rd, RefCoeff, rir_dur, Fs)
        if ii == 0:
            print('RIR length: %i samples' % (rir_dur*Fs))
        # Convolve with RIR 
        babbler = scipy.signal.fftconvolve(babbler, rir)
        if babbler.ndim == 1:
            babbler = babbler[:, np.newaxis]
        # Detect initial silence
        idxStart = np.argmax(np.abs(babbler) ** 2\
            >= np.amax(np.abs(babbler) ** 2) / 10)
        idxStart -= int(0.01 * Fs)  # get rid of potential clicks
        # Truncate
        uttDur = 1  # [s] duration of a typical utterance
        uttShift = int(uttDur * Fs)
        babbler = babbler[idxStart:int(bp.T*Fs) + idxStart + uttShift, :]
        # Build babble
        if ii == 0:
            mybabble = babbler[uttShift:]
        else:
            mybabble += babbler[uttShift:]

    # Normalize
    mybabble = mybabble / np.amax(np.abs(mybabble)) * 0.95  # normalize

    if 0:
        playthis(mybabble, Fs)

    return mybabble, Fs


if __name__== "__main__" :
    sys.exit(main())
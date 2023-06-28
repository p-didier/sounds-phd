# Purpose of script:
# xxx
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

SOUND_FILE_PATH = 'C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/danse/out/20230628_tests/forSOUNDSSC11/danse/12_as8_60s/wav/desired_N1_Sref1.wav'
EXPORT_FOLDER = '96_quick_plots/quick_sound_file_plot/figs'

def main():
    """Main function (called by default when running script)."""
    
    # Read sound file
    data, fs = sf.read(SOUND_FILE_PATH)

    # Plot
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(6.5, 2)
    axes.plot(data)
    axes.set_xlabel('Sample index')
    axes.set_ylabel('Amplitude')
    axes.grid(True)
    fig.tight_layout()
    # Export
    fig.savefig(f'{EXPORT_FOLDER}/{Path(SOUND_FILE_PATH).stem}_plot.png', dpi=300)
    fig.savefig(f'{EXPORT_FOLDER}/{Path(SOUND_FILE_PATH).stem}_plot.pdf', dpi=300)

if __name__ == '__main__':
    sys.exit(main())
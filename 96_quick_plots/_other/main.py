import sys

import utils.sroplot1
from pathlib import Path
import matplotlib.pyplot as plt

# Matplotlib settings
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 11})

def main():

    utils.sroplot1.plotsro(
        exportpath=f'{Path(__file__).parent}/figs/202209_icassp23/sros1',
        style='bw'
        )
    # utils.sroplot1.plotsto(exportpath=f'{Path(__file__).parent}/figs/stos1.pdf')

    return 0

    
# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
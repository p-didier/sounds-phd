import sys

import utils.sroplot1
from pathlib import Path

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
# Purpose of script:
# Transfer specific figures from a Python "results/"-type folder to a 
# desired LaTeX "graphics/"-type folder.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import shutil
from pathlib import Path, WindowsPath

PATH_TO_RESULTS_FOLDER = 'danse/out/20230508_tests/tigevddanse'
DESTINATION_FOLDER = 'C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/08_secondments/UOL/01_meetings/20230516_update_chinaev_enzner_withoutsimon/tex/graphics/tidanse'
FIGNAME = 'metrics.pdf'  # name of the figure to transfer

def main(
        destinationPath=DESTINATION_FOLDER,
        resultsPath=PATH_TO_RESULTS_FOLDER,
        figname=FIGNAME
    ):
    """Main function (called by default when running script)."""
    
    # Check that the destination folder exists
    if not Path(destinationPath).exists():
        raise ValueError(f'Destination folder "{destinationPath}" does not exist.')
    
    # Check that the results folder exists
    if not Path(resultsPath).exists():
        raise ValueError(f'Results folder "{resultsPath}" does not exist.')
    
    # Check for subfolders in the results folder
    subfolders = [x for x in Path(resultsPath).iterdir() if x.is_dir()]
    if len(subfolders) == 0:
        copy_fig(resultsPath, destinationPath, figname)
    else:
        # Recursively check for subfolders
        for subfolder in subfolders:
            copy_fig(resultsPath, destinationPath, figname)
            main(destinationPath, subfolder, figname)


def copy_fig(path: WindowsPath, destinationPath, figname):
    """Copy a figure from a results folder to a destination folder."""
    # Check if the results folder contains the figure
    if not Path(path, figname).exists():
        print(f'Figure "{figname}" not found in results folder "{path}", skipping.')
    else:
        # Copy the figure to the destination folder
        shutil.copy(
            Path(path, figname),
            Path(
                destinationPath,
                f'{Path(figname).stem}_from_{path.name}{Path(figname).suffix}'
            )
        )

if __name__ == '__main__':
    sys.exit(main())
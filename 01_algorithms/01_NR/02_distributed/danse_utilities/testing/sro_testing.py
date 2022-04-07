
from dataclasses import dataclass, field
import sys, time
from pathlib import Path, PurePath

import danse_main
from itertools import combinations
from danse_utilities.classes import ProgramSettings
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
if not any("01_acoustic_scenes" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/01_algorithms/03_signal_gen/01_acoustic_scenes')
from utilsASC.classes import AcousticScenario
# ------------------------

@dataclass
class DanseTestingParameters():
    ascBasePath: str = ''                   # Path to folder containing all acoustic scenarios to consider (only used if `specificAcousticScenario==''`).
    signalsPath: str = ''                   # Path to folder containing all sound fildes to include (only used if `specificDesiredSignalFiles==''` or `specificNoiseSignalFiles==''`).
    specificAcousticScenario: list[str] = field(default_factory=list)       # Path(s) to specific acoustic scenario(s). If not [''], `ascBasePath` is ignored.
    specificDesiredSignalFiles: list[str] = field(default_factory=list)     # Path(s) to specific desired signal file(s). If not [''], `signalsPath` is ignored for desired signals.
    specificNoiseSignalFiles: list[str] = field(default_factory=list)       # Path(s) to specific noise signal file(s). If not [''], `signalsPath` is ignored for noise signals.
    #
    sigDur: float = 1.      # signals duration [s]
    baseSNR: float = 0.     # # SNR between dry desired signals and dry noise [dB]
    nodeUpdating: str = 'simultaneous'  # node-updating strategy
    broadcastLength: int = 8    # length of broadcast chunk [samples]
    #
    possibleSROs: list[float] = field(default_factory=list)     # Possible SRO values [ppm]

    def __post_init__(self):
        # Check inputs variable type
        if not isinstance(self.specificAcousticScenario, list) and isinstance(self.specificAcousticScenario, str):
            self.specificAcousticScenario = [self.specificAcousticScenario]
        if not isinstance(self.specificDesiredSignalFiles, list) and isinstance(self.specificDesiredSignalFiles, str):
            self.specificDesiredSignalFiles = [self.specificDesiredSignalFiles]
        if not isinstance(self.specificNoiseSignalFiles, list) and isinstance(self.specificNoiseSignalFiles, str):
            self.specificNoiseSignalFiles = [self.specificNoiseSignalFiles]


def asc_path_selection(danseParams: DanseTestingParameters):
    """Acoustic scenario and sound files paths selection for DANSE tests.
    
    Parameters
    ----------
    danseParams : DanseTestingParameters object
        DANSE testing parameters. 

    Returns
    -------
    acousticScenarios : list of str
        Paths to acoustic scenarios to consider in further processing.
    speechFiles : list of str
        Paths to desired signal files to consider in further processing.
    noiseFiles : list of str
        Paths to noise signal files to consider in further processing.
    """

    # Select acoustic scenario(s)
    if danseParams.specificAcousticScenario == ['']:
        # List all acoustic scenarios folder paths
        acsSubDirs = list(Path(danseParams.ascBasePath).iterdir())
        acousticScenarios = []
        for ii in range(len(acsSubDirs)):
            acss = list(Path(acsSubDirs[ii]).iterdir())
            for jj in range(len(acss)):
                acousticScenarios.append(acss[jj])
    else:
        acousticScenarios = [Path(i) for i in danseParams.specificAcousticScenario]

    # Select desired signal files
    if danseParams.specificDesiredSignalFiles == ['']:
        print(f'Selecting all desired signal files contained in "{danseParams.signalsPath}/speech/"')
        speechFiles = [f for f in Path(f'{danseParams.signalsPath}/speech').glob('**/*') if f.is_file()]
    else:
        print(f'Selecting specified desired signal files')
        speechFiles = [Path(i) for i in danseParams.specificDesiredSignalFiles]
    
    # Select noise signal files
    if danseParams.specificNoiseSignalFiles == ['']:
        print(f'Selecting all noise signal files contained in "{danseParams.signalsPath}/noise/"')
        noiseFiles = [f for f in Path(f'{danseParams.signalsPath}/noise').glob('**/*') if f.is_file()]
    else:
        print(f'Selecting specified noise signal files')
        noiseFiles = [Path(i) for i in danseParams.specificNoiseSignalFiles]

    return acousticScenarios, speechFiles, noiseFiles


def build_experiment_parameters(danseParams: DanseTestingParameters, exportBasePath=''):
    """Builds `experiments` object for DANSE testing.
    
    Parameters
    ----------
    danseParams : DanseTestingParameters object
        DANSE testing parameters.
    exportBasePath : str
        Path to export folder for testing results.
    
    Returns
    -------
    experiments : list of dicts (ProgramSettings objects ; str)
        Experiment settings, one per acoustic scenario considered and per SROs combinations. 
    """

    # Get explicit path lists
    acousticScenarios, speechFiles, noiseFiles = asc_path_selection(danseParams)

    # Get all possible SRO combinations
    sros = []
    for ii in range(len(acousticScenarios)):
        asc = AcousticScenario().load(acousticScenarios[ii])
        # include all possible unique combinations of SRO values, given the number of nodes in the acoustic scenario
        # --> always including an SRO of 0 ppm (reference clock) at node 1.
        srosCurr = [list((0,) + i) for i in combinations(danseParams.possibleSROs, asc.numNodes - 1)]
        sros.append(srosCurr)   

    # Build experiments list
    experiments = []
    for ii in range(len(acousticScenarios)):
        for jj in range(len(sros[ii])):
            sets = ProgramSettings(
                    acousticScenarioPath=acousticScenarios[ii],
                    desiredSignalFile=speechFiles,
                    noiseSignalFile=noiseFiles,
                    signalDuration=danseParams.sigDur,
                    baseSNR=danseParams.baseSNR,
                    plotAcousticScenario=False,
                    VADwinLength=40e-3,
                    VADenergyFactor=4000,
                    performGEVD=1,
                    SROsppm=sros[ii][jj],
                    danseUpdating=danseParams.nodeUpdating,
                    broadcastLength=danseParams.broadcastLength,
                    computeLocalEstimate=False
                    )
            exportPath = f'{exportBasePath}/{acousticScenarios[ii].parent.name}/{acousticScenarios[ii].name}_SROs{sros[ii][jj]}'     # experiment export path
            experiments.append(dict([('sets', sets), ('path', exportPath)]))

    return experiments


def go(danseParams: DanseTestingParameters, exportBasePath=''):
    """Launch-function for DANSE tests with SROs.
    
    Parameters
    ----------
    danseParams : DanseTestingParameters object
        DANSE testing parameters.
    exportBasePath : str
        Path to export folder for testing results.
    """

    # Build experiments list
    experiments = build_experiment_parameters(danseParams, exportBasePath)
    print(f'Experiments parameters generated. Total: {len(experiments)} experiments.')

    for idxExp in range(len(experiments)):
        print(f'\nRunning experiment #{idxExp+1}/{len(experiments)}...\n')
        t0 = time.perf_counter()

        if not Path(experiments[idxExp]['path']).is_dir():
            danse_main.main(experiments[idxExp]['sets'], experiments[idxExp]['path'], showPlots=0, lightExport=True)
            print(f'\nExperiment #{idxExp+1}/{len(experiments)} ran in {round(time.perf_counter() - t0, 2)}s.\n')
        else:
            print(f'...NOT RUNNING "{Path(experiments[idxExp]["path"]).name}"...')

    return 0
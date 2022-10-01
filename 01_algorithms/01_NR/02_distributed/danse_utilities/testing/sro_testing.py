
from dataclasses import dataclass, field
import sys, time
from pathlib import Path, PurePath
from typing import Union
import numpy as np
import danse_main
import itertools
from danse_utilities.classes import ProgramSettings, SamplingRateOffsets, PrintoutsParameters
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
import class_methods.dataclass_methods as met
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
class TestSROs():
    type : str = 'list'     # type of SROs distribution to use.
                            # - 'list': pick specific SRO value from a given list (`listedSROs` field)
                            # - 'g': pick from a Gaussian distribution dictated by the `gaussianParams` field
    listedSROs : list[float] = field(default_factory=list)
    gaussianParams : list[float] = field(default_factory=list)  # Gaussian distribution parameters (only used if `type == 'g'`)
                                                                # - elements: [mean SRO, standard deviation]

@dataclass
class DanseTestingParameters():
    writeOver: bool = False                         # if True, the DANSE testing script will write over existing data if filenames conflict
    #
    ascBasePath: str = ''                   # Path to folder containing all acoustic scenarios to consider (only used if `specificAcousticScenario==''`).
    signalsPath: str = ''                   # Path to folder containing all sound fildes to include (only used if `specificDesiredSignalFiles==''` or `specificNoiseSignalFiles==''`).
    specificAcousticScenario: list[str] = field(default_factory=list)       # Path(s) to specific acoustic scenario(s). If not [''], `ascBasePath` is ignored.
    specificDesiredSignalFiles: list[str] = field(default_factory=list)     # Path(s) to specific desired signal file(s). If not [''], `signalsPath` is ignored for desired signals.
    specificNoiseSignalFiles: list[str] = field(default_factory=list)       # Path(s) to specific noise signal file(s). If not [''], `signalsPath` is ignored for noise signals.
    #
    fs: float = 16000.                      # based sampling frequency [Hz]
    sigDur: float = 1.                      # signals duration [s]
    baseSNR: float = 0.                     # SNR between dry desired signals and dry noise [dB]
    nodeUpdating: str = 'simultaneous'      # node-updating strategy
    broadcastLength: int = 8                # length of broadcast chunk [samples]
    timeBtwExternalFiltUpdates: float = 0   # [s] minimum time between 2 consecutive external filter update (i.e. filters that are used for broadcasting)
    broadcastScheme: str = 'wholeChunk'     # inter-node data broadcasting domain:
                                            # -- 'wholeChunk': broadcast whole chunks of compressed signals
                                            # -- 'samplePerSample': linear-convolution approximation of WOLA compression process, broadcast 1 sample at a time.
    performGEVD : bool = True               # if True, perform GEVD, else, regular DANSE
    #
    possibleSROs: list[float] = field(default_factory=list)     # Possible SRO values [ppm]
    SROdistribution: list[float] = field(default_factory=list)     # Possible SRO values [ppm]
    asynchronicity: SamplingRateOffsets = SamplingRateOffsets()
    printouts: PrintoutsParameters = PrintoutsParameters()
    #
    computeLocalEstimate: bool = False      # if True, compute also node-specific local estimate of desired signal

    def __post_init__(self):
        # Check inputs variable type
        if not isinstance(self.specificAcousticScenario, list) and isinstance(self.specificAcousticScenario, str):
            self.specificAcousticScenario = [self.specificAcousticScenario]
        if not isinstance(self.specificDesiredSignalFiles, list) and isinstance(self.specificDesiredSignalFiles, str):
            self.specificDesiredSignalFiles = [self.specificDesiredSignalFiles]
        if not isinstance(self.specificNoiseSignalFiles, list) and isinstance(self.specificNoiseSignalFiles, str):
            self.specificNoiseSignalFiles = [self.specificNoiseSignalFiles]
        # Ensure that SRO = 0 ppm is an option
        if 0 not in self.possibleSROs:
            self.possibleSROs = [0] + self.possibleSROs
        # Warnings
        if self.writeOver:
            inp = input(f'`danseParams.writeOver` is True -- Do you confirm you want to write over existing exports? [y/[n]]  ')
            if not inp in ['y', 'Y']:
                print('Setting `danseParams.writeOver` to False. Not overwriting.')
                self.writeOver = False

    def load(self, foldername: str, silent=False):
        return met.load(self, foldername, silent)

    def save(self, foldername: str):
        # Save most important parameters as quickly-readable .txt file
        met.save_as_txt(self, foldername)
        # Save data as archive
        met.save(self, foldername)



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
        srosCurr = [list((0,) + p) for p in itertools.product(danseParams.possibleSROs, repeat=asc.numNodes - 1)]
        sros.append(srosCurr)   

    # Build experiments list
    experiments = []
    for ii in range(len(acousticScenarios)):
        for jj in range(len(sros[ii])):

            compSROs = danseParams.asynchronicity.compensateSROs
            if all(v == 0 for v in sros[ii][jj]):  # <-- avoid automatic warning message at `ProgramSettings` object initialization
                compSROs = False

            # Interpret broadcast scheme entry
            if danseParams.broadcastScheme == 'wholeChunk':
                BCdomain = 'wholeChunk_td'
                ps = ProgramSettings()
                BClength = ps.DFTsize // 2
            elif danseParams.broadcastScheme == 'samplePerSample':
                BCdomain = 'fewSamples_td'
                BClength = 1

            settings = ProgramSettings(
                    samplingFrequency=danseParams.fs,
                    acousticScenarioPath=acousticScenarios[ii],
                    desiredSignalFile=speechFiles,
                    noiseSignalFile=noiseFiles,
                    #
                    signalDuration=danseParams.sigDur,
                    baseSNR=danseParams.baseSNR,
                    performGEVD=danseParams.performGEVD,
                    #
                    danseUpdating=danseParams.nodeUpdating,
                    broadcastDomain=BCdomain,
                    broadcastLength=BClength,
                    computeLocalEstimate=danseParams.computeLocalEstimate,
                    timeBtwExternalFiltUpdates=danseParams.timeBtwExternalFiltUpdates,
                    expAvg50PercentTime=2.,
                    #
                    asynchronicity=SamplingRateOffsets(
                        SROsppm=sros[ii][jj],
                        compensateSROs=compSROs,
                        estimateSROs=danseParams.asynchronicity.estimateSROs,
                        plotResult=danseParams.asynchronicity.plotResult,
                        cohDriftMethod=danseParams.asynchronicity.cohDriftMethod,
                    ),
                    printouts=danseParams.printouts,
                    )
            # Build export file path
            exportPath = f'{exportBasePath}/{acousticScenarios[ii].parent.name}/{acousticScenarios[ii].name}_SROs{sros[ii][jj]}'     # experiment export path
            if (np.array(settings.asynchronicity.SROsppm) != 0).any():
                if settings.asynchronicity.compensateSROs:
                    exportPath += f'_comp{settings.asynchronicity.estimateSROs}'
                else:
                    exportPath += '_nocomp'
            experiments.append(dict([('settings', settings), ('path', exportPath)]))

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
    print(f'Experiments parameters generated. Total: {len(experiments)} experiments to be run.')

    for idxExp in range(len(experiments)):
        print(f'\nRunning experiment #{idxExp+1}/{len(experiments)}...\n')
        t0 = time.perf_counter()

        if not Path(experiments[idxExp]['path']).is_dir() or danseParams.writeOver:
            danse_main.main(experiments[idxExp]['settings'], experiments[idxExp]['path'], showPlots=0, lightExport=True)
            print(f'\nExperiment #{idxExp+1}/{len(experiments)} ran in {round(time.perf_counter() - t0, 2)}s.\n')
        else:
            # If export folder already existing...
            print(f'...NOT RUNNING "{Path(experiments[idxExp]["path"]).name}"...')

    return 0
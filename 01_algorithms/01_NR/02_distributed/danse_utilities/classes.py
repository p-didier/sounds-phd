from dataclasses import dataclass
import sys, os
from pathlib import Path, PurePath
sys.path.append(os.path.join(os.path.expanduser('~'), 'py/sounds-phd/_general_fcts'))
from class_methods import dataclass_methods 

@dataclass
class ProgramSettings(object):
    """Class for keeping track of global simulation settings"""
    # Signal generation parameters
    acousticScenarioPath: str               # path to acoustic scenario CSV file to be used
    signalDuration: float                   # total signal duration [s]
    desiredSignalFile: list                 # list of paths to desired signal file(s)
    noiseSignalFile: list                   # list of paths to noise signal file(s)
    baseSNR: int                            # SNR between dry desired signals and dry noise
    # DANSE parameters
    weightsInitialization: str = 'zeros'    # type of DANSE filter weights initialization ("random", "zeros", "ones", ...)
    # Other parameters
    plotAcousticScenario: bool = False      # if true, plot visualization of acoustic scenario. 
    acScenarioPlotExportPath: str = ''      # path to directory where to export the acoustic scenario plot

    def __post_init__(self) -> None:
        # Checks on class attributes
        if self.acousticScenarioPath[-4:] != '.csv':
            self.acousticScenarioPath += '.csv'
            print('Automatically appended ".csv" to string setting "acousticScenarioPath".')
        return self

    def __repr__(self):
        string = f"""--------- Program settings ---------
        > Acoustic scenario from: '{PurePath(self.acousticScenarioPath).name}'
        > {self.signalDuration} seconds signals using desired file(s):
        {[PurePath(f).name for f in self.desiredSignalFile]}
        > and noise signal file(s):
        {[PurePath(f).name for f in self.noiseSignalFile]}
        > with a base SNR btw. dry signals of {self.baseSNR} dB.
        """
        return string

    @classmethod
    def load(cls, filename: str):
        return dataclass_methods.load(cls, filename)

    def save(self, filename: str):
        dataclass_methods.save(self, filename)
        


@dataclass
class Results:
    # """Class for keeping track of global simulation settings"""
    # data: 
    a = 1
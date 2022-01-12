from dataclasses import dataclass
import sys, os
sys.path.append(os.path.join(os.path.expanduser('~'), 'py/sounds-phd/_general_fcts'))
from class_methods import dataclass_methods 

@dataclass
class ProgramSettings(object):
    """Class for keeping track of global simulation settings"""
    samplingFrequency: int = 16e3
    signalDuration: float = 3

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
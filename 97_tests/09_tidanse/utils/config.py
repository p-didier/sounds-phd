import ast
import yaml
import numpy as np
from typing import List
from dataclasses import dataclass, field, is_dataclass

@dataclass
class SignalConfig:
    desiredSignalType: str = 'noise'  # noise, speech, noise+pauses
    fs: int = 16000
    nSamplesBatch: int = 1000
    # vvv (used iff `desiredSignalType == noise+pauses`) vvv
    pauseLength: int = 1000  # in samples
    pauseSpacing: int = 1000  # in samples
    samplesSinceLastPause: int = 0 
    sampleIdx: int = 0
    # ^^^ (used iff `desiredSignalType == noise+pauses`) ^^^

@dataclass
class Configuration:
    mcRuns: int = 1
    originalSeed: int = 0
    refSensorIdx: int = 0
    sigConfig: SignalConfig = SignalConfig()

    K: int = 15
    Mk: int = 5

    # Online processing
    B: int = 500
    overlapB: int = 0
    beta: float = 0.98
    betaRnn: float = None

    # TI-DANSE eta normalization
    gamma: float = 0.0
    maxBetaNf: float = 0.6

    nStft: int = 1024
    nNoiseSources: int = 1
    eps: float = 1e-4
    maxIter: int = 2000
    snr: int = 10
    snSnr: int = 20

    algos: List[str] = field(default_factory=lambda: ['danse', 'ti-danse'])
    mode: str = 'batch'

    gevd: bool = False

    nodeUpdating: str = 'seq'
    nIterBetweenUpdates: int = 0
    # Tests / debug
    normGkEvery: int = 1
    # Plot booleans
    plotOnlyCost: bool = False
    exportFolder: str = './figs'

    def __post_init__(self):
        np.random.seed(self.originalSeed)  # set RNG seed
        self.rngStateOriginal = np.random.get_state()  # save RNG state
        # Check for online VAD
        if self.mode == 'online' and self.sigConfig.desiredSignalType == 'noise+pauses':
            if self.sigConfig.pauseLength <= self.B or\
                self.sigConfig.pauseSpacing <= self.B:
                raise ValueError("['noise+pauses' desired signal] Pause length and spacing must be larger than `B`.")
        if self.mode == 'batch' and self.sigConfig.desiredSignalType == 'noise+pauses':
            print("Warning: `desiredSignalType == 'noise+pauses'` is not supported in batch mode. Switching to `desiredSignalType == 'noise'`.")
            self.sigConfig.desiredSignalType = 'noise'
        

    def to_string(self):
        """Converts the configuration to a TXT-writable
        nicely formatted string."""
        config_str = "Configuration:\n"
        config_str += "+----------------------+----------------------+\n"
        for key, value in self.__dict__.items():
            config_str += f"| {key.ljust(20)} | {str(value).ljust(20)} |\n"
        config_str += "+----------------------+----------------------+\n"
        return config_str
    
    def to_yaml(self, pathToYaml):
        """Export configuration to yaml file."""
        with open(pathToYaml, 'w') as f:
            yaml.dump(self.__dict__, f)

    def __repr__(self):
        """Return string representation of configuration."""
        return self.to_string()

    def from_yaml(self, pathToYaml: str = ''):
        """Read configuration from yaml file."""
        if pathToYaml == '':
            pathToYaml = self.yamlFile
        self.__dict__ = load_from_yaml(pathToYaml, self).__dict__
        self.__post_init__()
        return self


def load_from_yaml(path, myDataclass):
    """Loads data from a YAML file into a dataclass.
    
    Parameters
    ----------
    path : str
        The path to the YAML file to be loaded.
    myDataclass : instance of a dataclass
        The dataclass to load the data into.

    Returns
    -------
    myDataclass : instance of a dataclass
        The dataclass with the data loaded into it.
    """

    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    def _interpret_lists(d):
        """Interprets lists in the YAML file as lists of floats, not strings"""
        for key in d:
            if type(d[key]) is str and len(d[key]) >= 2:
                if d[key][0] == '[' and d[key][-1] == ']':
                    d[key] = ast.literal_eval(d[key])  # Convert string to list
                    # Use of `literal_eval` hinted at by https://stackoverflow.com/a/1894296
            elif type(d[key]) is dict:
                d[key] = _interpret_lists(d[key])
        return d

    # Detect lists
    d = _interpret_lists(d)

    def _deal_with_arrays(d):
        """Transforms lists that should be numpy arrays into numpy arrays"""
        for key in d:
            if type(d[key]) is list:
                if myDataclass.__annotations__[key] is np.ndarray:
                    d[key] = np.array(d[key])
            elif type(d[key]) is dict:
                d[key] = _deal_with_arrays(d[key])
        return d

    # Deal with expected numpy arrays
    d = _deal_with_arrays(d)

    def _load_into_dataclass(d, myDataclass):
        """Loads data from a dict into a dataclass"""
        for key in d:
            if type(d[key]) is dict:
                setattr(
                    myDataclass,
                    key,
                    _load_into_dataclass(d[key], getattr(myDataclass, key))
                )
            else:
                setattr(myDataclass, key, d[key])
        return myDataclass

    # myDataclass = dcw.fromdict(myDataclass, d)
    myDataclass = _load_into_dataclass(d, myDataclass)
    
    # If there is a __post_init__ method, call it
    if hasattr(myDataclass, '__post_init__'):
        myDataclass.__post_init__()

    return myDataclass


def dataclasses_equal(d1, d2):
    """
    Assesses whether two dataclasses (and their nested dataclasses) are equal.
    """
    for k, v in d1.__dict__.items():
        if is_dataclass(v):
            if not dataclasses_equal(v, getattr(d2, k)):
                return False
        elif isinstance(d2.__dict__[k], np.ndarray):
            if not d2.__dict__[k].shape == v.shape:
                return False
            if not np.allclose(d2.__dict__[k], v):
                return False
        elif d2.__dict__[k] != v:
            return False
    return True
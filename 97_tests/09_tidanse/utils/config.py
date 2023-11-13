import yaml
import numpy as np
from typing import List
from dataclasses import dataclass, field

@dataclass
class Configuration:
    mcRuns: int = 1
    originalSeed: int = 0
    rngState: np.random.RandomState = None  # initialized in __post_init__
    refSensorIdx: int = 0
    fs: int = 16000
    nSamplesTot: int = 1000
    nSamplesTotOnline: int = 1000000

    K: int = 15
    Mk: int = 5

    # Online processing
    B: int = 500
    overlapB: int = 0
    beta: float = 0.98

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
        # Set seed
        np.random.seed(self.originalSeed)
        self.rngState = np.random.get_state()

        if self.plotOnlyCost:
            raise NotImplementedError("`plotOnlyCost` not implemented yet")


    def __repr__(self):
        """Return string representation of configuration."""
        config_str = "Configuration:\n"
        config_str += "+----------------------+----------------------+\n"
        for key, value in self.__dict__.items():
            config_str += f"| {key.ljust(20)} | {str(value).ljust(20)} |\n"
        config_str += "+----------------------+----------------------+\n"
        return config_str

    def from_yaml(self, pathToYaml: str = ''):
        """Read configuration from yaml file."""
        with open(pathToYaml, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(self, key, value)
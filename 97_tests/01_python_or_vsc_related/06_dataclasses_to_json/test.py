
from dataclasses import dataclass, field, fields, replace, make_dataclass
import numpy as np
import os, json, sys, copy
import dataclass_wizard as dcw
from pathlib import Path
import pickle, gzip

@dataclass
class MyClass:
    a : np.ndarray = np.array([[1,2,5], [3,4,5]])
    b : str = ''
    c : float = 0.

    def save(self, filename: str):
        save_to_json(self, filename)

    def load(self, filename: str):
        return load_from_json(filename, self)


def main():

    fname = f'{Path(__file__).parent}'

    # c = MyClass()
    # c.save(fname)

    # c2 = MyClass()
    # c2.load(fname)

    c = ProgramSettings()
    c.save(fname)

    c2 = ProgramSettings()
    c2.load(fname)

    stop = 1


def save_to_json(mycls, filename):
    """Saves dataclass to JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(filename)
    if file_extension != '.json':
        print(f'The filename ("{filename}") should be that of a JSON file. Modifying...')
        filename = filename_alone + '.json'
        print(f'Filename modified to "{filename}".')

    # Convert arrays to lists before export
    for field in fields(mycls):
        if field.type is np.ndarray:
            newVal = field.default.tolist() + ['!!NPARRAY']  # include a flag to re-conversion to np.ndarray when reading the JSON file
            para = {field.name: newVal}
            mycls = replace(mycls, **para)

    jsondict = dcw.asdict(mycls)  # https://stackoverflow.com/a/69059343

    with open(filename, 'w') as file_json:
        json.dump(jsondict, file_json, indent=4)
    file_json.close()


def load_from_json(path_to_json_file, mycls):
    """Loads dataclass from JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(path_to_json_file)
    if file_extension != '.json':
        print(f'The filename ("{path_to_json_file}") should be that of a JSON file. Modifying...')
        path_to_json_file = filename_alone + '.json'
        print(f'Filename modified to "{path_to_json_file}".')

    with open(path_to_json_file) as fp:
        d = json.load(fp)

    # Create surrogate class
    c = make_dataclass('MySurrogate', [(key, type(d[key])) for key in d])
    # Fill it in with dict entries <-- TODO -- probably simpler to directly fill in `mycls_out` from `d`
    mycls_surrog = dcw.fromdict(c, d)

    # Fill in a new correct instance of the desired dataclass
    mycls_out = copy.copy(mycls)
    for field in fields(mycls_surrog):
        a = getattr(mycls_surrog, field.name)
        if field.type is list and a != []:
            if a[-1] == '!!NPARRAY':
                a = np.array(a[:-1])
        setattr(mycls_out, field.name, a)

    return mycls_out


def shorten_path(file_path, length=3):
    """Splits `file_path` into separate parts, select the last 
    `length` elements and join them again
    -- from: https://stackoverflow.com/a/49758154
    """
    return Path(*Path(file_path).parts[-length:])


def met_save(self, foldername: str, exportType='json'):
    """
    Saves program settings so they can be loaded again later
    
    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    exportType : str
        Type of export. "json": exporting to JSON file. "pkl": exporting to PKL.GZ archive.
    """

    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        Path(foldername).mkdir(parents=True)
        print(f'Created output directory ".../{shortPath}".')

    fullPath = f'{foldername}/{type(self).__name__}'
    if exportType == 'pkl':
        fullPath += '.pkl.gz'
        pickle.dump(self, gzip.open(fullPath, 'wb'))
    elif exportType == 'json':
        fullPath += '.json'
        save_to_json(self, fullPath)

    print(f'<{type(self).__name__}> object data exported to directory\n".../{shortPath}".')


def met_load(self, foldername: str, silent=False, dataType='json'):
    """
    Loads program settings object from file

    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    silent : bool
        If True, no printouts.
    dataType : str
        Type of file to import. "json": JSON file. "pkl": PKL.GZ archive.
    """
    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        raise ValueError(f'The folder "{foldername}" cannot be found.')

    if dataType == 'pkl':
        baseExtension = '.pkl.gz'
        altExtension = '.json'
    elif dataType == 'json':
        baseExtension = '.json'
        altExtension = '.pkl.gz'
    
    pathToFile = f'{foldername}/{type(self).__name__}{baseExtension}'
    if not Path(pathToFile).is_file():
        pathToAlternativeFile = f'{foldername}/{type(self).__name__}{altExtension}'
        if Path(pathToAlternativeFile).is_file():
            print(f'The file\n"{pathToFile}"\ndoes not exist. Loading\n"{pathToAlternativeFile}"\ninstead.')
            pathToFile = copy(pathToAlternativeFile)
            baseExtension = copy(altExtension)
        else:
            raise ValueError(f'Import issue, file\n"{pathToFile}"\nnot found (with either possible extensions).')

    if baseExtension == '.json':
        p = load_from_json(pathToFile, self)
    elif baseExtension == '.pkl.gz':
        p = pickle.load(gzip.open(pathToFile, 'r'))
    else:
        raise ValueError(f'Incorrect base extension: "{baseExtension}".')
    
    if not silent:
        print(f'<{type(self).__name__}> object data loaded from directory\n".../{shortPath}".')

    return p


@dataclass
class ProgramSettings(object):
    """Class for keeping track of global simulation settings"""
    # Signal generation
    samplingFrequency: float = 16000.       # [samples/s] base sampling frequency
    acousticScenarioPath: str = ''          # path to acoustic scenario to be used
    signalDuration: float = 1               # [s] total signal duration
    desiredSignalFile: list[str] = field(default_factory=list)            # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)              # list of paths to noise signal file(s)
    baseSNR: int = 0                        # [dB] SNR between dry desired signals and dry noise
    selfnoiseSNR: int = -50                 # [dB] microphone self-noise SNR
    referenceSensor: int = 0                # Index of the reference sensor at each node
    wasnTopology: str = 'fully_connected'   # WASN topology (fully connected or ad hoc)
    # VAD
    VADwinLength: float = 40e-3             # [s] VAD window length
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    # DANSE
    danseUpdating: str = 'sequential'       # node-updating scheme: "sequential" or "simultaneous"
    timeBtwExternalFiltUpdates: float = 0   # [s] minimum time between 2 consecutive external filter update (i.e. filters that are used for broadcasting)
                                            #  ^---> If 0: equivalent to updating coefficients every `chunkSize * (1 - chunkOverlap)` new captured samples 
    DFTsize: int = 1024                     # DFT/FFT size
    Ns: int = DFTsize // 2                  # window shift = number of new samples per chunk [samples] 
    winOvlp: float = 1 - Ns / DFTsize       # window overlap [/100%]
    danseWindow: np.ndarray = np.hanning(DFTsize)     # DANSE window for FFT/IFFT operations
    initialWeightsAmplitude: float = 1.     # maximum amplitude of initial random filter coefficients
    expAvg50PercentTime: float = 2.         # [s] Time in the past at which the value is weighted by 50% via exponential averaging
                                            # -- Used to compute beta in, e.g.: Ryy[l] = beta * Ryy[l - 1] + (1 - beta) * y[l] * y[l]^H
    performGEVD: bool = True                # if True, perform GEVD in DANSE
    GEVDrank: int = 1                       # GEVD rank approximation (only used is <performGEVD> is True)
    computeLocalEstimate: bool = False      # if True, compute also an estimate of the desired signal using only local sensor observations
    computeCentralizedEstimate: bool = False      # if True, compute also an estimate of the desired signal using all sensor observations, as in centralized processing
    bypassFilterUpdates: bool = False       # if True, only update covariance matrices, do not update filter coefficients (no adaptive filtering)
    # Inter-node broadcasting
    broadcastDomain: str = 'wholeChunk_fd'  # inter-node data broadcasting domain:
                                            # -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
                                            # -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
                                            # -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    broadcastLength: int = 8                # [samples] number of (compressed) signal samples to be broadcasted at a time to other nodes
    updateTDfilterEvery : float = 1.        # [s] duration of pause between two consecutive time-domain filter updates.
    # Desired signal estimation
    desSigProcessingType: str = 'wola'      # processing scheme used to compute the desired signal estimates:
                                            # "wola": WOLA synthesis,
                                            # "conv": linear convolution via T(z)-approximation.
    # Speech enhancement metrics
    minFiltUpdatesForMetricsComputation: int = 15   # minimum number of DANSE filter updates that must have been performed
                                                    # at the starting sample of the signal chunk used for computating the metrics
    gammafwSNRseg: float = 0.2              # gamma exponent for fwSNRseg computation
    frameLenfwSNRseg: float = 0.03          # [s] time window duration for fwSNRseg computation

    def load(self, filename: str, silent=False):
        return met_load(self, filename, silent)

    def save(self, filename: str):
        # Save data as archive
        met_save(self, filename)


if __name__ == '__main__':
    sys.exit(main())
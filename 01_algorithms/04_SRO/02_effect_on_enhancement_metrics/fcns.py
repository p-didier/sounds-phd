import numpy as np
from dataclasses import dataclass, field
from scipy.io import wavfile
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
if not any("_general_fcts" in s for s in sys.path):
    # Find path to root folder
    rootFolder = 'sounds-phd'
    pathToRoot = Path(__file__)
    while PurePath(pathToRoot).name != rootFolder:
        pathToRoot = pathToRoot.parent
    sys.path.append(f'{pathToRoot}/_general_fcts')
from metrics import eval_enhancement

@dataclass
class Parameters():
    signalPath: str = ''
    srosToTest: np.ndarray = np.array([])
    metricsToCompute: list[str] = field(default_factory=list)
    baseFs: int = 16000


def get_signal(path):

    fs, sig = wavfile.read(path)

    return sig, fs


def compute_metrics(referenceSignal, signalWithSRO, fs, metricsToCompute):

    metrics = dict()

    if 'fwSNRseg' in metricsToCompute:
        fwSNRseg = eval_enhancement.get_fwsnrseg(referenceSignal, signalWithSRO, fs)
        metrics.update({'fwSNRseg': np.mean(fwSNRseg)})
    if 'STOI' in metricsToCompute:
        stoi = eval_enhancement.stoi_fcn(referenceSignal, signalWithSRO, fs)
        metrics.update({'STOI': stoi})
    if 'PESQ' in metricsToCompute:
        flagNoPesq = False
        if fs == 16000:
            mode = 'wb'
        elif fs == 8000:
            mode = 'nb'
        else:
            print(f'fs should be 16 or 8 kHz (current value: {fs/1e3} kHz). Not computing PESQ')
            flagNoPesq = True
        if not flagNoPesq:
            pesq = eval_enhancement.pesq(fs, referenceSignal, signalWithSRO, mode)
        else:
            pesq = None
        metrics.update({'PESQ': pesq})   

    return metrics


def plot_metrics(metrics, x):

    metricNames = list(metrics.keys())

    fig = plt.figure(figsize=(8,3))
    for ii in range(len(metricNames)):
        ax = fig.add_subplot(100 + len(metricNames) * 10 + ii + 1)
        ax.plot(x, metrics[metricNames[ii]], 'k.-')
        ax.grid()
        ax.set_title(metricNames[ii])
        if metricNames[ii] in ['stoi', 'STOI']:
            ax.set_ylim([0,1])
        ax.set_xlabel('SRO [ppm]')
    plt.tight_layout()
    plt.show()

    return None
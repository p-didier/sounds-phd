
import numpy as np
import scipy.signal as sig
from dataclasses import dataclass
from .config import Configuration

class Node:
    # Class to store node information (from create_wasn())
    def __init__(self, signal, noiseOnly, neighbors):
        self.signal = signal
        self.noiseOnly = noiseOnly
        self.desiredOnly = signal - noiseOnly
        self.neighbors = neighbors

class WASN:
    # Class to store WASN information (from create_wasn())
    def __init__(self):
        self.nodes: list[Node] = []
        self.ySTFT = []
        self.refSensorIdx = 0

    def compute_stft_signals(self, fs, L, hop):
        nFrames = self.nodes[0].signal.shape[1] // hop + 1
        for k in range(len(self.nodes)):
            Mk = self.nodes[k].signal.shape[0]
            ySTFTcurr = np.zeros((Mk, L // 2 + 1, nFrames), dtype=np.complex)
            for m in range(Mk):
                _, _, ySTFTcurr[m, :, :] = sig.stft(
                    x=self.nodes[k].signal[m, :],
                    fs=fs,
                    window="hann",
                    nperseg=L,
                    noverlap=L - hop,
                    nfft=L,
                    return_onesided=True,
                    padded=False,
                )
            self.ySTFT.append(ySTFTcurr)

class SceneCreator:
    """Class to create acoustic scenes."""
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.x = None
        self.n = None
        self.wasn = None

    def prepare_scene(self):
        self.create_scene()
        self.create_wasn()

    def create_scene(self):
        """Create acoustic scene."""
        self.x, self.n = create_scene(
            nSamples=self.cfg.nSamplesTotOnline,
            nNoiseSources=self.cfg.nNoiseSources,
            K=self.cfg.K,
            Mk=self.cfg.Mk,
            snr=self.cfg.snr,
            snrSelfNoise=self.cfg.snSnr,
        )

    def create_wasn(self):
        """Create WASN."""
        # Create line topology
        neighs = []
        for k in range(self.cfg.K):
            if k == 0:
                neighs.append([1])
            elif k == self.cfg.K - 1:
                neighs.append([k - 1])
            else:
                neighs.append([k - 1, k + 1])
        # Create WASN
        self.wasn = WASN()
        self.wasn.refSensorIdx = self.cfg.refSensorIdx
        for k in range(self.cfg.K):
            self.wasn.nodes.append(Node(
                signal=self.x[k] + self.n[k],
                noiseOnly=self.n[k],
                neighbors=neighs[k]
            ))


def create_scene(nSamples, nNoiseSources, K, Mk, snr, snrSelfNoise):
    """Create acoustic scene."""
    # Generate desired source signal (random)
    desired = np.random.randn(nSamples, 1)

    # Generate noise signals (random)
    noise = np.random.randn(nSamples, nNoiseSources)

    # Generate microphone signals
    x = []
    n = []
    for _ in range(K):
        # Create random mixing matrix
        mixingMatrix = np.random.randn(Mk, 1)
        # Compute microphone signals
        x.append(mixingMatrix @ desired.T)
        # Create noise signals
        mixingMatrix = np.random.randn(Mk, nNoiseSources)
        noiseAtMic = mixingMatrix @ noise.T * 10 ** (-snr / 20) +\
            np.random.randn(Mk, nSamples) * 10 ** (-snrSelfNoise / 20)
        # noiseAtMic = np.random.randn(MK, NSAMPLES_TOT_ONLINE) * 10 ** (-SNR / 20)
        n.append(noiseAtMic)

    return x, n
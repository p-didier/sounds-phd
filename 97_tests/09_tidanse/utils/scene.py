
import numpy as np
import scipy.signal as sig
from dataclasses import dataclass
from .config import Configuration

class Node:
    # Class to store node information (from create_wasn())
    def __init__(self, signal=None, noiseOnly=None, smDesired=None, smNoise=None):
        self.signal = signal
        self.noiseOnly = noiseOnly
        if signal is not None and noiseOnly is not None:
            self.desiredOnly = signal - noiseOnly
        self.Ak_s = smDesired   # steering matrix for desired signal
        self.Ak_n = smNoise     # steering matrix for noise
    
    def query(self, Mk, B, nNoiseSources, snr, snSnr):
        """Query `B` samples from node."""
        if self.Ak_s is None or self.Ak_n is None:
            raise ValueError("Steering matrices not defined. Cannot query samples.")
        # Query samples
        s = self.Ak_s @ np.random.randn(1, B)
        selfNoise = np.random.randn(Mk, B)
        selfNoise *= 10 ** (-snSnr / 20)  # apply self-noise SNR
        localizedNoise = self.Ak_n @ np.random.randn(nNoiseSources, B)
        localizedNoise *= 10 ** (-snr / 20)  # apply noise SNR
        n = localizedNoise + selfNoise
        return s, n

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
        self.s = None  # Desired source signals, for each node (`K`-elements list)
        self.n = None  # Noise signals, for each node (`K`-elements list)
        self.Ak_s = []  # Mixing matrices for desired signals, for each node (`K`-elements list)
        self.Ak_n = []  # Mixing matrices for noise signals, for each node (`K`-elements list)
        self.wasn = None

    def prepare_scene(self):
        self.create_scene()
        self.create_wasn()

    def create_scene(self):
        """Create acoustic scene. In batch-mode, create the signals.
        In online-mode, create the mixing matrices."""
        if self.cfg.mode == 'batch':
            # Generate desired source signal (random)
            desired = np.random.randn(self.cfg.nSamplesTot, 1)
            # Generate noise signals (random)
            noise = np.random.randn(
                self.cfg.nSamplesTot,
                self.cfg.nNoiseSources
            )
            # Generate microphone signals
            x, n = [], []
            for _ in range(self.cfg.K):
                # Create random mixing matrix
                mixingMatrix = np.random.randn(self.cfg.Mk, 1)
                self.Ak_s.append(mixingMatrix)
                # Compute microphone signals
                x.append(mixingMatrix @ desired.T)
                # Create noise steering matrix
                mixingMatrixNoise = np.random.randn(self.cfg.Mk, self.cfg.nNoiseSources)
                self.Ak_n.append(mixingMatrixNoise)
                # Create noise signals
                noiseAtMic = mixingMatrixNoise @ noise.T *\
                    10 ** (-self.cfg.snr / 20) +\
                    np.random.randn(self.cfg.Mk, self.cfg.nSamplesTot) *\
                    10 ** (-self.cfg.snSnr / 20)
                n.append(noiseAtMic)
            self.s, self.n = x, n  # Store

        elif self.cfg.mode == 'online':
            mixMatDesired, mixMatNoise = [], []
            for _ in range(self.cfg.K):
                # Create random mixing matrices
                mixMatDesired.append(np.random.randn(self.cfg.Mk, 1))
                mixMatNoise.append(np.random.randn(self.cfg.Mk, self.cfg.nNoiseSources))
            self.Ak_s = mixMatDesired   # Store
            self.Ak_n = mixMatNoise     # Store

    def create_wasn(self):
        """Create WASN."""
        # Create WASN
        self.wasn = WASN()
        self.wasn.refSensorIdx = self.cfg.refSensorIdx
        for k in range(self.cfg.K):
            if self.cfg.mode == 'batch':
                self.wasn.nodes.append(Node(
                    signal=self.s[k] + self.n[k],
                    noiseOnly=self.n[k],
                ))
            elif self.cfg.mode == 'online':
                self.wasn.nodes.append(Node(
                    smDesired=self.Ak_s[k],
                    smNoise=self.Ak_n[k],
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
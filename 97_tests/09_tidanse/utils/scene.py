
import numpy as np
import scipy.signal as sig
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
    
    def query(self, Mk, B, nNoiseSources, snr, snSnr):
        """Query `B` samples from each node."""
        desiredSigChunk =  np.random.randn(1, B)
        noiseSigChunk = np.random.randn(nNoiseSources, B)
        s, n = [], []
        for k in range(len(self.nodes)):
            if self.nodes[k].Ak_s is None or self.nodes[k].Ak_n is None:
                raise ValueError(f"Steering matrices not defined at node {k}. Cannot query samples.")
            # Query samples
            sCurr = self.nodes[k].Ak_s @ desiredSigChunk
            localizedNoise = self.nodes[k].Ak_n @ noiseSigChunk
            localizedNoise *= 10 ** (-snr / 20)  # apply noise SNR
            selfNoise = np.random.randn(Mk, B)
            selfNoise *= 10 ** (-snSnr / 20)  # apply self-noise SNR
            nCurr = localizedNoise + selfNoise
            # Store
            s.append(sCurr)
            n.append(nCurr)
        return s, n

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
            desired = np.random.randn(self.cfg.nSamplesBatch, 1)
            # Generate noise signals (random)
            noise = np.random.randn(
                self.cfg.nSamplesBatch,
                self.cfg.nNoiseSources
            )
            # Generate microphone signals
            s, n = [], []
            for _ in range(self.cfg.K):
                # Create random mixing matrix
                mixingMatrix = np.abs(np.random.randn(self.cfg.Mk, 1))
                self.Ak_s.append(mixingMatrix)
                # Compute microphone signals
                s.append(mixingMatrix @ desired.T)
                # Create noise steering matrix
                mixingMatrixNoise = np.abs(np.random.randn(self.cfg.Mk, self.cfg.nNoiseSources))
                self.Ak_n.append(mixingMatrixNoise)
                # Create noise signals
                noiseAtMic = mixingMatrixNoise @ noise.T *\
                    10 ** (-self.cfg.snr / 20) +\
                    np.random.randn(self.cfg.Mk, self.cfg.nSamplesBatch) *\
                    10 ** (-self.cfg.snSnr / 20)
                n.append(noiseAtMic)
            self.s, self.n = s, n  # Store

        elif self.cfg.mode == 'online':
            mixMatDesired, mixMatNoise = [], []
            for _ in range(self.cfg.K):
                # Create random mixing matrices
                mixMatDesired.append(np.abs(np.random.randn(self.cfg.Mk, 1)))
                mixMatNoise.append(np.abs(np.random.randn(self.cfg.Mk, self.cfg.nNoiseSources)))
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
                    smDesired=self.Ak_s[k],
                    smNoise=self.Ak_n[k],
                ))
            elif self.cfg.mode == 'online':
                self.wasn.nodes.append(Node(
                    smDesired=self.Ak_s[k],
                    smNoise=self.Ak_n[k],
                ))
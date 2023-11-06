# Purpose of script:
# TI-DANSE trial from scratch.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

SEED = 0
REFSENSORIDX = 0  # Reference sensor index (for desired signal)
ROOM_DIM = 10  # [m]
FS = 16000
NSAMPLES = 10000
K = 5  # Number of nodes
MK = 1 # Number of microphones per node (same for all nodes)
N = 1024  # STFT window length
N_NOISE_SOURCES = 2
#
MODE = 'batch'  # 'wola' or 'online' or 'batch'

class Node:
    # Class to store node information (from create_wasn())
    def __init__(self, signal, noiseOnly, neighbors):
        self.signal = signal
        self.noiseOnly = noiseOnly
        self.desired = (signal - noiseOnly)[REFSENSORIDX, :]
        self.neighbors = neighbors

class WASN:
    # Class to store WASN information (from create_wasn())
    def __init__(self):
        self.nodes: list[Node] = []
        self.ySTFT = []

    def compute_stft_signals(self, L=N, hop=N // 2):
        nFrames = self.nodes[0].signal.shape[1] // hop + 1
        for k in range(len(self.nodes)):
            ySTFTcurr = np.zeros((MK, L // 2 + 1, nFrames), dtype=np.complex)
            for m in range(MK):
                _, _, ySTFTcurr[m, :, :] = sig.stft(
                    x=self.nodes[k].signal[m, :],
                    fs=FS,
                    window="hann",
                    nperseg=L,
                    noverlap=L - hop,
                    nfft=L,
                    return_onesided=True,
                    padded=False,
                )
            self.ySTFT.append(ySTFTcurr)

def main():
    """Main function (called by default when running script)."""
    
    np.random.seed(SEED)

    # Create acoustic scene
    x, n = create_scene()

    # Create WASN
    wasn = create_wasn(x, n)

    # Compute STFT signals
    wasn.compute_stft_signals()

    # TI-DANSE
    batch_ti_danse(wasn)


def ti_danse(wasn: WASN):
    if MODE == 'wola':
        raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
    elif MODE == 'online':
        raise NotImplementedError  # TODO: implement TI-DANSE online
    elif MODE == 'batch':
        batch_ti_danse(wasn)


def batch_ti_danse(wasn: WASN):
    wTilde = [
        np.ones((MK + 1)) / 100  # initialize wTilde to 1
        for _ in range(K)
    ]
    i = 0  # DANSE iteration index
    q = 0  # currently updating node index
    mmse = [[1] for _ in range(K)]  # MMSE (arbitrary init. value)
    stopcond = np.all([me[-1] < 1e-9 for me in mmse])
    while not stopcond:
        print(f"i = {i}, q = {q}, mmse = {[me[-1] for me in mmse]}")
        # Compute compressed signals
        z = np.zeros((K, NSAMPLES))
        z_noise = np.zeros((K, NSAMPLES))
        for k in range(K):
            yk = wasn.nodes[k].signal
            nk = wasn.nodes[k].noiseOnly
            pk = wTilde[k][:-1] / wTilde[k][-1]  # TI-DANSE fusion vector
            # Inner product of pk and yk across channels (fused signals)
            zk = np.sum(yk * pk[:, None], axis=0)
            zk_noise = np.sum(nk * pk[:, None], axis=0)
            z[k, :] = zk
            z_noise[k, :] = zk_noise
        
        # Compute eta
        eta = np.sum(z, axis=0)
        eta_noise = np.sum(z_noise, axis=0)

        yTilde = [_ for _ in wasn.nodes]
        nTilde = [_ for _ in wasn.nodes]
        for k in range(K):
            yTilde[k] = np.concatenate((
                wasn.nodes[k].signal,
                (eta - z[k, :])[np.newaxis, :]
            ), axis=0)
            nTilde[k] = np.concatenate((
                wasn.nodes[k].noiseOnly,
                (eta_noise - z_noise[k, :])[np.newaxis, :]
            ), axis=0)

            if k == q:
                # Compute batch covariance matrix
                Ryy = yTilde[k] @ yTilde[k].T.conj()
                Rnn = nTilde[k] @ nTilde[k].T.conj()
                # Update MMSE filter
                ek = np.zeros((MK + 1))
                ek[REFSENSORIDX] = 1
                wTilde[k] = np.linalg.inv(Ryy) @ (Ryy - Rnn) @ ek

        # Compute MMSE estimate of desired signal
        for k in range(K):
            # Filter `yTilde` with MMSE filter
            dHat = wTilde[k] @ yTilde[k]
            # Compute MMSE
            currMMSE = np.mean(np.abs(dHat - wasn.nodes[k].desired) ** 2)
            if np.isnan(currMMSE):
                raise ValueError("MMSE is NaN")
            else:
                mmse[k].append(currMMSE)
        
        # Update DANSE iteration index
        i += 1
        # Update node index
        q = (q + 1) % K

def create_wasn(x, n):
    # Create line topology
    neighs = []
    for k in range(K):
        if k == 0:
            neighs.append([1])
        elif k == K - 1:
            neighs.append([k - 1])
        else:
            neighs.append([k - 1, k + 1])

    # Create WASN
    wasn = WASN()
    for k in range(K):
        wasn.nodes.append(Node(
            signal=x[k] + n[k],
            noiseOnly=n[k],
            neighbors=neighs[k]
        ))

    return wasn


def create_scene():

    # Generate desired source signal (random)
    desired = np.random.randn(NSAMPLES)

    # Generate noise signals (random)
    noise = np.random.randn(NSAMPLES, N_NOISE_SOURCES)

    # Generate microphone signals
    x = []
    n = []
    for _ in range(K):
        curr_x = np.zeros((MK, NSAMPLES))
        curr_n = np.zeros((MK, NSAMPLES))
        for m in range(MK):
            # Desired signal contribution (random delay)
            delay = np.random.randint(0, N)
            desSigContr = np.roll(desired, delay)[:NSAMPLES]
            # Noise contribution (random delay)
            noiseContr = np.zeros(NSAMPLES)
            for nSources in range(N_NOISE_SOURCES):
                delay = np.random.randint(0, N)
                noiseContr += np.roll(noise[:, nSources], delay)[:NSAMPLES]
            # Add self-noise (uncorrelated across sensors)
            noiseContr += np.random.randn(NSAMPLES) * 0.1
            # Mix
            curr_x[m, :] = desSigContr
            curr_n[m, :] = noiseContr
        x.append(curr_x)
        n.append(curr_n)

    return x, n



if __name__ == '__main__':
    sys.exit(main())
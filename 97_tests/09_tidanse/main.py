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
#
K = 5  # Number of nodes
MK = 5 # Number of microphones per node (same for all nodes)
#
N = 1024  # STFT window length
N_NOISE_SOURCES = 2
EPS = 1e-5  # Stopping criterion constant
MAXITER = 1000  # Maximum number of iterations
SNR = 10  # [dB] SNR of desired signal
SNSNR = 5  # [dB] SNR of self-noise signals
#
ALGOS = ['danse', 'ti-danse']  # 'danse' or 'ti-danse'
MODE = 'batch'  # 'wola' or 'online' or 'batch'
GEVD = False  # Use GEVD-MWF instead of MWF

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

    # TI-DANSE
    run(wasn)


def run(wasn: WASN):
    if MODE == 'wola':
        # Compute STFT signals
        wasn.compute_stft_signals()
        raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
    elif MODE == 'online':
        raise NotImplementedError  # TODO: implement TI-DANSE online
    elif MODE == 'batch':
        batch_run(wasn)


def batch_run(wasn: WASN):
    # Compute centralized cost
    mmseCentral = get_centr_cost(wasn)

    mmsePerAlgo = [[] for _ in range(len(ALGOS))]
    for algo in ALGOS:
        if algo == 'danse':
            dimTilde = MK + K - 1  # fully connected
        else:
            dimTilde = MK + 1
        wTilde = [
            np.ones(dimTilde) / 100  # initialize wTilde to 1
            for _ in range(K)
        ]
        i = 0  # DANSE iteration index
        q = 0  # currently updating node index
        mmse = [[] for _ in range(K)]  # MMSE
        stopcond = False
        while not stopcond:
            # Compute compressed signals
            z = np.zeros((K, NSAMPLES))
            z_noise = np.zeros((K, NSAMPLES))
            for k in range(K):
                yk = wasn.nodes[k].signal
                nk = wasn.nodes[k].noiseOnly
                if algo == 'danse':
                    pk = wTilde[k][:MK]  # DANSE fusion vector
                elif algo == 'ti-danse':
                    pk = wTilde[k][:MK] / wTilde[k][-1]  # TI-DANSE fusion vector
                    # pk = wTilde[k][:MK]# / wTilde[k][-1]  # TI-DANSE fusion vector
                # Inner product of pk and yk across channels (fused signals)
                zk = np.sum(yk * pk[:, None], axis=0)
                zk_noise = np.sum(nk * pk[:, None], axis=0)
                z[k, :] = zk
                z_noise[k, :] = zk_noise
            
            if algo == 'ti-danse':
                # Compute eta
                eta = np.sum(z, axis=0)
                eta_noise = np.sum(z_noise, axis=0)

            yTilde = [_ for _ in wasn.nodes]
            nTilde = [_ for _ in wasn.nodes]
            for k in range(K):
                if algo == 'danse':
                    zMk = z[np.arange(K) != k, :]
                    zMk_noise = z_noise[np.arange(K) != k, :]
                    yTilde[k] = np.concatenate((
                        wasn.nodes[k].signal,
                        zMk
                    ), axis=0)
                    nTilde[k] = np.concatenate((
                        wasn.nodes[k].noiseOnly,
                        zMk_noise
                    ), axis=0)
                elif algo == 'ti-danse':
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
                    ek = np.zeros(dimTilde)
                    ek[REFSENSORIDX] = 1
                    wTilde[k] = filter_update(Ryy, Rnn, gevd=GEVD) @ ek

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
            
            # Print progress
            print(f"i = {i}, q = {q}, mmse = {[me[-1] for me in mmse]}")
            # Update DANSE iteration index
            i += 1
            # Update node index
            q = (q + 1) % K
            # Check stopping condition
            if i > K:
                stopcond = i >= MAXITER or np.all([np.abs(me[-1] - me[-1-K]) < EPS for me in mmse])
                
        # Store MMSE
        mmsePerAlgo[ALGOS.index(algo)] = mmse
    
    # Plot
    plot_results(mmsePerAlgo, mmseCentral, onlyLoss=True)


def plot_results(mmsePerAlgo, mmseCentral, onlyLoss=False):
    """Plot results."""
    if onlyLoss:
        symbols = ['o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8']
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(6.5, 3.5)
        for idxAlgo in range(len(ALGOS)):
            xmax = len(mmsePerAlgo[idxAlgo][0])-1
            data = np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0)
            axes.loglog(
                data,
                f'{symbols[idxAlgo]}-C{idxAlgo}',
                label=f'{ALGOS[idxAlgo].upper()} ({len(data)} iters, $\\mathcal{{L}}=${"{:.3g}".format(data[-1], -4)})'
            )
        axes.hlines(np.mean(mmseCentral), 0, xmax, 'k', linestyle="--", label="Centralized")
        axes.set_xlabel("Iteration index")
        axes.set_ylabel("Cost")
        axes.legend(loc='upper right')
        axes.set_xlim([0, xmax])
        axes.grid()
    else:
        fig, axes = plt.subplots(2, len(ALGOS))
        fig.set_size_inches(8.5, 3.5)
        for idxAlgo in range(len(ALGOS)):
            if len(ALGOS) == 1:
                currAx = axes
            else:
                currAx = axes[:, idxAlgo]
            xmax = len(mmsePerAlgo[idxAlgo][0])-1
            for k in range(K):
                currAx[0].loglog(mmsePerAlgo[idxAlgo][k], f'o-C{k}', label=f"Node {k}")
                currAx[0].hlines(mmseCentral[k], 0, xmax, f'C{k}', linestyle="--")
            currAx[0].set_xlabel(f"{ALGOS[idxAlgo].upper()} iteration index")
            currAx[0].set_ylabel("MMSE per node")
            currAx[0].legend(loc='upper right')
            currAx[0].set_xlim([0, xmax])
            currAx[0].grid()
            #
            currAx[1].loglog(np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0), 'o-k', label=ALGOS[idxAlgo].upper())
            currAx[1].hlines(np.mean(mmseCentral), 0, xmax, 'b', linestyle="--", label="Centralized")
            currAx[1].set_xlabel(f"{ALGOS[idxAlgo].upper()} iteration index")
            currAx[1].set_ylabel("Cost")
            currAx[1].legend(loc='upper right')
            currAx[1].set_xlim([0, xmax])
            currAx[1].grid()
    #
    fig.tight_layout()	
    plt.show()


def create_wasn(x, n):
    """Create WASN."""
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
    """Create acoustic scene."""
    # Generate desired source signal (random)
    desired = np.random.randn(NSAMPLES, 1)

    # Generate noise signals (random)
    noise = np.random.randn(NSAMPLES, N_NOISE_SOURCES)

    # Generate microphone signals
    x = []
    n = []
    for k in range(K):
        # Create random mixing matrix
        mixingMatrix = np.random.randn(MK, 1)
        # Compute microphone signals
        x.append(mixingMatrix @ desired.T)
        # Create noise signals
        mixingMatrix = np.random.randn(MK, N_NOISE_SOURCES)
        noiseAtMic = mixingMatrix @ noise.T + np.random.randn(MK, NSAMPLES) * 10 ** (-SNSNR / 20)
        noiseAtMic *= 10 ** (-SNR / 20)
        n.append(noiseAtMic)

    return x, n


def get_centr_cost(wasn: WASN):
    """Compute centralized cost (MMSE) for each node."""
    y = np.concatenate(tuple(
        wasn.nodes[k].signal
        for k in range(K)
    ), axis=0)
    n = np.concatenate(tuple(
        wasn.nodes[k].noiseOnly
        for k in range(K)
    ), axis=0)
    Ryy = y @ y.T.conj()
    Rnn = n @ n.T.conj()
    wCentral = filter_update(Ryy, Rnn, gevd=GEVD)
    mmseCentral = np.zeros(K)
    for k in range(K):
        ek = np.zeros(K * MK)
        ek[k * MK + REFSENSORIDX] = 1
        mmseCentral[k] = np.mean(np.abs((wCentral @ ek).T.conj() @ y - wasn.nodes[k].desired) ** 2)
    
    return mmseCentral


def filter_update(Ryy, Rnn, gevd=False):
    """Update filter using GEVD-MWF or MWF."""
    if gevd:
        raise NotImplementedError  # TODO: implement GEVD
    else:
        return np.linalg.inv(Ryy) @ (Ryy - Rnn)


if __name__ == '__main__':
    sys.exit(main())
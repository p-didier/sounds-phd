# Purpose of script:
# TI-DANSE trial from scratch.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
import scipy.signal as sig
import scipy.linalg as sla
import matplotlib.pyplot as plt

SEED = 0
REFSENSORIDX = 0  # Reference sensor index (for desired signal)
ROOM_DIM = 10  # [m]
FS = 16000
NSAMPLES = 10000
#
K = 15  # Number of nodes
MK = 5 # Number of microphones per node (same for all nodes)
# Online processing
B = 100  # Block size (number of samples per block)
OVERLAP_B = 0.5  # Overlap between blocks (in percentage)
BETA = 0.99  # Forgetting factor for batch covariance matrix
#
N_STFT = 1024  # STFT window length
N_NOISE_SOURCES = 2
EPS = 1e-5  # Stopping criterion constant
MAXITER = 1000  # Maximum number of iterations
SNR = 10  # [dB] SNR of desired signal
SNSNR = 5  # [dB] SNR of self-noise signals
#
# ALGOS = ['danse', 'ti-danse']  # 'danse' or 'ti-danse'
ALGOS = ['danse']  # 'danse' or 'ti-danse'
# MODE = 'batch'  # 'wola' or 'online' or 'batch'
MODE = 'online'  # 'wola' or 'online' or 'batch'
# GEVD = True  # Use GEVD-MWF
GEVD = False  # Use MWF

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

    def compute_stft_signals(self, L=N_STFT, hop=N_STFT // 2):
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
    """Run simulation."""
    if MODE == 'wola':
        # Compute STFT signals
        wasn.compute_stft_signals()
        raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
    elif MODE in ['batch','online']:
        mmsePerAlgo, mmseCentral = batch_or_online_run(wasn)
    # Plot
    plot_results(mmsePerAlgo, mmseCentral, onlyLoss=True)


def batch_or_online_run(wasn: WASN):
    # Compute centralized cost
    mmseCentral = get_centr_cost(wasn)

    mmsePerAlgo = [[] for _ in range(len(ALGOS))]
    for algo in ALGOS:
        if algo == 'danse':
            dimTilde = MK + K - 1  # fully connected
        else:
            dimTilde = MK + 1
        eq = np.zeros(dimTilde)
        eq[REFSENSORIDX] = 1  # reference sensor selection vector
        wTilde = [
            np.ones(dimTilde) / 100  # initialize wTilde to 1
            for _ in range(K)
        ]
        Rss = np.zeros((dimTilde, dimTilde))
        Rnn = np.zeros((dimTilde, dimTilde))
        i = 0  # DANSE iteration index
        q = 0  # currently updating node index
        mmse = [[] for _ in range(K)]  # MMSE
        stopcond = False
        while not stopcond:
            # Compute compressed signals
            if MODE == 'batch':
                z_desired = np.zeros((K, NSAMPLES))
                z_noise = np.zeros((K, NSAMPLES))
                idxBegFrame, idxEndFrame = 0, NSAMPLES
            elif MODE == 'online':
                z_desired = np.zeros((K, B))
                z_noise = np.zeros((K, B))
                idxBegFrame = int(i * B * (1 - OVERLAP_B))
                idxEndFrame = int(idxBegFrame + B)
            for k in range(K):
                if MODE == 'batch':
                    sk = wasn.nodes[k].desiredOnly
                    nk = wasn.nodes[k].noiseOnly
                elif MODE == 'online':
                    sk = wasn.nodes[k].desiredOnly[:, idxBegFrame:idxEndFrame]
                    nk = wasn.nodes[k].noiseOnly[:, idxBegFrame:idxEndFrame]
                z_desired[k, :], z_noise[k, :] = get_compressed_signals(
                    sk, nk, algo, wTilde[k]
                )

            # Compute sTilde and nTilde
            sTilde, nTilde = get_tildes(algo, z_desired, z_noise, wasn, idxBegFrame, idxEndFrame)

            # Compute batch covariance matrix at updating node `q`
            if MODE == 'batch':
                Rss = sTilde[q] @ sTilde[q].T.conj()
                Rnn = nTilde[q] @ nTilde[q].T.conj()
            elif MODE == 'online':
                Rss = BETA * Rss + (1 - BETA) * sTilde[q] @ sTilde[q].T.conj() / B
                Rnn = BETA * Rnn + (1 - BETA) * nTilde[q] @ nTilde[q].T.conj() / B
            # Update MMSE filter
            wTilde[q] = filter_update(Rss + Rnn, Rnn, gevd=GEVD) @ eq

            # Compute MMSE estimate of desired signal at each node
            mmses = get_mmse(wTilde, sTilde, nTilde, wasn, idxBegFrame, idxEndFrame)
            for k in range(K):
                mmse[k].append(mmses[k])
            
            # Print progress
            print(f"i = {i}, q = {q}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}")
            # Update indices
            i += 1
            q = (q + 1) % K
            if i > K:
                stopcond = i >= MAXITER or\
                    np.all([np.abs(me[-1] - me[-1-K]) < EPS for me in mmse]) or\
                    (MODE == 'online' and int(i * B * (1 - OVERLAP_B) + B) >= NSAMPLES)
                
        # Store MMSE
        mmsePerAlgo[ALGOS.index(algo)] = mmse
    
    return mmsePerAlgo, mmseCentral


def get_compressed_signals(sk, nk, algo, wTildek):
    """Compute compressed signals using the given desired source-only and
    noise-only data."""
    if algo == 'danse':
        pk = wTildek[:MK]  # DANSE fusion vector
    elif algo == 'ti-danse':
        pk = wTildek[:MK] / wTildek[-1]  # TI-DANSE fusion vector
    # Inner product of pk and yk across channels (fused signals)
    zk_desired = np.sum(sk * pk[:, None], axis=0)
    zk_noise = np.sum(nk * pk[:, None], axis=0)
    return zk_desired, zk_noise


def get_mmse(wTilde, sTilde, nTilde, wasn: WASN, idxBeginFrame=0, idxEndFrame=NSAMPLES):
    """Compute MMSE."""
    currMMSEs = np.zeros(K)
    for k in range(K):
        dHat = wTilde[k] @ (sTilde[k] + nTilde[k])
        currMMSEs[k] = np.mean(np.abs(
            dHat - wasn.nodes[k].desiredOnly[REFSENSORIDX, idxBeginFrame:idxEndFrame]
        ) ** 2)
    return currMMSEs


def get_tildes(algo, z_desired, z_noise, wasn: WASN, idxBegFrame=0, idxEndFrame=B):
    """Compute `sTilde` and `nTilde`."""
    sTilde = [_ for _ in range(K)]
    nTilde = [_ for _ in range(K)]
    if algo == 'ti-danse':
        # Compute eta
        eta_desired = np.sum(z_desired, axis=0)
        eta_noise = np.sum(z_noise, axis=0)
    for k in range(K):
        xk = wasn.nodes[k].desiredOnly[:, idxBegFrame:idxEndFrame]
        nk = wasn.nodes[k].noiseOnly[:, idxBegFrame:idxEndFrame]
        if algo == 'danse':
            zMk_desired = z_desired[np.arange(K) != k, :]
            zMk_noise = z_noise[np.arange(K) != k, :]
            sTilde[k] = np.concatenate((
                xk,
                zMk_desired
            ), axis=0)
            nTilde[k] = np.concatenate((
                nk,
                zMk_noise
            ), axis=0)
        elif algo == 'ti-danse':
            sTilde[k] = np.concatenate((
                xk,
                (eta_desired - z_desired[k, :])[np.newaxis, :]
            ), axis=0)
            nTilde[k] = np.concatenate((
                nk,
                (eta_noise - z_noise[k, :])[np.newaxis, :]
            ), axis=0)
    return sTilde, nTilde


def plot_results(mmsePerAlgo, mmseCentral, onlyLoss=False):
    """Plot results."""
    if onlyLoss:
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(6.5, 3.5)
        for idxAlgo in range(len(ALGOS)):
            xmax = len(mmsePerAlgo[idxAlgo][0])-1
            data = np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0)
            axes.loglog(
                data,
                f'-C{idxAlgo}',
                label=f'{ALGOS[idxAlgo].upper()} ({len(data)} iters, $\\mathcal{{L}}=${"{:.3g}".format(data[-1], -4)})'
            )
        axes.hlines(np.mean(mmseCentral), 0, xmax, 'k', linestyle="--", label="Centralized")
        axes.set_xlabel("Iteration index")
        axes.set_ylabel("Cost")
        axes.legend(loc='upper right')
        axes.set_xlim([0, xmax])
        axes.grid()
        axes.set_title(f'$K={K}$ nodes, $M={MK}$ sensors each, {NSAMPLES} samples, $\\mathrm{{SNR}}={SNR}$ dB, $\\mathrm{{SNR}}_{{\\mathrm{{self}}}}={SNSNR}$ dB, $\\mathrm{{GEVD}}={GEVD}$')
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
    for _ in range(K):
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
    if MODE == 'batch':
        s = np.concatenate(tuple(
            wasn.nodes[k].desiredOnly
            for k in range(K)
        ), axis=0)
        n = np.concatenate(tuple(
            wasn.nodes[k].noiseOnly
            for k in range(K)
        ), axis=0)
        Rss = s @ s.T.conj()
        Rnn = n @ n.T.conj()
        wCentral = filter_update(Rss + Rnn, Rnn, gevd=GEVD)
        mmseCentral = np.zeros(K)
        for k in range(K):
            ek = np.zeros(K * MK)
            ek[k * MK + REFSENSORIDX] = 1
            mmseCentral[k] = np.mean(
                np.abs((wCentral @ ek).T.conj() @ (s + n) -\
                    wasn.nodes[k].desiredOnly[REFSENSORIDX, :]) ** 2
            )
    elif MODE == 'online':
        raise NotImplementedError  # TODO: implement centralized cost for online processing
    
    return mmseCentral


def filter_update(Ryy, Rnn, gevd=False, rank=1):
    """Update filter using GEVD-MWF or MWF."""
    if gevd:
        s, Xmat = sla.eigh(Ryy, Rnn)
        idx = np.flip(np.argsort(s))
        s = s[idx]
        Xmat = Xmat[:, idx]
        Qmat = np.linalg.inv(Xmat.T.conj())
        Dmat = np.zeros_like(Ryy)
        for r in range(rank):
            Dmat[r, r] = 1 - 1 / s[r]
        return Xmat @ Dmat @ Qmat.T.conj()
    else:
        return np.linalg.inv(Ryy) @ (Ryy - Rnn)


if __name__ == '__main__':
    sys.exit(main())
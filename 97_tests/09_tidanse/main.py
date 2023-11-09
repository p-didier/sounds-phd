# Purpose of script:
# TI-DANSE trial from scratch.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import numpy as np
import scipy.signal as sig
import scipy.linalg as sla
import matplotlib.pyplot as plt

SEED = 0
REFSENSORIDX = 0  # Reference sensor index (for desired signal)
FS = 16000
NSAMPLES_TOT = 10000  # Total number of samples for batch processing
NSAMPLES_TOT_ONLINE = 1000000  # Total number of samples for online processing
#
K = 2  # Number of nodes
MK = 1 # Number of microphones per node (same for all nodes)
# K = 3  # Number of nodes
# MK = 10 # Number of microphones per node (same for all nodes)
# K = 3  # Number of nodes
# MK = 5 # Number of microphones per node (same for all nodes)
# ---- Online processing
B = 100  # Block size (number of samples per block)
OVERLAP_B = 0  # Overlap between blocks (in percentage)
BETA = 0.98  # Forgetting factor for online covariance matrix
#
N_STFT = 1024  # STFT window length
N_NOISE_SOURCES = 1  # Number of noise sources
EPS = 1e-4  # Stopping criterion constant
MAXITER = 2000  # Maximum number of iterations
SNR = 10  # [dB] SNR of desired signal
SNSNR = 5  # [dB] SNR of self-noise signals
#
# ALGOS = ['danse', 'ti-danse']  # 'danse' or 'ti-danse'
ALGOS = ['ti-danse']  # 'danse' or 'ti-danse'
# ALGOS = ['danse']  # 'danse' or 'ti-danse'
# MODE = 'batch'  # 'wola' or 'online' or 'batch'
MODE = 'online'  # 'wola' or 'online' or 'batch'
# GEVD = True  # Use GEVD-MWF
GEVD = False  # Use MWF
# NU = 'sim'  # Node-updating strategy: 'sim' or 'seq'
NU = 'seq'  # Node-updating strategy: 'sim' or 'seq'
# ----- External filter relaxation for simultaneous node-updating
BETAEXT = 0.7  # Forgetting factor for external filter relaxation
ALPHAEXT = 0.5  # External filter relaxation factor
T_EXT = 0.4  # Update external filter every `TEXT` seconds
# BETAEXT = 0.  # Forgetting factor for external filter relaxation
# ALPHAEXT = 0.  # External filter relaxation factor
# T_EXT = 0.  # Update external filter every `TEXT` seconds
# ----- Tests / debug
NORM_GK_EVERY = 1  # Normalize `gk` coefficient every `NORM_GK_EVERY` iterations
A = 1#/100  # test normalization factor
# ----- Plot booleans
# PLOT_ONLY_COST = True  # Plot only cost (no per-node MMSE)
PLOT_ONLY_COST = False  # Plot only cost (no per-node MMSE)

# ----- Checks to ensure correct parameters combinations
if K > 10:
    PLOT_ONLY_COST = True

# Set random seed
np.random.seed(SEED)
RNG_STATE = np.random.get_state()

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
    plot_results(mmsePerAlgo, mmseCentral, onlyLoss=PLOT_ONLY_COST)


def batch_or_online_run(wasn: WASN):
    # Compute centralized cost
    mmseCentral = get_centr_cost(wasn)

    mmsePerAlgo = [[] for _ in range(len(ALGOS))]
    eta_desired = [np.array([]) for _ in range(K)]
    eta_noise = [np.array([]) for _ in range(K)]
    for algo in ALGOS:
        if algo == 'danse':
            dimTilde = MK + K - 1  # fully connected
        else:
            dimTilde = MK + 1
        e = np.zeros(dimTilde)
        e[REFSENSORIDX] = 1  # reference sensor selection vector
        wTilde = [np.ones(dimTilde) for _ in range(K)]
        wTildeExt = copy.deepcopy(wTilde)
        wTildeExtTarget = copy.deepcopy(wTilde)
        np.random.set_state(RNG_STATE)
        if MODE == 'online':
            singleSCM = np.random.randn(dimTilde, dimTilde)
            Rss = [copy.deepcopy(singleSCM) for _ in range(K)]
            singleSCM = np.random.randn(dimTilde, dimTilde)
            Rnn = [copy.deepcopy(singleSCM) for _ in range(K)]
        i = 0  # DANSE iteration index
        q = 0  # currently updating node index
        mmse = [[] for _ in range(K)]  # MMSE
        stopcond = False
        wTildeSaved = [[] for _ in range(K)]
        wTildeExtSaved = [[] for _ in range(K)]
        while not stopcond:
            # Compute compressed signals
            if MODE == 'batch':
                z_desired = np.zeros((K, NSAMPLES_TOT))
                z_noise = np.zeros((K, NSAMPLES_TOT))
                indices = np.arange(0, NSAMPLES_TOT)
            elif MODE == 'online':
                z_desired = np.zeros((K, B))
                z_noise = np.zeros((K, B))
                idxBegFrame = int(i * B * (1 - OVERLAP_B)) % NSAMPLES_TOT_ONLINE
                idxEndFrame = int(idxBegFrame + B) % NSAMPLES_TOT_ONLINE
                if idxEndFrame < idxBegFrame:
                    indices = np.concatenate((
                        np.arange(idxBegFrame, NSAMPLES_TOT_ONLINE),
                        np.arange(0, idxEndFrame)
                    ))
                else:
                    indices = np.arange(idxBegFrame, idxEndFrame)
            for k in range(K):
                if MODE == 'batch':
                    sk = wasn.nodes[k].desiredOnly[:, :NSAMPLES_TOT]
                    nk = wasn.nodes[k].noiseOnly[:, :NSAMPLES_TOT]
                elif MODE == 'online':
                    sk = wasn.nodes[k].desiredOnly[:, indices]
                    nk = wasn.nodes[k].noiseOnly[:, indices]
                
                z_desired[k, :], z_noise[k, :] = get_compressed_signals(
                    sk, nk, algo, wTildeExt[k]
                )

            # Compute `sTilde` and `nTilde`
            sTilde, nTilde = get_tildes(algo, z_desired, z_noise, wasn, indices)

            if algo == 'ti-danse' and MODE == 'online':
                # Save `eta` in one long file
                for k in range(K):
                    eta_desired[k] = np.concatenate((eta_desired[k], sTilde[k][-1, :].T))
                    eta_noise[k] = np.concatenate((eta_noise[k], nTilde[k][-1, :].T))

            # Update covariance matrices
            if MODE == 'batch':
                Rss = sTilde[u] @ sTilde[u].T.conj()
                Rnn = nTilde[u] @ nTilde[u].T.conj()
            elif MODE == 'online':
                for k in range(K):
                    Rss[k] = BETA * Rss[k] + (1 - BETA) * sTilde[k] @ sTilde[k].T.conj()
                    Rnn[k] = BETA * Rnn[k] + (1 - BETA) * nTilde[k] @ nTilde[k].T.conj()

            # Perform filter updates
            if NU == 'seq':
                upNodes = np.array([q])
            elif NU == 'sim':
                upNodes = np.arange(K)
            for u in upNodes:
                if MODE == 'batch':
                    # Update MMSE filter
                    if GEVD and not check_matrix_validity(Rss + Rnn, Rnn):
                        print(f"i={i} [batch] -- Warning: matrices are not valid for GEVD-based filter update.")
                    else:
                        wTilde[u] = filter_update(Rss + Rnn, Rnn, gevd=GEVD) @ e
                elif MODE == 'online':
                    # Update MMSE filter
                    if GEVD and not check_matrix_validity(Rss[u] + Rnn[u], Rnn[u]):
                        print(f"i={i} [online] -- Warning: matrices are not valid for GEVD-based filter update.")
                    else:
                        wTilde[u] = filter_update(Rss[u] + Rnn[u], Rnn[u], gevd=GEVD) @ e


            # if i % 100 == 0:
            A = np.linalg.norm(wTilde[0][-1])
            # A = 879
            # else:
            # A = 1.0

            # Update external filters
            wTildeExt = copy.deepcopy(wTilde)  # default (used, e.g., if `NU == 'seq'`)
            # if i % 2 == 1:
            for k in range(K):
                wTildeExt[k][-1] /= A  # normalize `gk` coefficient

            for u in upNodes:
                if NU == 'sim':
                    if (int(T_EXT * FS / B) != 0 and i % int(T_EXT * FS / B) == 0) or\
                        (int(T_EXT * FS / B) == 0):
                        # Update target every `TEXT` seconds
                        wTildeExtTarget[u] = ALPHAEXT * wTildeExtTarget[u] + (1 - ALPHAEXT) * wTilde[u]
                    wTildeExt[u] = BETAEXT * wTildeExt[u] + (1 - BETAEXT) * wTildeExtTarget[u]

            # Normalize to avoid divergence in TI-DANSE
            if algo == 'ti-danse':# and i % 2 == 1:
                if NU == 'sim':
                    raise NotImplementedError('The normalization (to avoid divergence) of TI-DANSE coefficient is not implemented for simultaneous node-updating.')
                elif NU == 'seq':# and i % NORM_GK_EVERY == 0:
                    for k in range(K):
                        # # Normalize all gk's with respect to `upNodes[0]`'s (== `q`'s)
                        # wTilde[k][-1] /= np.linalg.norm(wTilde[q][-1])
                        # wTilde[k][-1] *= 0.5 * (i + 1)
                        # sTilde[k][-1, :] /= np.linalg.norm(sTilde[q][-1, :])
                        # nTilde[k][-1, :] /= np.linalg.norm(nTilde[q][-1, :])
                        # if k in upNodes:
                        #     wTilde[k][-1] /= A ** i
                        # else:
                        wTilde[k][-1] /= A
                        # pass

            # Compute MMSE estimate of desired signal at each node
            mmses = get_mmse(wTilde, sTilde, nTilde, wasn, indices)
            for k in range(K):
                mmse[k].append(mmses[k])
            
            # Print progress
            print(f"[{algo.upper()} {MODE} {NU}] i = {i}, u = {upNodes}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}")
            # Update indices
            i += 1
            q = (q + 1) % K
            stopcond = update_stop_condition(i, mmse)

            # Save `wTilde` and `wTildeExt`
            for k in range(K):
                wTildeSaved[k].append(wTilde[k])
                wTildeExtSaved[k].append(wTildeExt[k])
            
        # Store MMSE
        mmsePerAlgo[ALGOS.index(algo)] = mmse

        if algo == 'ti-danse' and MODE == 'online':
            fig, axes = plt.subplots(1, K, sharey=True, sharex=True)
            fig.set_size_inches(8.5, 3.5)
            for k in range(K):
                axes[k].semilogy(np.abs(np.array(eta_desired[k]).flatten()))
                axes[k].grid()
                axes[k].set_title(f'$\\eta_{{-k}}$ Node {k}')
            fig.tight_layout()
            plt.show()

        if K * MK < 20:
            # Plot network-wide filters
            fig, axes = plt.subplots(2, K, sharey=True, sharex=True)
            fig.set_size_inches(8.5, 3.5)
            for k in range(K):
                # TI-DANSE coefficients
                for m in range(MK + 1):
                    if m < MK:
                        lab = f'$w_{{kk,{m}}}$'
                    else:
                        lab = '$g_k$'
                    axes[0, k].semilogy(
                        np.array(wTildeSaved[k])[:, m],
                        label=lab
                    )
                axes[0, k].grid()
                axes[0, k].legend(loc='upper right')
                axes[0, k].set_xlabel('Iteration index')
                axes[0, k].set_ylabel('TI-DANSE coefficient')
                axes[0, k].set_title(f'Node {k}')
                # Network-wide filter coefficients
                counter = np.zeros(K, dtype=int)
                for m in range(MK * K):
                    if m // MK == k:
                        axes[1, k].semilogy(
                            np.array(wTildeSaved[k])[1:, counter[k]],
                            label=f'$w_{{kk,{counter[k]}}}$'
                        )
                        counter[k] += 1
                    else:
                        neigIdx = m // MK
                        wqq = np.array(wTildeExtSaved[neigIdx])[:-1, counter[neigIdx]]
                        gq = np.array(wTildeExtSaved[neigIdx])[:-1, -1]
                        if algo == 'ti-danse':
                            nwFilt = wqq / gq * np.array(wTildeSaved[k])[1:, -1]
                        elif algo == 'danse':
                            nwFilt = wqq * np.array(wTildeSaved[k])[1:, -1]
                        axes[1, k].semilogy(
                            nwFilt,
                            '--',
                            label=f'$w_{{{neigIdx}{neigIdx},{counter[neigIdx]}}}^{{i-1}}(g_{{{neigIdx}}}^{{i-1}})^{{-1}}g_k^i$'
                        )
                        counter[neigIdx] += 1
                axes[1, k].grid()
                axes[1, k].legend(loc='upper right')
                axes[1, k].set_xlabel('Iteration index')
                axes[1, k].set_ylabel('Net.-wide coefficient')
                axes[1, k].set_title(f'Node {k}')
            fig.tight_layout()
            plt.show()
    
    return mmsePerAlgo, mmseCentral


def update_stop_condition(i, mmse):
    """Stop condition for DANSE `while`-loops."""
    if i > K:
        return i >= MAXITER or\
            any([np.isnan(me[-1]) for me in mmse]) or\
            np.all([
                np.abs(me[-1] - me[-1-K]) / np.abs(me[-1-K]) < EPS for me in mmse
            ])
    else:
        return False


def check_matrix_validity(Ryy, Rnn):
    """Check if `Ryy` is valid for GEVD-based filter updates."""
    def _is_hermitian_and_posdef(x):
        """Finds out whether 3D complex matrix `x` is Hermitian along 
        the two last axes, as well as positive definite."""
        # Get rid of machine-precision residual imaginary parts
        x = np.real_if_close(x)
        # Assess Hermitian-ness
        b1 = np.allclose(x.T.conj(), x)
        # Assess positive-definiteness
        b2 = not any(np.linalg.eigvalsh(x) < 0)
        return b1 and b2
    def __full_rank_check(mat):
        """Helper subfunction: check full-rank property."""
        return (np.linalg.matrix_rank(mat) == mat.shape[-1]).all()
    check1 = _is_hermitian_and_posdef(Rnn)
    check2 = _is_hermitian_and_posdef(Ryy)
    check3 = __full_rank_check(Rnn)
    check4 = __full_rank_check(Ryy)
    return check1 and check2 and check3 and check4


def get_compressed_signals(sk, nk, algo, wkEXT):
    """Compute compressed signals using the given desired source-only and
    noise-only data."""
    if algo == 'danse':
        pk = wkEXT[:MK]  # DANSE fusion vector
    elif algo == 'ti-danse':
        pk = wkEXT[:MK] / wkEXT[-1]  # TI-DANSE fusion vector
    # Inner product of `pk` and `yk` across channels (fused signals)
    zk_desired = np.sum(sk * pk[:, np.newaxis], axis=0)
    zk_noise = np.sum(nk * pk[:, np.newaxis], axis=0)
    return zk_desired, zk_noise


def get_mmse(wTilde, sTilde, nTilde, wasn: WASN, indices):
    """Compute MMSE."""
    currMMSEs = np.zeros(K)
    for k in range(K):
        dHat = wTilde[k] @ (sTilde[k] + nTilde[k])
        currMMSEs[k] = np.mean(np.abs(
            dHat - wasn.nodes[k].desiredOnly[REFSENSORIDX, indices]
        ) ** 2)
    return currMMSEs


def get_tildes(algo, z_desired, z_noise, wasn: WASN, indices):
    """Compute `sTilde` and `nTilde`."""
    sTilde = [_ for _ in range(K)]
    nTilde = [_ for _ in range(K)]
    for k in range(K):
        # Local signals
        xk = wasn.nodes[k].desiredOnly[:, indices]
        nk = wasn.nodes[k].noiseOnly[:, indices]
        # $z_{-k}$ compressed signal vectors
        zMk_desired = z_desired[np.arange(K) != k, :]
        zMk_noise = z_noise[np.arange(K) != k, :]
        if algo == 'danse':
            sTilde[k] = np.concatenate((xk, zMk_desired), axis=0)
            nTilde[k] = np.concatenate((nk, zMk_noise), axis=0)
        elif algo == 'ti-danse':
            # vvv `sum(zMk_desired)` == $\eta_{-k}$ vvv
            etaMk_desired = np.sum(zMk_desired, axis=0)[np.newaxis, :]
            etaMk_noise = np.sum(zMk_noise, axis=0)[np.newaxis, :]
            sTilde[k] = np.concatenate((xk, etaMk_desired), axis=0)
            nTilde[k] = np.concatenate((nk, etaMk_noise), axis=0)
    
    return sTilde, nTilde


def plot_results(mmsePerAlgo, mmseCentral, onlyLoss=False):
    """Plot results."""
    if onlyLoss:
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(6.5, 3.5)
        for idxAlgo in range(len(ALGOS)):
            if MODE == 'batch':
                xmax = len(mmsePerAlgo[idxAlgo][0])-1
            elif MODE == 'online':
                xmax = np.amax([len(mmsePerAlgo[idxAlgo][0])-1, len(mmseCentral[0])-1])
            data = np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0)
            axes.loglog(
                data,
                f'-C{idxAlgo}',
                label=f'{ALGOS[idxAlgo].upper()} ({len(data)} iters, $\\mathcal{{L}}=${"{:.3g}".format(data[-1], -4)})'
            )
        if MODE == 'batch':
            axes.hlines(np.mean(mmseCentral), 0, xmax, 'k', linestyle="--", label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(mmseCentral), -4)})')
        elif MODE == 'online':
            axes.loglog(np.mean(mmseCentral, axis=0), '--k', label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(mmseCentral, axis=0)[-1], -4)})')
        axes.set_xlabel("Iteration index")
        axes.set_ylabel("Cost")
        axes.legend(loc='upper right')
        axes.set_xlim([0, xmax])
        axes.grid()
        ti_str = f'$K={K}$, $M={MK}$ mics/node, $\\mathrm{{SNR}}={SNR}$ dB, $\\mathrm{{SNR}}_{{\\mathrm{{self}}}}={SNSNR}$ dB, $\\mathrm{{GEVD}}={GEVD}$'
        if MODE == 'online':
            ti_str += f', $B={B}$ ({int(OVERLAP_B * 100)}%ovlp), $\\beta={BETA}$'
        elif MODE == 'batch':
            ti_str += f', {NSAMPLES_TOT} samples'
        axes.set_title(ti_str)
    else:
        fig, axes = plt.subplots(2, len(ALGOS))
        fig.set_size_inches(8.5, 3.5)
        for idxAlgo in range(len(ALGOS)):
            if len(ALGOS) == 1:
                currAx = axes
            else:
                currAx = axes[:, idxAlgo]
            if MODE == 'batch':
                xmax = len(mmsePerAlgo[idxAlgo][0])-1
            elif MODE == 'online':
                xmax = np.amax([len(mmsePerAlgo[idxAlgo][0])-1, len(mmseCentral[0])-1])
            for k in range(K):
                currAx[0].loglog(mmsePerAlgo[idxAlgo][k], f'-C{k}', label=f"Node {k}")
                if MODE == 'batch':
                    currAx[0].hlines(mmseCentral[k], 0, xmax, f'C{k}', linestyle="--")
                elif MODE == 'online':
                    currAx[0].loglog(mmseCentral[k, :], f'--C{k}')
            currAx[0].set_xlabel(f"{ALGOS[idxAlgo].upper()} iteration index")
            currAx[0].set_ylabel("MMSE per node")
            currAx[0].legend(loc='upper right')
            currAx[0].set_xlim([0, xmax])
            currAx[0].grid()
            #
            currAx[1].loglog(np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0), '-k', label=f'{ALGOS[idxAlgo].upper()} ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(np.array(mmsePerAlgo[idxAlgo]), axis=0)[-1], -4)})')
            if MODE == 'batch':
                currAx[1].hlines(np.mean(mmseCentral), 0, xmax, 'k', linestyle="--", label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(mmseCentral), -4)})')
            elif MODE == 'online':
                currAx[1].loglog(np.mean(mmseCentral, axis=0), '--k', label=f'Centralized ($\\mathcal{{L}}=${"{:.3g}".format(np.mean(mmseCentral, axis=0)[-1], -4)})')
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
    desired = np.random.randn(NSAMPLES_TOT_ONLINE, 1)

    # Generate noise signals (random)
    noise = np.random.randn(NSAMPLES_TOT_ONLINE, N_NOISE_SOURCES)

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
        # noiseAtMic = mixingMatrix @ noise.T +\
        #     np.random.randn(MK, NSAMPLES_TOT_ONLINE) * 10 ** (-SNSNR / 20)
        # noiseAtMic *= 10 ** (-SNR / 20)
        noiseAtMic = np.random.randn(MK, NSAMPLES_TOT_ONLINE) * 10 ** (-SNR / 20)
        n.append(noiseAtMic)

    return x, n


def get_centr_cost(wasn: WASN):
    """Compute centralized cost (MMSE) for each node."""

    # Full observation matrices
    s = np.concatenate(tuple(wasn.nodes[k].desiredOnly for k in range(K)), axis=0)
    n = np.concatenate(tuple(wasn.nodes[k].noiseOnly for k in range(K)), axis=0)
    nSensors = K * MK

    if MODE == 'batch':
        Rss = s @ s.T.conj()
        Rnn = n @ n.T.conj()
        wCentral = filter_update(Rss + Rnn, Rnn, gevd=GEVD)
        mmseCentral = np.zeros(K)
        for k in range(K):
            ek = np.zeros(nSensors)
            ek[k * MK + REFSENSORIDX] = 1
            mmseCentral[k] = np.mean(
                np.abs((wCentral @ ek).T.conj() @ (s + n) -\
                    wasn.nodes[k].desiredOnly[REFSENSORIDX, :]) ** 2
            )
    elif MODE == 'online':
        np.random.set_state(RNG_STATE)
        singleSCM = np.random.randn(nSensors, nSensors)
        Rss = copy.deepcopy(singleSCM)
        Rnn = copy.deepcopy(singleSCM)
        mmseCentral = [[] for _ in range(K)]
        wCentral = np.zeros((nSensors, nSensors))
        stopcond = False
        i = 0
        while not stopcond:
            idxBegFrame = int(i * B * (1 - OVERLAP_B)) % NSAMPLES_TOT_ONLINE
            idxEndFrame = int(idxBegFrame + B) % NSAMPLES_TOT_ONLINE
            if idxEndFrame < idxBegFrame:
                indices = np.concatenate((
                    np.arange(idxBegFrame, NSAMPLES_TOT_ONLINE),
                    np.arange(0, idxEndFrame)
                ))
            else:
                indices = np.arange(idxBegFrame, idxEndFrame)
            sCurr = s[:, indices]
            nCurr = n[:, indices]
            Rss = BETA * Rss + (1 - BETA) * sCurr @ sCurr.T.conj()
            Rnn = BETA * Rnn + (1 - BETA) * nCurr @ nCurr.T.conj()
            if GEVD and not check_matrix_validity(Rss + Rnn, Rnn):
                print(f"i={i} [centr online] -- Warning: matrices are not valid for GEVD-based filter update.")
            else:
                wCentral = filter_update(Rss + Rnn, Rnn, gevd=GEVD)
            for k in range(K):
                ek = np.zeros(nSensors)
                ek[k * MK + REFSENSORIDX] = 1
                mmseCentral[k].append(np.mean(
                    np.abs((wCentral @ ek).T.conj() @ (sCurr + nCurr) -\
                        wasn.nodes[k].desiredOnly[REFSENSORIDX, indices]) ** 2
                ))
            print(f"[Centr. {MODE}] i = {i}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmseCentral]), -4)}")
            i += 1
            stopcond = update_stop_condition(i, mmseCentral)
        mmseCentral = np.array(mmseCentral)
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
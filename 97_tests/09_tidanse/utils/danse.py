import copy
import numpy as np
import scipy.linalg as sla
from .scene import SceneCreator, WASN
import matplotlib.pyplot as plt

class Launcher:
    """Class to launch DANSE simulations."""
    def __init__(self, scene: SceneCreator):
        self.cfg = scene.cfg
        self.wasn = scene.wasn
        self.mmsePerAlgo = None
        self.mmseCentral = None
    
    def run(self):
        """Run simulation."""
        if self.cfg.mode == 'wola':
            # Compute STFT signals
            self.wasn.compute_stft_signals()
            raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
        elif self.cfg.mode in ['batch', 'online']:
            self.mmsePerAlgo, self.mmseCentral = self.batch_or_online_run()

    def batch_or_online_run(self):
        # Compute centralized cost
        mmseCentral = self.get_centr_cost()

        mmsePerAlgo = [[] for _ in range(len(self.cfg.algos))]
        for algo in self.cfg.algos:
            # Initialize DANSE variables
            dimTilde = self.cfg.Mk + self.cfg.K - 1 if algo == 'danse' else self.cfg.Mk + 1
            e = np.zeros(dimTilde)
            e[self.cfg.refSensorIdx] = 1  # reference sensor selection vector
            wTilde = [np.ones(dimTilde) for _ in range(self.cfg.K)]
            wTildeExt = copy.deepcopy(wTilde)
            # np.random.set_state(self.cfg.rngState)  # reset RNG state

            if self.cfg.mode == 'online':
                singleSCM = np.random.randn(dimTilde, dimTilde)
                Rss = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
                singleSCM = np.random.randn(dimTilde, dimTilde)
                Rnn = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            i = 0  # DANSE iteration index
            q = 0  # currently updating node index
            nf = 1  # normalization factor
            mmse = [[] for _ in range(self.cfg.K)]  # MMSE
            wTildeSaved = [[wTilde[k]] for k in range(self.cfg.K)]
            wTildeExtSaved = [[wTildeExt[k]] for k in range(self.cfg.K)]
            avgAmpEtaMk = [[] for _ in range(self.cfg.K)]
            normFact = []
            nIterSinceLastUp = [0 for _ in range(self.cfg.K)]
            stopcond = False
            while not stopcond:
                # Compute compressed signals
                if self.cfg.mode == 'batch':
                    z_desired = np.zeros((self.cfg.K, self.cfg.nSamplesTot))
                    z_noise = np.zeros((self.cfg.K, self.cfg.nSamplesTot))
                    indices = np.arange(0, self.cfg.nSamplesTot)
                elif self.cfg.mode == 'online':
                    z_desired = np.zeros((self.cfg.K, self.cfg.B))
                    z_noise = np.zeros((self.cfg.K, self.cfg.B))
                    idxBegFrame = int(i * self.cfg.B * (1 - self.cfg.overlapB))\
                        % self.cfg.nSamplesTotOnline
                    idxEndFrame = int(idxBegFrame + self.cfg.B)\
                        % self.cfg.nSamplesTotOnline
                    if idxEndFrame < idxBegFrame:
                        indices = np.concatenate((
                            np.arange(idxBegFrame, self.cfg.nSamplesTotOnline),
                            np.arange(0, idxEndFrame)
                        ))
                    else:
                        indices = np.arange(idxBegFrame, idxEndFrame)
                for k in range(self.cfg.K):
                    sk = self.wasn.nodes[k].desiredOnly[:, indices]
                    nk = self.wasn.nodes[k].noiseOnly[:, indices]
                    z_desired[k, :], z_noise[k, :] = get_compressed_signals(
                        sk, nk, algo, wTildeExt[k],
                        onlyWkk=(
                            (i < self.cfg.K * (self.cfg.nIterBetweenUpdates + 1)\
                            if self.cfg.nodeUpdating == 'seq' else i == 0)
                        ) if self.cfg.mode == 'online' else False
                        # ^^^ in batch-mode, always use `wkk/gk` for TI-DANSE
                    )

                # Compute `sTilde` and `nTilde`
                sTilde, nTilde = get_tildes(
                    algo, z_desired, z_noise, self.wasn, indices
                )

                # Normalize `sTilde` and `nTilde` (TI-DANSE only)
                if algo == 'ti-danse' and self.cfg.mode == 'online':
                    betaNf = np.amin([1 - self.cfg.gamma ** i, self.cfg.maxBetaNf])  # slowly increase `betaNf` from 0 towards 0.75
                    nfCurr = np.mean(np.abs(np.sum(z_desired + z_noise, axis=0)))#*\
                    nf = betaNf * nf + (1 - betaNf) * nfCurr / 1e6
                    for k in range(self.cfg.K):
                        if self.cfg.nodeUpdating == 'sim':
                            raise NotImplementedError('The normalization (to avoid divergence) of TI-DANSE coefficient is not implemented for simultaneous node-updating.')
                        elif self.cfg.nodeUpdating == 'seq':
                            sTilde[k][-1, :] /= nf
                            nTilde[k][-1, :] /= nf
                    normFact.append(nf)

                # Update covariance matrices
                if self.cfg.nodeUpdating == 'seq':
                    if nIterSinceLastUp[q] >= self.cfg.nIterBetweenUpdates:
                        upNodes = np.array([q])
                        nIterSinceLastUp[q] = 0
                    else:
                        upNodes = np.array([])
                        nIterSinceLastUp[q] += 1
                elif self.cfg.nodeUpdating == 'sim':
                    upNodes = np.arange(self.cfg.K)
                if self.cfg.mode == 'batch':
                    Rss = sTilde[q] @ sTilde[q].T.conj()
                    Rnn = nTilde[q] @ nTilde[q].T.conj()
                elif self.cfg.mode == 'online':
                    for k in range(self.cfg.K):
                        Rss[k] = self.cfg.beta * Rss[k] +\
                            (1 - self.cfg.beta) * sTilde[k] @ sTilde[k].T.conj()
                        Rnn[k] = self.cfg.beta * Rnn[k] +\
                            (1 - self.cfg.beta) * nTilde[k] @ nTilde[k].T.conj()

                # Perform filter updates
                for u in upNodes:
                    if self.cfg.mode == 'batch':
                        if self.cfg.gevd and not check_matrix_validity(Rss + Rnn, Rnn):
                            print(f"i={i} [batch] -- Warning: matrices are not valid for self.cfg.gevd-based filter update.")
                        else:
                            wTilde[u] = filter_update(
                                Rss + Rnn, Rnn, gevd=self.cfg.gevd
                            ) @ e
                    elif self.cfg.mode == 'online':
                        if self.cfg.gevd and not check_matrix_validity(Rss[u] + Rnn[u], Rnn[u]):
                            print(f"i={i} [online] -- Warning: matrices are not valid for self.cfg.gevd-based filter update.")
                        else:
                            wTilde[u] = filter_update(
                                Rss[u] + Rnn[u], Rnn[u], gevd=self.cfg.gevd
                            ) @ e

                    # Update external filters
                    wTildeExt[u] = copy.deepcopy(wTilde[u])  # default (used, e.g., if `self.cfg.nodeUpdating == 'seq'`)

                # Compute MMSE estimate of desired signal at each node
                mmses = get_mmse(wTilde, sTilde, nTilde, self.wasn, indices)
                for k in range(self.cfg.K):
                    mmse[k].append(mmses[k])
                
                # Print progress
                print(f"[{algo.upper()} {self.cfg.mode} {self.cfg.nodeUpdating}] i = {i}, u = {upNodes}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}")
                # Update indices
                i += 1
                q = (q + 1) % self.cfg.K
                if i > 1000:
                    stop = 1
                # Randomly pick node to update
                stopcond = self.update_stop_condition(i, mmse)

                # # Save `wTilde` and `wTildeExt`
                # for k in range(self.cfg.K):
                #     wTildeSaved[k].append(wTilde[k])
                #     wTildeExtSaved[k].append(wTildeExt[k])
                #     avgAmpEtaMk[k].append(np.mean(np.abs(sTilde[k][-1, :])))
                
            # Store MMSE
            mmsePerAlgo[self.cfg.algos.index(algo)] = mmse
        
        return mmsePerAlgo, mmseCentral

    def get_centr_cost(self):
        """Compute centralized cost (MMSE) for each node."""
        # Full observation matrices
        s = np.concatenate(tuple(self.wasn.nodes[k].desiredOnly for k in range(self.cfg.K)), axis=0)
        n = np.concatenate(tuple(self.wasn.nodes[k].noiseOnly for k in range(self.cfg.K)), axis=0)
        nSensors = self.cfg.K * self.cfg.Mk

        if self.cfg.mode == 'batch':
            Rss = s[:, :self.cfg.nSamplesTot] @ s[:, :self.cfg.nSamplesTot].T.conj()
            Rnn = n[:, :self.cfg.nSamplesTot] @ n[:, :self.cfg.nSamplesTot].T.conj()
            wCentral = filter_update(Rss + Rnn, Rnn, gevd=self.cfg.gevd)
            mmseCentral = np.zeros(self.cfg.K)
            for k in range(self.cfg.K):
                ek = np.zeros(nSensors)
                ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                mmseCentral[k] = np.mean(
                    np.abs((wCentral @ ek).T.conj() @ (s + n)[:, :self.cfg.nSamplesTot] -\
                        self.wasn.nodes[k].desiredOnly[self.cfg.refSensorIdx, :self.cfg.nSamplesTot]) ** 2
                )
        elif self.cfg.mode == 'online':
            np.random.set_state(self.cfg.rngState)
            singleSCM = np.random.randn(nSensors, nSensors)
            Rss = copy.deepcopy(singleSCM)
            Rnn = copy.deepcopy(singleSCM)
            mmseCentral = [[] for _ in range(self.cfg.K)]
            wCentral = np.zeros((nSensors, nSensors))
            stopcond = False
            i = 0
            while not stopcond:
                idxBegFrame = int(i * self.cfg.B * (1 - self.cfg.overlapB)) % self.cfg.nSamplesTotOnline
                idxEndFrame = int(idxBegFrame + self.cfg.B) % self.cfg.nSamplesTotOnline
                if idxEndFrame < idxBegFrame:
                    indices = np.concatenate((
                        np.arange(idxBegFrame, self.cfg.nSamplesTotOnline),
                        np.arange(0, idxEndFrame)
                    ))
                else:
                    indices = np.arange(idxBegFrame, idxEndFrame)
                sCurr = s[:, indices]
                nCurr = n[:, indices]
                Rss = self.cfg.beta * Rss + (1 - self.cfg.beta) * sCurr @ sCurr.T.conj()
                Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * nCurr @ nCurr.T.conj()
                if self.cfg.gevd and not check_matrix_validity(Rss + Rnn, Rnn):
                    print(f"i={i} [centr online] -- Warning: matrices are not valid for self.cfg.gevd-based filter update.")
                else:
                    wCentral = filter_update(Rss + Rnn, Rnn, gevd=self.cfg.gevd)
                for k in range(self.cfg.K):
                    ek = np.zeros(nSensors)
                    ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                    mmseCentral[k].append(np.mean(
                        np.abs((wCentral @ ek).T.conj() @ (sCurr + nCurr) -\
                            self.wasn.nodes[k].desiredOnly[self.cfg.refSensorIdx, indices]) ** 2
                    ))
                print(f"[Centr. {self.cfg.mode}] i = {i}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmseCentral]), -4)}")
                i += 1
                stopcond = self.update_stop_condition(i, mmseCentral)
            mmseCentral = np.array(mmseCentral)
        return mmseCentral
    
    def update_stop_condition(self, i, mmse):
        """Stop condition for DANSE `while`-loops."""
        if i > self.cfg.K:
            return i >= self.cfg.maxIter or\
                any([np.isnan(me[-1]) for me in mmse]) or\
                np.all([
                    np.abs(
                        me[-1] - me[-1 - self.cfg.K]
                    ) / np.abs(me[-1 - self.cfg.K]) < self.cfg.eps
                    for me in mmse
                ])
        else:
            return False

def get_mmse(wTilde, sTilde, nTilde, wasn: WASN, indices):
    """Compute MMSE."""
    currMMSEs = np.zeros(len(wTilde))
    for k in range(len(wTilde)):
        dHat = wTilde[k] @ (sTilde[k] + nTilde[k])
        currMMSEs[k] = np.mean(np.abs(
            dHat - wasn.nodes[k].desiredOnly[wasn.refSensorIdx, indices]
        ) ** 2)
    return currMMSEs


def get_tildes(algo, z_desired, z_noise, wasn: WASN, indices):
    """Compute `sTilde` and `nTilde`."""
    nNodes = len(wasn.nodes)
    sTilde = [_ for _ in range(nNodes)]
    nTilde = [_ for _ in range(nNodes)]
    for k in range(nNodes):
        # Local signals
        xk = wasn.nodes[k].desiredOnly[:, indices]
        nk = wasn.nodes[k].noiseOnly[:, indices]
        # $z_{-k}$ compressed signal vectors
        zMk_desired = z_desired[np.arange(nNodes) != k, :]
        zMk_noise = z_noise[np.arange(nNodes) != k, :]
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


def get_compressed_signals(sk, nk, algo, wkEXT, onlyWkk=False):
    """Compute compressed signals using the given desired source-only and
    noise-only data."""
    Mk = sk.shape[0]
    if algo == 'danse':
        pk = wkEXT[:Mk]  # DANSE fusion vector
    elif algo == 'ti-danse':
        if onlyWkk:
            pk = wkEXT[:Mk]
        else:
            pk = wkEXT[:Mk] / wkEXT[-1]  # TI-DANSE fusion vector
    # Inner product of `pk` and `yk` across channels (fused signals)
    zk_desired = np.sum(sk * pk[:, np.newaxis], axis=0)
    zk_noise = np.sum(nk * pk[:, np.newaxis], axis=0)
    return zk_desired, zk_noise


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
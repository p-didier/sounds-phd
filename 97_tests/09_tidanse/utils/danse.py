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

    def query(self, k):
        return self.wasn.nodes[k].query(
            Mk=self.cfg.Mk,
            B=self.cfg.B,
            nNoiseSources=self.cfg.nNoiseSources,
            snr=self.cfg.snr,
            snSnr=self.cfg.snSnr
        )

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

            if self.cfg.mode == 'online':
                singleSCM = np.random.randn(dimTilde, dimTilde)
                Rss = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
                singleSCM = np.random.randn(dimTilde, dimTilde)
                Rnn = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            i = 0  # DANSE iteration index
            q = 0  # currently updating node index
            nf = 1  # normalization factor
            mmse = [[] for _ in range(self.cfg.K)]  # MMSE per node
            nIterSinceLastUp = [0 for _ in range(self.cfg.K)]
            stopcond = False
            while not stopcond:
                # Compute compressed signals
                if self.cfg.mode == 'batch':
                    z_desired = np.zeros((self.cfg.K, self.cfg.nSamplesTot))
                    z_noise = np.zeros((self.cfg.K, self.cfg.nSamplesTot))
                elif self.cfg.mode == 'online':
                    z_desired = np.zeros((self.cfg.K, self.cfg.B))
                    z_noise = np.zeros((self.cfg.K, self.cfg.B))
                targetsMMSE = []
                s, n = [], []
                for k in range(self.cfg.K):
                    if self.cfg.mode == 'batch':
                        sk = self.wasn.nodes[k].desiredOnly
                        nk = self.wasn.nodes[k].noiseOnly
                    elif self.cfg.mode == 'online':
                        sk, nk = self.query(k)  # Query new data
                    # Save the signals for tilde-vectors computation
                    s.append(sk)
                    n.append(nk)
                    # Save the target signal for MMSE computation
                    targetsMMSE.append(sk[self.wasn.refSensorIdx, :])
                    z_desired[k, :], z_noise[k, :] = get_compressed_signals(
                        sk,
                        nk,
                        algo,
                        wTildeExt[k],
                        # onlyWkk=(
                        #     (i < self.cfg.K * (self.cfg.nIterBetweenUpdates + 1)\
                        #     if self.cfg.nodeUpdating == 'seq' else i == 0)
                        # ) if self.cfg.mode == 'online' else False
                        # # ^^^ in batch-mode, always use `wkk/gk` for TI-DANSE
                    )

                # Compute `sTilde` and `nTilde`
                sTilde, nTilde = self.get_tildes(algo, s, n, z_desired, z_noise)

                # Normalize `sTilde` and `nTilde` (TI-DANSE only)
                if algo == 'ti-danse' and self.cfg.mode == 'online'\
                    and i > 0 and i % self.cfg.normGkEvery == 0:
                    nf = np.mean(np.abs(np.sum(z_desired + z_noise, axis=0)))
                    for k in range(self.cfg.K):
                        if self.cfg.nodeUpdating == 'sim':
                            raise NotImplementedError('The normalization (to avoid divergence) of TI-DANSE coefficient is not implemented for simultaneous node-updating.')
                        elif self.cfg.nodeUpdating == 'seq':
                            sTilde[k][-1, :] /= nf
                            nTilde[k][-1, :] /= nf

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
                mmses = self.get_mmse(
                    wTilde, sTilde, nTilde, targetsMMSE
                )
                for k in range(self.cfg.K):
                    mmse[k].append(mmses[k])
                
                # Print progress
                print(f"[{algo.upper()} {self.cfg.mode} {self.cfg.nodeUpdating}] i = {i}, u = {upNodes}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}")
                # Update indices
                i += 1
                q = (q + 1) % self.cfg.K
                # Randomly pick node to update
                stopcond = self.update_stop_condition(i, mmse)

            # Store MMSE
            mmsePerAlgo[self.cfg.algos.index(algo)] = mmse
        
        return mmsePerAlgo, mmseCentral

    def get_centr_cost(self):
        """Compute centralized cost (MMSE) for each node."""
        # Full observation matrices
        nSensors = self.cfg.K * self.cfg.Mk

        if self.cfg.mode == 'batch':
            s = np.concatenate(tuple(self.wasn.nodes[k].desiredOnly for k in range(self.cfg.K)), axis=0)
            n = np.concatenate(tuple(self.wasn.nodes[k].noiseOnly for k in range(self.cfg.K)), axis=0)
            Rss = s @ s.T.conj()
            Rnn = n @ n.T.conj()
            wCentral = filter_update(Rss + Rnn, Rnn, gevd=self.cfg.gevd)
            mmseCentral = np.zeros(self.cfg.K)
            for k in range(self.cfg.K):
                ek = np.zeros(nSensors)
                ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                mmseCentral[k] = np.mean(
                    np.abs((wCentral @ ek).T.conj() @ (s + n) -\
                        self.wasn.nodes[k].desiredOnly[self.cfg.refSensorIdx, :]) ** 2
                )
        elif self.cfg.mode == 'online':
            singleSCM = np.random.randn(nSensors, nSensors)
            Rss = copy.deepcopy(singleSCM)
            Rnn = copy.deepcopy(singleSCM)
            mmseCentral = [[] for _ in range(self.cfg.K)]
            wCentral = np.zeros((nSensors, nSensors))
            stopcond = False
            i = 0
            while not stopcond:
                sCurr = np.concatenate(tuple(
                    self.query(k)[0] for k in range(self.cfg.K)
                ), axis=0)
                nCurr = np.concatenate(tuple(
                    self.query(k)[1] for k in range(self.cfg.K)
                ), axis=0)
                Rss = self.cfg.beta * Rss + (1 - self.cfg.beta) * sCurr @ sCurr.T.conj()
                Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * nCurr @ nCurr.T.conj()
                if self.cfg.gevd and not check_matrix_validity(Rss + Rnn, Rnn):
                    print(f"i={i} [centr online] -- Warning: matrices are not valid for self.cfg.gevd-based filter update.")
                else:
                    wCentral = filter_update(Rss + Rnn, Rnn, gevd=self.cfg.gevd)
                for k in range(self.cfg.K):
                    ek = np.zeros(nSensors)
                    ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                    target = sCurr.T @ ek
                    mmseCentral[k].append(np.mean(np.abs(
                        (wCentral @ ek).T.conj() @ (sCurr + nCurr) - target
                    ) ** 2))
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
        
    def get_tildes(self, algo, sk_forAllk, nk_forAllk, z_desired, z_noise):
        """Compute `sTilde` and `nTilde`."""
        sTilde = [_ for _ in range(self.cfg.K)]
        nTilde = [_ for _ in range(self.cfg.K)]
        for k in range(self.cfg.K):
            # Local signals
            sk = sk_forAllk[k]
            nk = nk_forAllk[k]
            # $z_{-k}$ compressed signal vectors
            zMk_desired = z_desired[np.arange(self.cfg.K) != k, :]
            zMk_noise = z_noise[np.arange(self.cfg.K) != k, :]
            if algo == 'danse':
                sTilde[k] = np.concatenate((sk, zMk_desired), axis=0)
                nTilde[k] = np.concatenate((nk, zMk_noise), axis=0)
            elif algo == 'ti-danse':
                # vvv `sum(zMk_desired)` == $\eta_{-k}$ vvv
                etaMk_desired = np.sum(zMk_desired, axis=0)[np.newaxis, :]
                etaMk_noise = np.sum(zMk_noise, axis=0)[np.newaxis, :]
                sTilde[k] = np.concatenate((sk, etaMk_desired), axis=0)
                nTilde[k] = np.concatenate((nk, etaMk_noise), axis=0)    
        
        return sTilde, nTilde

    def get_mmse(self, wTilde, sTilde, nTilde, target):
        """
        Compute MMSE.
        
        Parameters
        ----------
        wTilde : list of array-like
            List of external filters.
        sTilde : list of array-like
            List of compressed desired signals.
        nTilde : list of array-like
            List of compressed noise signals.
        target : list of array-like
            Desired signals at each node.
        
        Returns
        -------
        currMMSEs : array-like
            MMSEs at each node.
        """
        currMMSEs = np.zeros(len(wTilde))
        for k in range(len(wTilde)):
            dHat = wTilde[k] @ (sTilde[k] + nTilde[k])
            currMMSEs[k] = np.mean(np.abs(dHat - target[k]) ** 2)
        return currMMSEs

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
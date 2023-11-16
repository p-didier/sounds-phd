import copy
import numpy as np
import scipy.linalg as sla
from .scene import SceneCreator

class Launcher:
    """Class to launch DANSE simulations."""
    def __init__(self, scene: SceneCreator):
        self.cfg = scene.cfg
        self.wasn = scene.wasn
        self.mmsePerAlgo = None
        self.mmseCentral = None
        self.filtersPerAlgo = None
        self.filtersCentral = None
        self.vadSaved = None
        self.y = None  # Current desired signal chunk, for each node (`K`-elements list)
        self.n = None  # Current noise signal chunk, for each node (`K`-elements list)
        self.startFilterUpdates = False
        self.q = 0  # currently updating node index

    def run(self):
        """Run simulation."""
        if self.cfg.mode == 'wola':
            # Compute STFT signals
            self.wasn.compute_stft_signals()
            raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
        elif self.cfg.mode in ['batch', 'online']:
            self.mmsePerAlgo, self.mmseCentral,\
                self.filtersPerAlgo, self.filtersCentral,\
                    self.vadSaved = self.batch_or_online_run()

    def batch_or_online_run(self):
        # Compute centralized cost
        mmseCentr, filterCoeffsCentr = self.get_centr_cost()

        mmsePerAlgo = [[] for _ in range(len(self.cfg.algos))]
        filterCoeffsPerAlgo = [[] for _ in range(len(self.cfg.algos))]
        for algo in self.cfg.algos:
            # Set RNG state
            np.random.set_state(self.cfg.rngStateOriginal)
            # Initialize DANSE variables
            dimTilde = self.cfg.Mk + self.cfg.K - 1 if algo == 'danse' else self.cfg.Mk + 1
            e = np.zeros(dimTilde)
            e[self.cfg.refSensorIdx] = 1  # reference sensor selection vector
            wTilde = [np.ones(dimTilde) for _ in range(self.cfg.K)]
            wTildeExt = copy.deepcopy(wTilde)
            wTildeSaved = [copy.deepcopy(wTilde)]
            vadSaved = []
            # singleSCM = random_posdef_fullrank_matrix(dimTilde)
            singleSCM = np.zeros((dimTilde, dimTilde))
            Ryy = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            # singleSCM = random_posdef_fullrank_matrix(dimTilde)
            singleSCM = np.zeros((dimTilde, dimTilde))
            Rnn = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            i = 0  # DANSE iteration index
            self.q = 0  # currently updating node index
            nf = 1  # normalization factor
            mmse = [[] for _ in range(self.cfg.K)]  # MMSE per node
            nIterSinceLastUp = [0 for _ in range(self.cfg.K)]
            stopcond = False
            self.startFilterUpdates = False
            self.cfg.sigConfig.sampleIdx = 0
            self.nUpRyy, self.nUpRnn = 0, 0
            # DEBUG STUFF
            normFact_zn = []
            normFact_zy = []
            while not stopcond:

                # Get new data
                if self.cfg.mode == 'online':
                    s, self.n = self.wasn.query()
                    self.y = [s[k] + self.n[k] for k in range(self.cfg.K)]
                    z_y = np.zeros((self.cfg.K, self.cfg.B))
                    z_n = np.zeros((self.cfg.K, self.cfg.B))  # used iff `flagVAD == False`
                elif self.cfg.mode == 'batch':
                    self.y = [self.wasn.nodes[k].signal for k in range(self.cfg.K)]
                    self.n = [self.wasn.nodes[k].noiseOnly for k in range(self.cfg.K)]
                    z_y = np.zeros((self.cfg.K, self.cfg.sigConfig.nSamplesBatch))
                    z_n = np.zeros((self.cfg.K, self.cfg.sigConfig.nSamplesBatch))  # used iff `flagVAD == False`
                
                # Compute compressed signals
                for k in range(self.cfg.K):
                    z_y[k, :], z_n[k, :] = get_compressed_signals(
                        self.y[k], self.n[k], algo, wTildeExt[k]
                    )

                # Compute `sTilde` and `nTilde`
                yTilde, nTilde = self.get_tildes(algo, z_y, z_n)

                # Normalize \eta (TI-DANSE only)
                if algo == 'ti-danse' and self.conds_for_ti_danse_norm():
                    # Compute normalization factor
                    nf = np.mean(np.abs(np.sum(z_y, axis=0)))
                    for k in range(self.cfg.K):
                        if self.cfg.nodeUpdating == 'seq':
                            yTilde[k][-1, :] /= nf
                            nTilde[k][-1, :] /= nf
                        elif self.cfg.nodeUpdating == 'sim':
                            raise NotImplementedError('The normalization (to avoid divergence) of TI-DANSE coefficient is not implemented for simultaneous node-updating.')

                # Update covariance matrices
                Ryy, Rnn = self.update_scms(yTilde, nTilde, Ryy, Rnn)

                # Check if filter updates should start
                if not self.startFilterUpdates:
                    self.startFilterUpdates = True  # by default, start filter updates
                    if self.cfg.mode == 'online' and self.cfg.gevd:
                        for k in range(self.cfg.K):
                            if not check_matrix_validity(Ryy[k], Rnn[k]):# or i < 150:# or\
                                # np.abs(self.nUpRyy - self.nUpRnn) > np.amax([self.nUpRyy, self.nUpRnn]) * 0.25:
                                print(f"i={i} [{self.cfg.mode}] -- Warning: matrices are not valid for gevd-based filter update.")
                                self.startFilterUpdates = False
                                break

                # Perform filter updates
                if self.cfg.nodeUpdating == 'seq':
                    if nIterSinceLastUp[self.q] >= self.cfg.nIterBetweenUpdates:
                        upNodes = np.array([self.q])
                        nIterSinceLastUp[self.q] = 0
                    else:
                        upNodes = np.array([])
                        nIterSinceLastUp[self.q] += 1
                elif self.cfg.nodeUpdating == 'sim':
                    upNodes = np.arange(self.cfg.K)
                if self.startFilterUpdates:
                    for u in upNodes:
                        if self.cfg.mode == 'batch':
                            mats = {'Ryy': Ryy, 'Rnn': Rnn}
                        elif self.cfg.mode == 'online':
                            # diagLoad = np.sum(np.abs(np.diag(Ryy[u]))) * np.eye(dimTilde)
                            # mats = {'Ryy': Ryy[u] + diagLoad, 'Rnn': Rnn[u] + diagLoad}
                            mats = {'Ryy': Ryy[u], 'Rnn': Rnn[u]}
                        out = filter_update(**mats, gevd=self.cfg.gevd)
                        if out is None:
                            pass  # <-- filter update failed
                        else:
                            wTilde[u] = out @ e

                    # Update external filters
                    wTildeExt[u] = copy.deepcopy(wTilde[u])  # default (actually purposeful if `self.cfg.nodeUpdating == 'sim'`)

                # Compute MMSE estimate of desired signal at each node
                mmses = self.get_mmse(wTilde, yTilde)
                for k in range(self.cfg.K):
                    mmse[k].append(mmses[k])
                
                # Print progress
                toPrint = f"[{algo.upper()} {self.cfg.mode} {self.cfg.nodeUpdating}] i = {i}, u = {upNodes}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}, vad = {self.wasn.vadOnline}"
                if not self.startFilterUpdates:
                    toPrint += " (filter updates not started yet)"
                print(toPrint)
                # Update indices
                i += 1
                self.q = (self.q + 1) % self.cfg.K
                # Randomly pick node to update
                stopcond = self.update_stop_condition(i, mmse)

                # Store filter coefficients
                wTildeSaved.append(copy.deepcopy(wTilde))
                vadSaved.append(copy.deepcopy(self.wasn.vadOnline))

            if algo == 'ti-danse':
                stop = 1
            # Store MMSE
            mmsePerAlgo[self.cfg.algos.index(algo)] = mmse
            filterCoeffsPerAlgo[self.cfg.algos.index(algo)] = wTildeSaved
        
        return mmsePerAlgo, mmseCentr, filterCoeffsPerAlgo, filterCoeffsCentr, vadSaved

    def conds_for_ti_danse_norm(self, i):
        """Check conditions for TI-DANSE normalization. Returns True if
        normalization should be applied at this iteration."""
        conds = []
        conds.append(self.cfg.mode == 'online')     # online-mode
        conds.append(self.wasn.vadOnline is False)  # noise-only period
        conds.append(i > 0)                         # not first iteration
        conds.append(i % self.cfg.normGkEvery == 0) # every `normGkEvery` iterations
        return sum(conds) == 1

    def update_scms(self, yTilde, nTilde, Ryy, Rnn):
        """Update spatial covariance matrices."""
        def _outer_prod(a: np.ndarray):
            """Compute inner product of `a` across channels."""
            if a.shape[0] > a.shape[1]:
                a = a.T
            return a @ a.T.conj()

        if self.cfg.mode == 'batch':
            if self.wasn.vadBatch is None:
                # No VAD -- update both `Ryy` and `Rnn` using oracle knowledge
                # of the noise-only signal.
                Ryy = _outer_prod(yTilde[self.q])
                Rnn = _outer_prod(nTilde[self.q])
            else:
                # VAD available -- update `Ryy` and `Rnn` using the VAD.
                Ryy = _outer_prod(yTilde[self.q][:, self.wasn.vadBatch])
                Rnn = _outer_prod(yTilde[self.q][:, ~self.wasn.vadBatch])
            self.nUpRyy += 1
            self.nUpRnn += 1
        elif self.cfg.mode == 'online':
            b = self.cfg.beta  # forgetting factor (alias)
            bRnn = self.cfg.betaRnn  # forgetting factor for noise-only covariance matrix (alias)
            for k in range(self.cfg.K):
                if self.wasn.vadOnline is None:
                    Ryy[k] = b * Ryy[k] + (1 - b) * _outer_prod(yTilde[k])
                    Rnn[k] = b * Rnn[k] + (1 - b) * _outer_prod(nTilde[k])
                    self.nUpRyy += 1
                    self.nUpRnn += 1
                else:
                    if self.wasn.vadOnline is True:
                        Ryy[k] = b * Ryy[k] + (1 - b) * _outer_prod(yTilde[k])
                        self.nUpRyy += 1
                    else:
                        Rnn[k] = bRnn * Rnn[k] + (1 - bRnn) * _outer_prod(yTilde[k])
                        self.nUpRnn += 1
        return Ryy, Rnn


    def get_centr_cost(self):
        """Compute centralized cost (MMSE) for each node."""

        # Set RNG state
        np.random.set_state(self.cfg.rngStateOriginal)

        nSensors = self.cfg.K * self.cfg.Mk
        if self.cfg.mode == 'batch':
            y = np.concatenate(tuple(self.wasn.nodes[k].signal for k in range(self.cfg.K)), axis=0)
            n = np.concatenate(tuple(self.wasn.nodes[k].noiseOnly for k in range(self.cfg.K)), axis=0)
            Ryy = y @ y.T.conj()
            Rnn = n @ n.T.conj()
            wCentral = filter_update(Ryy, Rnn, gevd=self.cfg.gevd)
            mmseCentral = np.zeros(self.cfg.K)
            for k in range(self.cfg.K):
                ek = np.zeros(nSensors)
                ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                mmseCentral[k] = np.mean(
                    np.abs((wCentral @ ek).T.conj() @ y -\
                        self.wasn.nodes[k].desiredOnly[self.cfg.refSensorIdx, :]) ** 2
                )
            filterCoeffsSaved = [copy.deepcopy(wCentral)]
        elif self.cfg.mode == 'online':
            singleSCM = np.random.randn(nSensors, nSensors)
            Ryy = copy.deepcopy(singleSCM)
            Rnn = copy.deepcopy(singleSCM)
            mmseCentral = [[] for _ in range(self.cfg.K)]
            wCentral = np.ones((nSensors, nSensors))
            stopcond = False
            i = 0
            self.cfg.sigConfig.sampleIdx = 0
            filterCoeffsSaved = []
            while not stopcond: 
                s, n = self.wasn.query()  # Get new data
                y = [s[k] + n[k] for k in range(self.cfg.K)]
                yStacked = np.concatenate(tuple(y[k] for k in range(self.cfg.K)), axis=0)
                nStacked = np.concatenate(tuple(n[k] for k in range(self.cfg.K)), axis=0)
                if self.wasn.vadOnline is None:
                    Ryy = self.cfg.beta * Ryy + (1 - self.cfg.beta) * yStacked @ yStacked.T.conj()
                    Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * nStacked @ nStacked.T.conj()
                else:
                    if self.wasn.vadOnline is True:
                        Ryy = self.cfg.beta * Ryy + (1 - self.cfg.beta) * yStacked @ yStacked.T.conj()
                    else:
                        Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * yStacked @ yStacked.T.conj()
                if self.cfg.gevd and not check_matrix_validity(Ryy, Rnn):
                    print(f"i={i} [centr online] -- Warning: matrices are not valid for self.cfg.gevd-based filter update.")
                else:
                    self.startFilterUpdates = True
                    wCentral = filter_update(Ryy, Rnn, gevd=self.cfg.gevd)
                for k in range(self.cfg.K):
                    ek = np.zeros(nSensors)
                    ek[k * self.cfg.Mk + self.cfg.refSensorIdx] = 1
                    target = (yStacked - nStacked).T @ ek
                    if np.allclose(np.abs(target), np.zeros_like(target)):
                        # If the target is zero (e.g., "off" period of speech),
                        # set the MMSE to `np.nan`.
                        mmseCentral[k].append(np.nan)
                    else:
                        mmseCentral[k].append(np.mean(np.abs(
                            (wCentral @ ek).T.conj() @ yStacked - target
                        ) ** 2))
                print(f"[Centr. {self.cfg.mode}] i = {i}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmseCentral]), -4)}")
                i += 1
                stopcond = self.update_stop_condition(i, mmseCentral)
                filterCoeffsSaved.append(copy.deepcopy(wCentral))
            mmseCentral = np.array(mmseCentral)
        return mmseCentral, filterCoeffsSaved
    
    def update_stop_condition(self, i, mmse):
        """Stop condition for DANSE `while`-loops."""
        if i > self.cfg.K:
            return i >= self.cfg.maxIter or np.all([
                np.abs(
                    me[-1] - me[-1 - self.cfg.K]
                ) / np.abs(me[-1 - self.cfg.K]) < self.cfg.eps
                for me in mmse
            ]) and self.startFilterUpdates  # <-- important: don't stop before the updates have started
        else:
            return False

    def get_tildes(self, algo, z_y, z_n):
        """Compute `yTilde` and `nTilde`."""
        yTilde = [_ for _ in range(self.cfg.K)]
        nTilde = [_ for _ in range(self.cfg.K)]
        for k in range(self.cfg.K):
            # $z_{-k}$ compressed signal vectors
            zMk_y = z_y[np.arange(self.cfg.K) != k, :]
            zMk_n = z_n[np.arange(self.cfg.K) != k, :]
            if algo == 'danse':
                yTilde[k] = np.concatenate((self.y[k], zMk_y), axis=0)
                nTilde[k] = np.concatenate((self.n[k], zMk_n), axis=0)
            elif algo == 'ti-danse':
                # vvv `sum(zMk_desired)` == $\eta_{-k}$ vvv
                etaMk_y = np.sum(zMk_y, axis=0)[np.newaxis, :]
                etaMk_n = np.sum(zMk_n, axis=0)[np.newaxis, :]
                yTilde[k] = np.concatenate((self.y[k], etaMk_y), axis=0)
                nTilde[k] = np.concatenate((self.n[k], etaMk_n), axis=0)    
        return yTilde, nTilde


    def get_mmse(self, wTilde, yTilde):
        """Compute MMSE."""
        currMMSEs = np.zeros(len(wTilde), dtype=object)
        for k in range(len(wTilde)):
            dHat = wTilde[k] @ yTilde[k].conj()
            target = (self.y[k] - self.n[k])[self.wasn.refSensorIdx, :]
            if np.allclose(np.abs(target), np.zeros_like(target)):
                # If the target is zero (e.g., "off" period of speech),
                # set the MMSE to `np.nan`.
                currMMSEs[k] = np.nan
                # currMMSEs[k] = None
            else:
                currMMSEs[k] = np.mean(np.abs(dHat - target) ** 2)
        return currMMSEs


def get_compressed_signals(yk, nk, algo, wkEXT):
    """Compute compressed signals using the given desired source-only and
    noise-only data."""
    Mk = yk.shape[0]
    if algo == 'danse':
        pk = wkEXT[:Mk]  # DANSE fusion vector
    elif algo == 'ti-danse':
        pk = wkEXT[:Mk] / wkEXT[-1]  # TI-DANSE fusion vector
    # Inner product of `pk` and `yk` across channels (fused signals)
    zk_y = np.sum(yk * pk[:, np.newaxis], axis=0)
    zk_n = np.sum(nk * pk[:, np.newaxis], axis=0)
    return zk_y, zk_n


def check_matrix_validity(Ryy, Rnn):
    """Check if `Ryy` is valid for GEVD-based filter updates."""
    def _is_posdef(x):
        """Check whether matrix `x` is positive definite."""
        return not any(np.linalg.eigvalsh(np.real_if_close(x)) < 0)
    def _has_full_rank(mat: np.ndarray):
        """Helper subfunction: check full-rank property."""
        return (np.linalg.matrix_rank(mat) == mat.shape[-1]).all()
    check1 = _is_posdef(Rnn)
    check2 = _is_posdef(Ryy)
    check3 = _has_full_rank(Rnn)
    check4 = _has_full_rank(Ryy)
    return check1 and check2 and check3 and check4


def filter_update(Ryy, Rnn, gevd=False, rank=1):
    """Update filter using GEVD-MWF or MWF."""
    if gevd:
        try:
            s, Xmat = sla.eigh(Ryy, Rnn)
        except ValueError as err:
            print(f"`scipy.linalg.eigh` error: {err}")
            return None
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
    

def random_posdef_fullrank_matrix(n):
    """Generates a full-rank, positive-definite matrix of size `n` with
    random entries."""
    Amat = np.random.randn(n, n)
    return Amat @ Amat.T.conj() + np.eye(n) * 0.01
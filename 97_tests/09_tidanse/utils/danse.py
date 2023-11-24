import copy
import numpy as np
import scipy.linalg as sla
from .scene import SceneCreator
import matplotlib.pyplot as plt

class Launcher:
    """Class to launch DANSE simulations."""
    def __init__(self, scene: SceneCreator):
        self.cfg = scene.cfg
        self.wasn = scene.wasn
        self.mmsePerAlgo = None
        self.mmseCentral = None
        self.filtersPerAlgo = None
        self.filtersCentral = None
        self.RyyPerAlgo = None
        self.RnnPerAlgo = None
        self.RssPerAlgo = None
        self.vadSaved = None
        self.etaMeanSaved = None
        self.nfSaved = None
        self.y = None  # Current desired signal chunk, for each node (`K`-elements list)
        self.n = None  # Current noise signal chunk, for each node (`K`-elements list)
        self.startFilterUpdates = False
        self.q = 0  # currently updating node index
        self.i = 0  # DANSE iteration index
        self.currAlgo = None
        # test / debug
        self.nVADflips = 0

    def run(self):
        """Run simulation."""
        if self.cfg.mode == 'wola':
            # Compute STFT signals
            self.wasn.compute_stft_signals()
            raise NotImplementedError  # TODO: implement TI-DANSE with WOLA
        elif self.cfg.mode in ['batch', 'online']:
            self.mmsePerAlgo, self.mmseCentral,\
            self.filtersPerAlgo, self.filtersCentral,\
            self.RyyPerAlgo, self.RnnPerAlgo, self.RssPerAlgo,\
            self.vadSaved, self.etaMeanSaved, self.nfSaved\
                = self.batch_or_online_run()

    def batch_or_online_run(self):
        # Compute centralized cost
        mmseCentr, filterCoeffsCentr = self.get_centr_cost()

        mmsePerAlgo = [[] for _ in range(len(self.cfg.algos))]
        filterCoeffsPerAlgo = [[] for _ in range(len(self.cfg.algos))]
        RyyPerAlgo = [[] for _ in range(len(self.cfg.algos))]
        RnnPerAlgo = [[] for _ in range(len(self.cfg.algos))]
        RssPerAlgo = [[] for _ in range(len(self.cfg.algos))]
        for algo in self.cfg.algos:
            self.currAlgo = algo
            # Set RNG state
            np.random.set_state(self.cfg.rngStateOriginal)
            # Initialize DANSE variables
            dimTilde = self.cfg.Mk + self.cfg.K - 1 if algo == 'danse' else self.cfg.Mk + 1
            e = np.zeros(dimTilde)
            e[self.cfg.refSensorIdx] = 1  # reference sensor selection vector
            wInit = np.ones(dimTilde)
            # wInit = np.random.randn(dimTilde)
            # wInit = np.zeros(dimTilde)
            # wInit[0] = 1
            # wInit[-1] = 1
            wTilde = [wInit for _ in range(self.cfg.K)]
            wTildeExt = copy.deepcopy(wTilde)
            # singleSCM = random_posdef_fullrank_matrix(dimTilde)
            singleSCM = np.zeros((dimTilde, dimTilde))
            Ryy = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            # singleSCM = random_posdef_fullra)ink_matrix(dimTilde)
            singleSCM = np.zeros((dimTilde, dimTilde))
            Rnn = [copy.deepcopy(singleSCM) for _ in range(self.cfg.K)]
            self.i = 0  # DANSE iteration index
            self.q = 0  # currently updating node index
            nf = 1  # normalization factor
            mmse = [[] for _ in range(self.cfg.K)]  # MMSE per node
            nIterSinceLastUp = [0 for _ in range(self.cfg.K)]
            stopcond = False
            self.startFilterUpdates = False
            self.cfg.sigConfig.sampleIdx = 0
            self.nUpRyy = [0 for _ in range(self.cfg.K)]
            self.nUpRnn = [0 for _ in range(self.cfg.K)]
            yTildeSaved = [[] for _ in range(self.cfg.K)]
            nTildeSaved = [[] for _ in range(self.cfg.K)]
            RyySaved = [[] for _ in range(self.cfg.K)]
            RnnSaved = [[] for _ in range(self.cfg.K)]
            RssSaved = [[] for _ in range(self.cfg.K)]
            vadSaved = []
            wTildeSaved = [copy.deepcopy(wTilde)]
            etaMeanSaved = []
            self.nVADflips = 0  # reset counter
            self.lastVADflipIdx = 0
            nfSaved = []
            while not stopcond:

                # Get new data
                if self.cfg.mode == 'online':
                    previousVAD = copy.deepcopy(self.wasn.vadOnline)
                    s, self.n = self.wasn.query()  # query new samples and update VAD status
                    if self.wasn.vadOnline != previousVAD:
                        self.nVADflips += 1  # <-- count VAD flips
                        self.lastVADflipIdx = self.i
                    vadSaved.append(copy.deepcopy(self.wasn.vadOnline))
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
                for k in range(self.cfg.K):
                    yTildeSaved[k].append(copy.deepcopy(yTilde[k]))
                    nTildeSaved[k].append(copy.deepcopy(nTilde[k]))
                etaMeanSaved.append(np.mean(np.sum(z_y, axis=0)))

                # Normalize \eta (TI-DANSE only)
                if algo == 'ti-danse' and self.conds_for_ti_danse_norm():
                    # Compute normalization factor
                    # if ~self.wasn.vadOnline:
                    if self.i > self.cfg.sigConfig.pauseSpacing // self.cfg.B:
                        nf = np.mean(np.abs(np.sum(z_y, axis=0)))
                    # nf = np.sum(z_y, axis=0)
                    for k in range(self.cfg.K):
                        if self.cfg.nodeUpdating == 'seq':
                            yTilde[k][-1, :] /= nf
                            nTilde[k][-1, :] /= nf
                        elif self.cfg.nodeUpdating == 'sim':
                            raise NotImplementedError('The normalization (to avoid divergence) of TI-DANSE coefficient is not implemented for simultaneous node-updating.')
                    nfSaved.append(copy.deepcopy(nf))

                # Update covariance matrices
                Ryy, Rnn = self.update_scms(
                    yTilde, nTilde,
                    Ryy, Rnn,
                    yTildeSaved=yTildeSaved,
                    nTildeSaved=nTildeSaved
                )
                for k in range(self.cfg.K):
                    RyySaved[k].append(copy.deepcopy(Ryy[k]))
                    RnnSaved[k].append(copy.deepcopy(Rnn[k]))

                # Check if filter updates should start
                if not self.startFilterUpdates:# and\
                    # self.i > self.cfg.sigConfig.pauseSpacing // self.cfg.B:
                    self.startFilterUpdates = True  # by default, start filter updates
                    if self.cfg.mode == 'online':
                        if self.nUpRnn[k] > dimTilde and self.nUpRyy[k] > dimTilde:
                            if self.cfg.gevd:
                                for k in range(self.cfg.K):
                                    if not check_matrix_validity_gevd(Ryy[k], Rnn[k]):# or self.i < 150:# or\
                                        print(f"i={self.i} [{self.cfg.mode}] -- Warning: matrices are not valid for gevd-based filter update.")
                                        self.startFilterUpdates = False
                                        break
                        else:
                            self.startFilterUpdates = False

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
                            mats = {'Ryy': Ryy[u], 'Rnn': Rnn[u]}
                        out, Rss = filter_update(**mats, gevd=self.cfg.gevd)
                        if out is None:
                            pass  # <-- filter update failed
                        else:
                            wTilde[u] = out @ e

                    # Update external filters
                    wTildeExt[u] = copy.deepcopy(wTilde[u])  # default (actually purposeful if `self.cfg.nodeUpdating == 'sim'`)
                else:
                    # Default Rss matrix
                    Rss = [np.zeros_like(Ryy[0]) for _ in range(self.cfg.K)]

                # Store Rss
                for k in range(self.cfg.K):
                    RssSaved[k].append(copy.deepcopy(Rss[k]))

                # Compute MMSE estimate of desired signal at each node
                mmses = self.get_mmse(wTilde, yTilde)
                for k in range(self.cfg.K):
                    mmse[k].append(mmses[k])
                
                # Print progress
                toPrint = f"[{algo.upper()} {self.cfg.mode} {self.cfg.nodeUpdating}] i = {self.i}, u = {upNodes}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmse]), -4)}, vad = {self.wasn.vadOnline}"
                if not self.startFilterUpdates:
                    toPrint += " (filter updates not started yet)"
                print(toPrint)
                # Update indices
                self.i += 1
                self.q = (self.q + 1) % self.cfg.K
                # Randomly pick node to update
                stopcond = self.update_stop_condition(mmse)

                # Store filter coefficients
                wTildeSaved.append(copy.deepcopy(wTilde))

            # Store MMSE
            mmsePerAlgo[self.cfg.algos.index(algo)] = mmse
            filterCoeffsPerAlgo[self.cfg.algos.index(algo)] = wTildeSaved
            RyyPerAlgo[self.cfg.algos.index(algo)] = RyySaved
            RnnPerAlgo[self.cfg.algos.index(algo)] = RnnSaved
            RssPerAlgo[self.cfg.algos.index(algo)] = RssSaved
        
        return mmsePerAlgo, mmseCentr, filterCoeffsPerAlgo, filterCoeffsCentr,\
            RyyPerAlgo, RnnPerAlgo, RssPerAlgo, vadSaved, etaMeanSaved, nfSaved

    def conds_for_ti_danse_norm(self):
        """Check conditions for TI-DANSE normalization. Returns True if
        normalization should be applied at this iteration."""
        if self.cfg.sigConfig.desiredSignalType == 'noise':
            return False  # no need for normalization if no VAD
        conds = []
        conds.append(self.cfg.mode == 'online')             # online-mode
        # conds.append(~self.wasn.vadOnline)                  # noise-only period
        conds.append(self.i > 0)                            # not first iteration
        conds.append(self.i % self.cfg.normGkEvery == 0)    # every `normGkEvery` iterations
        return sum(conds) == len(conds)

    def update_scms(
            self,
            yTilde, nTilde,
            Ryy, Rnn,
            yTildeSaved=None, nTildeSaved=None
        ):
        """Update spatial covariance matrices."""

        def _update_scm(scm, beta, y, ySaved):
            if self.cfg.scmEstType == 'exp':
                scm = beta * scm + (1 - beta) * outer_prod(y)
            elif self.cfg.scmEstType == 'rec':  # recursive updating
                scm += outer_prod(y)
                if self.i - self.cfg.L >= 0:
                    scm -= outer_prod(ySaved)
            return scm

        if self.cfg.mode == 'batch':
            if self.wasn.vadBatch is None:
                # No VAD -- update both `Ryy` and `Rnn` using oracle knowledge
                # of the noise-only signal.
                Ryy = outer_prod(yTilde[self.q])
                Rnn = outer_prod(nTilde[self.q])
            else:
                # VAD available -- update `Ryy` and `Rnn` using the VAD.
                Ryy = outer_prod(yTilde[self.q][:, self.wasn.vadBatch])
                Rnn = outer_prod(yTilde[self.q][:, ~self.wasn.vadBatch])
        elif self.cfg.mode == 'online':
            b = self.cfg.beta  # forgetting factor (alias)
            bRnn = self.cfg.betaRnn  # forgetting factor for noise-only covariance matrix (alias)
            for k in range(self.cfg.K):
                if self.wasn.vadOnline is None:
                    if self.cfg.scmEstType == 'exp':
                        Ryy[k] = b * Ryy[k] + (1 - b) * outer_prod(yTilde[k])
                        Rnn[k] = b * Rnn[k] + (1 - b) * outer_prod(nTilde[k])
                        self.nUpRyy[k] += 1
                        self.nUpRnn[k] += 1
                    elif self.cfg.scmEstType == 'rec':  # recursive updating
                        if self.i - self.cfg.L < 0:
                            pass  # do nothing
                            # Ryy[k] += outer_prod(yTilde[k]) * np.amax([1, self.i]) / (self.i + 1)
                            # Rnn[k] += outer_prod(nTilde[k]) * np.amax([1, self.i]) / (self.i + 1)
                        else:
                            if self.i - self.cfg.L == 0:
                                Ryy[k] = 1 / self.cfg.L * np.sum(np.array([outer_prod(y) for y in yTildeSaved[k]]), axis=0)
                                Rnn[k] = 1 / self.cfg.L * np.sum(np.array([outer_prod(n) for n in nTildeSaved[k]]), axis=0)
                            else:
                                Ryy[k] += 1 / self.cfg.L * outer_prod(yTilde[k]) -\
                                    1 / self.cfg.L * outer_prod(yTildeSaved[k][self.i - self.cfg.L])
                                Rnn[k] += 1 / self.cfg.L * outer_prod(nTilde[k]) -\
                                    1 / self.cfg.L * outer_prod(nTildeSaved[k][self.i - self.cfg.L])
                            self.nUpRyy += 1
                            self.nUpRnn += 1
                else:
                    kw = {
                        'y': yTilde[k],
                        'ySaved': yTildeSaved[k],
                    }
                    # if self.currAlgo == 'ti-danse' and\
                    #     self.nVADflips < self.cfg.minNumVADflips:
                    #     # for TI-DANSE, only update SCMs
                    #     # `numEarlySCMupPerVADperiod` after each VAD flip.
                    #     if self.i - self.lastVADflipIdx <=\
                    #         self.cfg.numEarlySCMupPerVADperiod:
                    #         # Update `Ryy` or `Rnn` depending on the VAD.
                    #         if self.wasn.vadOnline:
                    #             Ryy[k] = _update_scm(Ryy[k], b, **kw)
                    #             self.nUpRyy[k] += 1
                    #         else:
                    #             Rnn[k] = _update_scm(Rnn[k], bRnn, **kw)
                    #             self.nUpRnn[k] += 1
                    #     else:
                    #         # Don't update SCMs.
                    #         print(f"i={self.i} [early SCM update] -- Warning: not updating SCM because the number of VAD flips is less than `minNumVADflips` and the number of SCM updates since the last flip is greater than `numEarlySCMupPerVADperiod`.")
                    # else:
                    # Otherwise, update SCMs whenever possible.
                    if self.wasn.vadOnline:
                        Ryy[k] = _update_scm(Ryy[k], b, **kw)
                        self.nUpRyy[k] += 1
                    else:
                        Rnn[k] = _update_scm(Rnn[k], bRnn, **kw)
                        self.nUpRnn[k] += 1
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
            wCentral, _ = filter_update(Ryy, Rnn, gevd=self.cfg.gevd)
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
            singleSCM = np.zeros((nSensors, nSensors))
            Ryy = copy.deepcopy(singleSCM)
            Rnn = copy.deepcopy(singleSCM)
            mmseCentral = [[] for _ in range(self.cfg.K)]
            wCentral = np.ones((nSensors, nSensors))
            stopcond = False
            self.i = 0
            self.cfg.sigConfig.sampleIdx = 0
            filterCoeffsSaved = []
            yStackedSaved = []
            nStackedSaved = []
            while not stopcond: 
                s, n = self.wasn.query()  # Get new data
                y = [s[k] + n[k] for k in range(self.cfg.K)]
                yStacked = np.concatenate(tuple(y[k] for k in range(self.cfg.K)), axis=0)
                nStacked = np.concatenate(tuple(n[k] for k in range(self.cfg.K)), axis=0)
                yStackedSaved.append(copy.deepcopy(yStacked))
                nStackedSaved.append(copy.deepcopy(nStacked))
                # SCM updates
                if self.wasn.vadOnline is None:
                    if self.cfg.scmEstType == 'exp':
                        Ryy = self.cfg.beta * Ryy + (1 - self.cfg.beta) * outer_prod(yStacked)
                        Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * outer_prod(nStacked)
                    elif self.cfg.scmEstType == 'rec':  # recursive updating
                        if self.i - self.cfg.L < 0:
                            Ryy += outer_prod(yStacked) * np.amax([1, self.i]) / (self.i + 1)
                            Rnn += outer_prod(nStacked) * np.amax([1, self.i]) / (self.i + 1)
                        else:
                            Ryy += 1 / self.cfg.L * outer_prod(yStacked) -\
                                1 / self.cfg.L * outer_prod(yStackedSaved[self.i - self.cfg.L])
                            Rnn += 1 / self.cfg.L * outer_prod(nStacked) -\
                                1 / self.cfg.L * outer_prod(nStackedSaved[self.i - self.cfg.L])
                else:
                    if self.wasn.vadOnline:
                        if self.cfg.scmEstType == 'exp':
                            Ryy = self.cfg.beta * Ryy + (1 - self.cfg.beta) * outer_prod(yStacked)
                        elif self.cfg.scmEstType == 'rec':  # recursive updating
                            Ryy += outer_prod(yStacked)
                            if self.i - self.cfg.L > 0:
                                Ryy -= outer_prod(yStackedSaved[self.i - self.cfg.L])
                    else:
                        if self.cfg.scmEstType == 'exp':
                            Rnn = self.cfg.beta * Rnn + (1 - self.cfg.beta) * outer_prod(nStacked)
                        elif self.cfg.scmEstType == 'rec':  # recursive updating
                            Rnn += outer_prod(yStacked)
                            if self.i - self.cfg.L > 0:
                                Rnn -= outer_prod(yStackedSaved[self.i - self.cfg.L])


                if self.cfg.gevd and not check_matrix_validity_gevd(Ryy, Rnn):
                    print(f"i={self.i} [centr online] -- Warning: matrices are not valid for gevd-based filter update.")
                else:
                    self.startFilterUpdates = True
                    wCentral, _ = filter_update(Ryy, Rnn, gevd=self.cfg.gevd)
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
                print(f"[Centr. {self.cfg.mode}] i = {self.i}, mmse = {'{:.3g}'.format(np.mean([me[-1] for me in mmseCentral]), -4)}")
                self.i += 1
                stopcond = self.update_stop_condition(mmseCentral)
                filterCoeffsSaved.append(copy.deepcopy(wCentral))
            mmseCentral = np.array(mmseCentral)
        return mmseCentral, filterCoeffsSaved
    
    def update_stop_condition(self, mmse):
        """Stop condition for DANSE `while`-loops."""
        if self.i > self.cfg.K:
            return self.i >= self.cfg.maxIter or np.all([
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
        # pk = np.zeros_like(wkEXT[:Mk])
        # pk[0] = 1
    elif algo == 'ti-danse':
        pk = wkEXT[:Mk] / wkEXT[-1]  # TI-DANSE fusion vector
    # Inner product of `pk` and `yk` across channels (fused signals)
    zk_y = np.sum(yk * pk[:, np.newaxis], axis=0)
    zk_n = np.sum(nk * pk[:, np.newaxis], axis=0)
    return zk_y, zk_n


def check_matrix_validity_gevd(Ryy, Rnn):
    """Check if SCMs are valid for GEVD-based filter updates."""
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


def check_matrix_validity(Ryy):
    """Check if `Ryy` is valid for regular filter updates."""
    return (np.linalg.matrix_rank(Ryy) == Ryy.shape[-1]).all()


def filter_update(Ryy, Rnn, gevd=False, rank=1, pseudoInvGEVD=False):
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

        # Compute Rss matrix for comparison with steering matrix
        sTrunc = np.zeros_like(s)
        sTrunc[:rank] = s[:rank] - 1  # matrix (L - I)
        Rss = Qmat @ np.diag(sTrunc) @ Qmat.T.conj()

        if pseudoInvGEVD:
            # Pseudo-inverse GEVD-based filter update
            # sTrunc = np.ones_like(s) * 1e-9 * np.amax(s)
            sTrunc = np.zeros_like(s)
            sTrunc[:rank] = s[:rank]
            Ryypinv = np.linalg.pinv(  # <-- pseudo-inverse
                Qmat @ np.diag(sTrunc) @ Qmat.T.conj()
            )
            deltaMat = np.zeros_like(s)
            deltaMat[:rank] = s[:rank] - 1
            # deltaMat = sTrunc - 1
            return Ryypinv @ Qmat @ np.diag(deltaMat) @ Qmat.T.conj()
        else:
            Dmat = np.zeros_like(Ryy)
            for r in range(rank):
                Dmat[r, r] = 1 - 1 / s[r]  # <-- actual inverse
            return Xmat @ Dmat @ Qmat.T.conj(), Rss
    else:
        return np.linalg.inv(Ryy) @ (Ryy - Rnn), Ryy - Rnn
        # return np.linalg.pinv(Ryy) @ (Ryy - Rnn)  # `pinv` to deal with
                # "easiest" low/-rank scenarios (few localized sources
                # and/or low sensor-noise).
    

def random_posdef_fullrank_matrix(n):
    """Generates a full-rank, positive-definite matrix of size `n` with
    random entries."""
    Amat = np.random.randn(n, n)
    return Amat @ Amat.T.conj() + np.eye(n) * 0.01


def outer_prod(a: np.ndarray):
    """Compute outer product of `a` across channels."""
    if a.shape[0] > a.shape[1]:
        a = a.T
    return a @ a.T.conj()
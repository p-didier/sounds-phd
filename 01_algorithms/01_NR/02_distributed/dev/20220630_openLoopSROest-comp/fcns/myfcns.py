import numpy as np
from dataclasses import dataclass
import resampy
import matplotlib.pyplot as plt
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
import soundfile as sf

@dataclass
class SimParams:
    T : float   # signal duration [s]
    eps : int   # SRO [ppm]
    blockSize : int   # FFT size
    overlap : float   # frame overlap
    ld : int            # number of frames between two ``consecutive`` coherence functions
    seed : int = 12345  # random generator seed
    alpha : float = .95 # coherence product exp. avg. constant
    compensateSRO : bool = False    # if True, compensate SROs
    alphaEps : float = .01          # damping constant for combined SRO est.+comp.
    compType : str = 'closedloop'   # closed vs. open-loop compensation (see 2022 Word journal week26)

class Signal:
    def __init__(self, T, nChannels=1, seed=12345) -> None:
        self.T = T
        self.nChannels = nChannels
        self.seed = seed
        tmp, fs = sf.read('U:/py/sounds-phd/02_data/00_raw_signals/01_speech/speech1.wav')
        # Truncate
        tmp = tmp[:int(T * fs)]
        # Add a bit of noise to avoid pure silence periods
        tmpnoise, fs = sf.read('U:/py/sounds-phd/02_data/00_raw_signals/02_noise/whitenoise_signal_1.wav')
        tmpnoise = tmpnoise[:int(T * fs)]
        tmp2 = tmp + tmpnoise / np.amax(tmpnoise) * np.amax(tmp) / 10
        self.data = (np.tile(tmp2, (nChannels, 1))).T 
        self.fs = np.full(nChannels, fill_value=fs)

    def apply_sro(self, sroppm, channelIdx):
        tmp = resampy.core.resample(self.data[:, channelIdx],
                                                    self.fs[channelIdx],
                                                    self.fs[channelIdx] * (1 + sroppm * 1e-6))
        self.data[:, channelIdx] = tmp[:self.data.shape[0]]  # truncate
        self.fs[channelIdx] = self.fs[channelIdx] * (1 + sroppm * 1e-6)
        print(f'SRO of {sroppm} ppm applied to channel #{channelIdx + 1}.')

    def plot_waveform(self):
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(14.5, 6.5)
        axes.plot(self.data)
        axes.grid()
        plt.tight_layout()	
        plt.show()


def sro_est(wPos: np.ndarray, wPri: np.ndarray, avg_res_prod, Ns, ld, alpha=0.95, flagFirstSROEstimate=False):
    """Estimates residual SRO using a coherence drift technique.
    
    Parameters
    ----------
    wPos : [N x 1] np.ndarray (complex)
        A posteriori (iteration `i + 1`) value for every frequency bin
    wPri : [N x 1] np.ndarray (complex)
        A priori (iteration `i`) value for every frequency bin
    avg_res_prod : [2*(N-1) x 1] np.ndarray (complex)
        Exponentially averaged complex conjugate product of `wPos` and `wPri`
    Ns : int
        Number of new samples at each new STFT frame, counting overlap (`Ns=N*(1-O)`, where `O` is the amount of overlap [/100%])
    ld : int
        Number of STFT frames separating `wPos` from `wPri`.
    alpha : float
        Exponential averaging constant (DWACD method: .95).
    flagFirstSROEstimate : bool
        If True, this is the first SRO estimation round --> do not apply exponential averaging.

    Returns
    -------
    sro_est : float
        Estimated residual SRO
        -- `nLocalSensors` first elements of output should be zero (no intra-node SROs)
    avg_res_prod : [2*(N-1) x 1] np.ndarray (complex)
        Exponentially averaged residuals (complex conjugate) product - post-processing.
    """

    # "Residuals" product
    # res_prod = wPri * wPos.conj()
    res_prod = wPos * wPri.conj()
    # Prep for ISTFT (negative frequency bins too)
    res_prod = np.concatenate(
        [res_prod[:-1],
            np.conj(res_prod)[::-1][:-1]],
        -1
    )
    # Update the average coherence product
    if flagFirstSROEstimate:
        avg_res_prod = res_prod     # <-- 1st SRO estimation, no exponential averaging (initialization)
    else:
        avg_res_prod = alpha * avg_res_prod + (1 - alpha) * res_prod  

    # Estimate SRO
    # --------- DWACD-inspired "golden section search"
    sro_est = - max_time_lag_search(avg_res_prod) / (ld * Ns)

    return sro_est, avg_res_prod


def plotit(sroEst, sroResEst, legLabels, gt):

    flagCompensation = (sroResEst != 0).any()

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(14.5, 6.5)
    if sroEst.ndim == 1:
        axes.plot(sroEst * 1e6, label='$\\hat{{\\varepsilon}}_{{kq}}[l]$')
        if flagCompensation:
            axes.plot(sroResEst * 1e6, label='$\\Delta\\hat{{\\varepsilon}}_{{kq}}[l]$')
    else:
        for ii in range(sroEst.shape[-1]):
            axes.plot(sroEst[:, ii] * 1e6, label=f'$\\hat{{\\varepsilon}}_{{kq}}[l]$ -- {legLabels[ii]}')
            if flagCompensation:
                axes.plot(sroResEst[:, ii] * 1e6, label=f'$\\Delta\\hat{{\\varepsilon}}_{{kq}}[l]$ -- {legLabels[ii]}')
    axes.legend()
    axes.hlines(gt, xmin=0, xmax=len(sroEst), colors='r', linestyles='--')
    axes.grid()
    axes.set_xlabel('Frame index')
    axes.set_ylabel('[ppm]')
    plt.tight_layout()	
    plt.show()


def run(y, p: SimParams):
    """Performs frame-wise SRO estimation"""

    Ns = int(p.blockSize * (1 - p.overlap))
    nFrames = int(y.data.shape[0] // Ns + 1)

    # Frame-wise processing
    l = 0
    acr = 0     # initial average coherences product
    varphi = 0
    yyH = np.zeros((nFrames, int(p.blockSize / 2 + 1), y.nChannels, y.nChannels), dtype=complex)
    sroResEst = np.zeros(nFrames)
    sroEst = np.zeros(nFrames)
    while 1:
        # Corresponding time instant
        idxBeg = l * Ns
        idxEnd = idxBeg + p.blockSize
        if idxEnd > y.data.shape[0] - 1:
            break   # breaking point

        # Current chunk
        ycurr = y.data[idxBeg:idxEnd, :]

        # Go to freq. domain
        ycurrhat = np.fft.fft(ycurr, p.blockSize, axis=0)
        ycurrhat = ycurrhat[:int(ycurrhat.shape[0] / 2 + 1), :]     # keep only >0 freqs.

        if p.compensateSRO and p.compType == 'closedloop':
            # Compensate SROs
            ycurrhat[:, 1] *= np.exp(1j * 2 * np.pi / p.blockSize * np.arange(ycurrhat.shape[0]) * varphi)

        # Compute correlation matrix
        yyH[l, :, :, :] = np.einsum('ij,ik->ijk', ycurrhat, ycurrhat.conj())

        if l >= p.ld:

            cohPosteriori = yyH[l, :, 0, 1] / np.sqrt(yyH[l, :, 0, 0] * yyH[l, :, 1, 1])     # a posteriori coherence
            cohPriori = yyH[l - p.ld, :, 0, 1] / np.sqrt(yyH[l - p.ld, :, 0, 0] * yyH[l - p.ld, :, 1, 1])     # a posteriori coherence

            sroEstCurr, acr = sro_est(wPos=cohPosteriori,
                                wPri=cohPriori,
                                avg_res_prod=acr,
                                Ns=Ns,
                                ld=p.ld,
                                alpha=p.alpha,
                                flagFirstSROEstimate=l==p.ld)

            if p.compensateSRO:
                sroResEst[l] = sroEstCurr
                if p.compType == 'closedloop':
                    sroEst[l] = sroEst[l - 1] + p.alphaEps * sroResEst[l]
                    varphi += sroEst[l] * Ns
                elif p.compType == 'openloop':
                    varphi += sroEstCurr * Ns
            else:
                sroEst[l] = sroEstCurr

        l += 1  # next frame

    return sroEst, sroResEst


import sys
import resampy
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("danse_utilities" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed/danse_utilities')
from classes import DWACDParameters, ProgramSettings


def apply_sto(x, stoDelay, fs):
    """Applies a given sampling time offset (STO) to a signal.
    ---
    As of 11/04/2022: only applying full-sample STOs. 
    TODO: include possibility of applying sub-sample STOs.
    ---
    NOTE: STOs are always applied backwards: data is discarded
    from the beginning of the vector. 

    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Input signal (single-channel).
    stoDelay : float
        [s] STO-induced delay.
    fs : int or float
        [samples/s] Sampling frequency.
    
    Returns
    -------
    xsto : [N x 1] np.ndarray (float)
        Output signal (STO-disturbed).
    """

    sto = int(np.floor(stoDelay * fs))

    if sto > 0:
        xsto = np.concatenate((x[sto:], np.zeros(sto)))
    else:
        xsto = x

    return xsto


def resample_for_sro(x, baseFs, SROppm):
    """Resamples a vector given an SRO and a base sampling frequency.

    Parameters
    ----------
    x : [N x 1] np.ndarray
        Signal to be resampled.
    baseFs : float or int
        Base sampling frequency [samples/s].
    SROppm : float
        SRO [ppm].

    Returns
    -------
    xResamp : [N x 1] np.ndarray
        Resampled signal
    t : [N x 1] np.ndarray
        Corresponding resampled time stamp vector.
    fsSRO : float
        Re-sampled signal's sampling frequency [Hz].
    """
    fsSRO = baseFs * (1 + SROppm / 1e6)
    if baseFs != fsSRO:
        xResamp = resampy.core.resample(x, baseFs, fsSRO)
    else:
        xResamp = copy(x)

    # tOriginal = np.arange(len(x)) / baseFs
    # numSamplesPostResamp = int(np.floor(fsSRO / baseFs * len(x)))
    # xResamp, t = sig.resample(x, num=numSamplesPostResamp, t=tOriginal)
    # t = np.arange(0, numSamplesPostResamp) * (tOriginal[1] - tOriginal[0]) * x.shape[0] / float(numSamplesPostResamp) + tOriginal[0]    # based on line 3116 in `scipy.signal.resample`

    t = np.arange(len(xResamp)) / fsSRO

    if len(xResamp) >= len(x):
        xResamp = xResamp[:len(x)]
        t = t[:len(x)]
    else:
        # Append zeros
        xResamp = np.concatenate((xResamp, np.zeros(len(x) - len(xResamp))))
        # Extend time stamps vector
        dt = t[1] - t[0]
        tadd = np.linspace(t[-1]+dt, t[-1]+dt*(len(x) - len(xResamp)), len(x) - len(xResamp))
        t = np.concatenate((t, tadd))

    return xResamp, t, fsSRO


def apply_sro_sto(sigs, baseFs, sensorToNodeTags, SROsppm, STOinducedDelays, plotit=False):
    """Applies sampling rate offsets (SROs) and sampling time offsets (STOs) to signals.

    Parameters
    ----------
    sigs : [N x Ns] np.ndarray
        Signals onto which to apply SROs (<Ns> sensors with <N> samples each).
    baseFs : float
        Base sampling frequency [samples/s]
    sensorToNodeTags : [Ns x 1] np.ndarray
        Tags linking each sensor (channel) to a node (i.e. an SRO).
    SROsppm : [Nn x 1] np.ndarray or list (floats)
        SROs per node [ppm].
    STOinducedDelays : [Nn x 1] np.ndarray or list (floats)
        STO-induced delay per node [s].
    plotit : bool
        If True, plots a visualization of the applied signal changes.

    Returns
    -------
    sigsOut : [N x Ns] np.ndarray of floats
        Signals after SROs application.
    timeVectorOut : [N x Nn] np.ndarray of floats
        Corresponding sensor-specific time stamps vectors.
    fs : [Ns x 1] np.ndarray of floats
        Sensor-specific sampling frequency, after SRO application. 
    """
    # Extract useful variables
    numSamples = sigs.shape[0]
    numSensors = sigs.shape[-1]
    numNodes = len(np.unique(sensorToNodeTags))

    # Check / adapt SROs object size
    if len(SROsppm) != numNodes:
        if len(SROsppm) == 1:
            print(f'Applying the same SRO ({SROsppm} ppm) to all {numNodes} nodes.')
            SROsppm = np.repeat(SROsppm[0], numNodes)
        else:
            raise ValueError(f'An incorrect number of SRO values was provided ({len(SROsppm)} for {numNodes} nodes).')

    # Apply STOs / SROs
    sigsOut       = np.zeros((numSamples, numSensors))
    timeVectorOut = np.zeros((numSamples, numNodes))
    fs = np.zeros(numSensors)
    for idxSensor in range(numSensors):
        k = sensorToNodeTags[idxSensor] - 1     # corresponding node index
        # Apply SROs
        sigsOut[:, idxSensor], timeVectorOut[:, k], fs[idxSensor] = resample_for_sro(sigs[:, idxSensor], baseFs, SROsppm[k])
        # Apply STOs
        sigsOut[:, idxSensor] = apply_sto(sigsOut[:, idxSensor], STOinducedDelays[k], fs[idxSensor])

    # Plot
    if plotit:
        minimumObservableDrift = 1   # plot enough samples to observe drifts of at least that many samples on all signals
        smallestDriftFrequency = np.amin(SROsppm[SROsppm != 0]) / 1e6 * baseFs  # [samples/s]
        samplesToPlot = int(minimumObservableDrift / smallestDriftFrequency * baseFs)
        markerFormats = ['o','v','^','<','>','s','*','D']
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        for k in range(numNodes):
            allSensors = np.arange(numSensors)
            idxSensor = allSensors[sensorToNodeTags == (k + 1)]
            if isinstance(idxSensor, np.ndarray):
                idxSensor = idxSensor[0]
            markerline, _, _ = ax.stem(timeVectorOut[:samplesToPlot, k], sigsOut[:samplesToPlot, idxSensor],
                    linefmt=f'C{k}', markerfmt=f'C{k}{markerFormats[k % len(markerFormats)]}',
                    label=f'Node {k + 1} - $\\varepsilon={SROsppm[k]}$ ppm')
            markerline.set_markerfacecolor('none')
        ax.set(xlabel='$t$ [s]', title='SROs / STOs visualization')
        ax.grid()
        plt.tight_layout()
        plt.show()

    return sigsOut, timeVectorOut, fs

def cohdrift_sro_estimation(wPos: np.ndarray,
                            wPri: np.ndarray,
                            avgResProd,
                            Ns,
                            ld,
                            alpha=0.95,
                            method='gs',
                            flagFirstSROEstimate=False,
                            bufferFlagPos=0,
                            bufferFlagPri=0):
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
    method : str
        Method to use to retrieve SRO once the exponentially averaged product has been computed.
    flagFirstSROEstimate : bool
        If True, this is the first SRO estimation round --> do not apply exponential averaging.
    bufferFlag : TODO

    Returns
    -------
    sro_est : float
        Estimated residual SRO
        -- `nLocalSensors` first elements of output should be zero (no intra-node SROs)
    avg_res_prod_out : [2*(N-1) x 1] np.ndarray (complex)
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
    # Account for potential buffer flags (extra / missing sample)
    res_prod *= np.exp(1j * 2 * np.pi / len(res_prod) * np.arange(len(res_prod)) * (bufferFlagPos - bufferFlagPri))

    # Update the average coherence product
    if flagFirstSROEstimate:
        avgResProd_out = res_prod     # <-- 1st SRO estimation, no exponential averaging (initialization)
    else:
        avgResProd_out = alpha * avgResProd + (1 - alpha) * res_prod 

    # Estimate SRO
    if method == 'gs':
        # --------- DWACD-inspired "golden section search"
        sro_est = - max_time_lag_search(avgResProd_out) / (ld * Ns)
        
    elif method == 'mean':
        # --------- Online-WACD-inspired "average phase"
        print('Filter shift "mean" method for SRO estimation --> Not quite working?')
        # TODO: avoid hard-coding
        epsmax = 400 * 1e-6
        kmin, kmax = 1, len(wPri)
        kappa = np.arange(kmin, kmax)    # freq. bins indices
        norm_phase = np.angle(avgResProd_out[(len(avgResProd_out) // 2 - 1 + kmin):(len(avgResProd_out) // 2 - 1 + kmax)])\
                                                * len(wPri) / (2 * (ld * Ns) * kappa * epsmax)
        mean_phase = np.mean(np.abs(avgResProd_out[(len(avgResProd_out) // 2 - 1 + kmin):(len(avgResProd_out) // 2 - 1 + kmax)])\
                                                * np.exp(1j * norm_phase))
        sro_est = - epsmax / np.pi * np.angle(mean_phase)

    elif method == 'ls':
        # --------- Least-squares solution over frequency bins
        kappa = np.arange(0, len(wPri))    # freq. bins indices
        # b = 2 * np.pi * kappa * (ld * Ns) / len(kappa)
        b = np.pi * kappa * (ld * Ns) / (len(kappa) * 2)
        sro_est = - b.T @ np.angle(avgResProd_out[-len(kappa):]) / (b.T @ b)

    return sro_est, avgResProd_out


def dwacd_sro_estimation(sigSTFT, ref_sigSTFT, activity_sig, activity_ref_sig,
                        paramsDWACD: DWACDParameters,
                        seg_idx,
                        sro_est,
                        avg_coh_prod
                        ):
    """Dynamic Weighted Average Coherence Drift (DWACD)-based SRO
    estimation.

    References
    ----------
    [1] Gburrek, Tobias, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach.
        "On Synchronization of Wireless Acoustic Sensor Networks in the Presence
        of Time-Varying Sampling Rate Offsets and Speaker Changes."
        ICASSP 2022-2022 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2022.
    [2] DWACD repository: https://github.com/fgnt/paderwasn

    Estimate the SRO of the single channel signal sig w.r.t. the single
    channel reference signal ref_sig

    Parameters
    ----------
    sigSTFT : [Nf x Nl] np.ndarray (complex)
        Matrix corresponding to the signal whose SRO should be
        estimated, in the STFT domain (`Nf` freq. bins, `Nl` time frames)
    ref_sigSTFT : [Nf x Nl] np.ndarray (complex)
        Matrix corresponding to the reference signal (Should have the
        same length as sig), in the STFT domain (`Nf` freq. bins, `Nl` time frames)
    activity_sig : [Nl x 1] np.ndarray (int or bool)
        Vector containing the sample-wise information of source
        activity in the signal sig
    activity_ref_sig : [Nl x 1] np.ndarray (int or bool)
        Vector containing the sample-wise information of source
        activity in the reference signal ref_sig
    paramsDWACD : DWACDParameters object
        DWACD parameters
    seg_idx : int
        DWACD segment index
    sro_est : float
        SRO estimate from previous DWACD segment
    avg_coh_prod : [Nf x 1] np.ndarray (complex)
        Average coherence product from previous DWACD segment

    Returns
    -------
    sro_estimate : float
        SRO estimate [NOT in ppm]
    """

    def calc_psd_stft(frame_shift_welch, fft_size, stft_seg_i, stft_seg_j, sro=0.):
        """Estimate the (cross) power spectral density (PSD) from the given
        signal segments using a Welch method, starting from the STFT domain.

        Parameters
        ----------
        stft_seg_i : [Nf x Nl] np.ndarray (complex)
            Matrix with seg_len elements corresponding to the segment taken
            from the i-th signal, in the STFT domain
        stft_seg_j : [Nf x Nl] np.ndarray (complex)
            Matrix with seg_len elements corresponding to the segment taken
            from the j-th signal, in the STFT domain
        sro : float
            SRO to be compensated in the Welch method [NOT in ppm]

        Returns
        -------
        psd_ij : [Nf x 1] np.ndarray (complex)
            PSD of the the i-th signal and the j-th signal
        """
        shifts = sro * frame_shift_welch * np.arange(stft_seg_j.shape[-1])
        stft_seg_j *= \
            np.exp(1j * 2 * np.pi / fft_size
                   * np.arange(fft_size // 2 + 1)[None] * shifts[:, None]).T
        return np.mean(stft_seg_i * np.conj(stft_seg_j), axis=-1)


    def calc_coherence_stft(frame_shift_welch, fft_size, seg, seg_ref, sro):
        """Estimate the coherence from the given signal segments, 
        directly in the STFT domain.

        Parameters
        ----------
        seg [Nf x Nl] np.ndarray (complex)
            Matrix with seg_len elements corresponding to the segment taken
            from the i-th signal, in the STFT domain
        seg_ref [Nf x Nl] np.ndarray (complex)
            Matrix with seg_len elements corresponding to the segment taken
            from the j-th signal, in the STFT domain
        sro : float
            SRO to be compensated when calculating the PSDs needed for
            coherence estimation [NOT in ppm]

        Returns
        -------
        gamma [Nf x 1] np.ndarray (complex)
            Coherence of the signal and the reference signal
        """
        cpsd = calc_psd_stft(frame_shift_welch, fft_size, seg_ref, seg, sro)
        psd_ref_sig = calc_psd_stft(frame_shift_welch, fft_size, seg_ref, seg_ref)
        psd_sig = calc_psd_stft(frame_shift_welch, fft_size, seg, seg)
        gamma = cpsd / (np.sqrt(psd_ref_sig * psd_sig) + 1e-13)
        return gamma

    # Extract parameters
    seg_len = paramsDWACD.seg_len
    seg_shift = paramsDWACD.seg_shift
    frame_shift_welch = paramsDWACD.frame_shift_welch
    fft_size = paramsDWACD.fft_size
    temp_dist = paramsDWACD.temp_dist
    alpha = paramsDWACD.alpha
    src_activity_th = paramsDWACD.src_activity_th
    # settling_time = paramsDWACD.settling_time

    # Useful quantities
    Nl_segShift = seg_shift // frame_shift_welch        # number of STFT frames per segment shift 
    Nl_segLen   = seg_len // frame_shift_welch          # number of STFT frames per segment 
    Nl_cohDelay = temp_dist // frame_shift_welch        # number of STFT frames corresponding to the time 
                                                        # interval between two consecutive coherence functions

    # Define STFT frame indices for current segment
    lStart = seg_idx * Nl_segShift + Nl_cohDelay
    lEnd = lStart + Nl_segLen

    # Segment activity
    thresholdVAD = src_activity_th / seg_len * (lEnd - lStart + 1)
    activity_seg = activity_sig[lStart:lEnd]
    activity_seg_ref = activity_ref_sig[lStart:lEnd]
    activity_seg_delayed = activity_sig[(lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
    activity_seg_ref_delayed = activity_ref_sig[(lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
    activity = (
            np.sum(activity_seg_ref_delayed) > thresholdVAD
            and np.sum(activity_seg_ref) > thresholdVAD
            and np.sum(activity_seg_delayed) > thresholdVAD
            and np.sum(activity_seg) > thresholdVAD
    )

    if activity:
        # Compute coherence directly from STFT domain signals
        seg_ref = ref_sigSTFT[:, lStart:lEnd] 
        seg = sigSTFT[:, lStart:lEnd]
        coherence = calc_coherence_stft(frame_shift_welch, fft_size,
                                        seg, seg_ref, sro_est)

        seg_ref_delayed = ref_sigSTFT[:, (lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
        seg_delayed = sigSTFT[:, (lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
        coherence_delayed = calc_coherence_stft(frame_shift_welch, fft_size,
                                                seg_delayed, seg_ref_delayed, sro_est)

        # Compute complex conjugate product
        coherence_product = coherence * np.conj(coherence_delayed)

        # Prep for ISTFT (negative frequency bins too)
        coherence_product = np.concatenate(
            [coherence_product[:-1],
                np.conj(coherence_product)[::-1][:-1]],
            -1
        )

        # Update the average coherence product
        avg_coh_prod = (alpha * avg_coh_prod
                        + (1 - alpha) * coherence_product)

        # Estimate SRO
        sro_est = - max_time_lag_search(avg_coh_prod) / temp_dist

    return sro_est, avg_coh_prod


def update_sro_estimates(settings: ProgramSettings, iter,
                        nLocalSensors,
                        cohDriftSROupdateIndices,
                        neighbourNodes,
                        yyH, yyHuncomp, 
                        avgProdRes,
                        oracleSRO,
                        bufferFlagPos,
                        bufferFlagPri):
    """
    Update SRO estimates.

    Parameters
    ----------
    TODO:
    """

    sroOut = np.zeros(len(neighbourNodes))
    avgProdResOut = np.zeros((avgProdRes.shape[0], len(neighbourNodes)), dtype=complex)

    if settings.asynchronicity.estimateSROs == 'CohDrift':
        
        ld = settings.asynchronicity.cohDriftMethod.segLength

        if iter in cohDriftSROupdateIndices:

            flagFirstSROEstimate = False
            if iter == np.amin(cohDriftSROupdateIndices):
                flagFirstSROEstimate = True     # let `cohdrift_sro_estimation()` know that this is the 1st SRO estimation round

            # Residuals method
            for q in range(len(neighbourNodes)):

                idxq = nLocalSensors + q     # index of the compressed signal from node `q` inside `yyH`
                if settings.asynchronicity.cohDriftMethod.loop == 'closed':
                    # Use SRO-compensated correlation matrix entries (closed-loop SRO est. + comp.)
                    cohPosteriori = (yyH[iter, :, 0, idxq]
                                            / np.sqrt(yyH[iter, :, 0, 0] * yyH[iter, :, idxq, idxq]))     # a posteriori coherence
                    cohPriori = (yyH[iter - ld, :, 0, idxq]
                                            / np.sqrt(yyH[iter - ld, :, 0, 0] * yyH[iter - ld, :, idxq, idxq]))     # a priori coherence
                elif settings.asynchronicity.cohDriftMethod.loop == 'open':
                    # Use SRO-_un_compensated correlation matrix entries (open-loop SRO est. + comp.)
                    cohPosteriori = (yyHuncomp[iter, :, 0, idxq]
                                            / np.sqrt(yyHuncomp[iter, :, 0, 0] * yyHuncomp[iter, :, idxq, idxq]))     # a posteriori coherence
                    cohPriori = (yyHuncomp[iter - ld, :, 0, idxq]
                                            / np.sqrt(yyHuncomp[iter - ld, :, 0, 0] * yyHuncomp[iter - ld, :, idxq, idxq]))     # a priori coherence

                sroRes, apr = cohdrift_sro_estimation(
                                    wPos=cohPosteriori,
                                    wPri=cohPriori,
                                    avgResProd=avgProdRes[:, q],
                                    Ns=settings.Ns,
                                    ld=ld,
                                    method=settings.asynchronicity.cohDriftMethod.estimationMethod,
                                    alpha=settings.asynchronicity.cohDriftMethod.alpha,
                                    flagFirstSROEstimate=flagFirstSROEstimate,
                                    bufferFlagPri=bufferFlagPri[q],
                                    bufferFlagPos=bufferFlagPos[q]
                                    )
            
                sroOut[q] = sroRes
                avgProdResOut[:, q] = apr

    elif settings.asynchronicity.estimateSROs == 'Oracle':        # no data-based dynamic SRO estimation: use oracle knowledge
        sroOut = (settings.asynchronicity.SROsppm[neighbourNodes] - oracleSRO) * 1e-6

    return sroOut, avgProdResOut
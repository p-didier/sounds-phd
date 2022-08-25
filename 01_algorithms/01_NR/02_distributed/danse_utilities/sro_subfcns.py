
from this import d
import numpy as np
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
from .classes import DWACDParameters, ProgramSettings
import matplotlib.pyplot as plt


def cohdrift_sro_estimation(wPos: np.ndarray, wPri: np.ndarray, avgResProd, Ns, ld, alpha=0.95, method='gs', flagFirstSROEstimate=False):
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
                        oracleSRO):
    """
    Update SRO estimates.

    Parameters
    ----------

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
                                    Ns=int(settings.DFTsize * (1 - settings.Ns)),
                                    ld=ld,
                                    method=settings.asynchronicity.cohDriftMethod.estimationMethod,
                                    alpha=settings.asynchronicity.cohDriftMethod.alpha,
                                    flagFirstSROEstimate=flagFirstSROEstimate,
                                    )
            
                sroOut[q] = sroRes
                avgProdResOut[:, q] = apr

    elif settings.asynchronicity.estimateSROs == 'Oracle':        # no data-based dynamic SRO estimation: use oracle knowledge
        sroOut = (settings.asynchronicity.SROsppm[neighbourNodes] - oracleSRO) * 1e-6

    return sroOut, avgProdResOut
import numpy as np
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
from .classes import DWACDParameters


def residual_sro_estimation(resPos, resPri, nLocalSensors, N, Ns):
    """Estimates residual SRO using the residuals phase technique.
    
    Parameters TODO: change inputs to accomodate other SRO estimation methods
    ----------
    resPos : [N x M] np.ndarray (complex)
        A posteriori (iteration `i + 1`) residuals for every frequency bin and every element of $\\tilde{y}$.
    resPri : [N x M] np.ndarray (complex)
        A priori (iteration `i`) residuals for every frequency bin and every element of $\\tilde{y}$.
    nLocalSensors : int
        Number of local sensors at current node.
    N : int
        Processing frame length (== DANSE DFT size).
    Ns : int
        Number of new samples at each new frame, counting overlap (`Ns = N * (1 - O)`, where `O` is the amount of overlap [/100%])

    Returns
    -------
    residualSROs : [M x 1] np.ndarray (float)
        Estimated residual SROs for each node.
        -- `nLocalSensors` first elements of output should be zero (no intra-node SROs)
    """
    
    # Check correct input orientation
    if resPos.shape[0] < resPos.shape[1]:
        resPos = resPos.T
    if resPri.shape[0] < resPri.shape[1]:
        resPri = resPri.T
    # Create useful explicit variables
    numFreqLines = resPos.shape[0]
    dimYTilde = resPos.shape[1]
    
    # Compute "residuals angle" matrix
    phi = np.angle(resPri * resPos.conj())      # see LaTeX journal 2022 week 02
    phi[:, :nLocalSensors] = 0              # NOTE: force a 0 residual at the local sensors (no intra-node SRO)

    # Create `kappa` frequency bins vector
    kappa = 2 * np.pi / N * Ns * np.arange(numFreqLines)  # TODO: check if that is correct -- Eq. (3), week 02 in LaTeX journal 2022
    
    # Estimate residual SRO as least-square solution to system of equation across frequency bins
    nfToDiscard = 0    # number of lowest frequency bins to discard -- see Word journal, TUE 26/04, week 17 2022
    residualSROs = np.zeros(dimYTilde)
    for q in range(dimYTilde):
        residualSROs[q] = np.dot(kappa[nfToDiscard:], phi[nfToDiscard:, q]) / np.dot(kappa[nfToDiscard:], kappa[nfToDiscard:])

    return residualSROs


def dwacd_sro_estimation(sigSTFT, ref_sigSTFT, activity_sig, activity_ref_sig,
                        paramsDWACD: DWACDParameters,
                        seg_idx,
                        sro_est,
                        tau_sro,
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
    tau_sro : float
        [s] Time shift estimate from previous DWACD segment
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
        # shifts = np.zeros_like(shifts)  # TMP <-- OVERRIDE SHIFT EFFECT
        # print(shifts)
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
    Nf = sigSTFT.shape[0]      # number of STFT bins
    Nl_segShift = seg_shift // frame_shift_welch      # number of STFT frames per segment shift 
    Nl_segLen   = seg_len // frame_shift_welch        # number of STFT frames per segment 
    Nl_cohDelay = temp_dist // frame_shift_welch      # number of STFT frames corresponding to the time 
                                                      # interval between two consecutive coherence functions
    # Estimate of the SRO-induced integer shift to be compensated
    shift = int(np.round(tau_sro))
    # Corresponding STFT frequency shift (Linear Phase Drift model)
    phaseShift = np.exp( 1j * 2 * np.pi / Nf * np.arange(Nf) * shift)
    phaseShift = phaseShift[:, np.newaxis]  # adapt array format for subsequent element-wise product

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
        seg = phaseShift * sigSTFT[:, lStart:lEnd]
        coherence = calc_coherence_stft(frame_shift_welch, fft_size,
                                        seg, seg_ref, sro_est)

        seg_ref_delayed = ref_sigSTFT[:, (lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
        seg_delayed = phaseShift * sigSTFT[:, (lStart - Nl_cohDelay):(lEnd - Nl_cohDelay)]
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

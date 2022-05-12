import numpy as np
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search


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


def dwacd_sro_estimation(seg_len,
                        seg_shift,
                        frame_shift_welch,
                        fft_size,
                        temp_dist,
                        alpha,
                        src_activity_th,
                        settling_time,
                        sig, ref_sig, activity_sig, activity_ref_sig):
    """Dynamic Weighted Average Coherence Drift (DWACD)-based SRO
    estimation.
    
    Parameters
    ----------
    -

    Returns
    -------
    -

    References
    ----------
    [1] Gburrek, Tobias, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach.
        "On Synchronization of Wireless Acoustic Sensor Networks in the Presence
        of Time-Varying Sampling Rate Offsets and Speaker Changes."
        ICASSP 2022-2022 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2022.
    """

    
    """Estimate the SRO of the single channel signal sig w.r.t. the single
    channel reference signal ref_sig

    Args:
        seg_len (int):
            Length of the segments used for coherence estimation (= Length
            of the segments used for power spectral density (PSD)
            estimation based on a Welch method)
        seg_shift (int):
            Shift of the segments used for coherence estimation (The SRO is
            estimated every seg_shift samples)
        frame_shift_welch (int):
            Frame shift used for the Welch method utilized for
            PSD estimation
        fft_size (int):
            Frame size / FFT size used for the Welch method utilized for
            PSD estimation
        temp_dist (int):
            Amount of samples between the two consecutive coherence
            functions
        alpha (float):
            Smoothing factor used for the autoregressive smoothing for time
            averaging of the complex conjugated coherence product
        src_activity_th (float):
            If the amount of time with source activity within one segment
            is smaller than the threshold src_activity_th the segment will
            not be used to update th average coherence product.
        settling_time (int):
            Amount of segments after which the SRO is estimated for the
            first time
        sig (array-like):
            Vector corresponding to the signal whose SRO should be
            estimated
        ref_sig (array-like):
            Vector corresponding to the reference signal (Should have the
            same length as sig)
        activity_sig(array-like):
            Vector containing the sample-wise information of source
            activity in the signal sig
        activity_ref_sig(array-like):
            Vector containing the sample-wise information of source
            activity in the reference signal ref_sig
    Returns:
        sro_estimates (numpy.ndarray):
            Vector containing the SRO estimates in ppm
    """

    raise ValueError('[NOT YET IMPLEMENTED]')

    def calc_psd(seg_i, seg_j, sro=0.):
        """Estimate the (cross) power spectral density (PSD) from the given
        signal segments using a Welch method

        Args:
            seg_i (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the i-th signal
            seg_j (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the j-th signal
            sro (float):
                SRO to be compensated in the Welch method
        Returns:
            psd_ij (numpy.ndarray):
                PSD of the the i-th signal and the j-th signal
        """
        stft_seg_j = self.stft(seg_j)
        shifts = sro * self.frame_shift_welch * np.arange(len(stft_seg_j))
        stft_seg_j *= \
            np.exp(1j * 2 * np.pi / self.fft_size
                   * np.arange(self.fft_size // 2 + 1)[None] * shifts[:, None])
        return np.mean(self.stft(seg_i) * np.conj(stft_seg_j), axis=0)

    def calc_coherence(seg, seg_ref, sro):
        """Estimate the coherence from the given signal segments

        Args:
            seg (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from signal whose SRO should be estimated
            seg_ref (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the reference signal
            sro (float):
                SRO to be compensated when calculating the PSDs needed for
                coherence estimation
        Returns:
            gamma (numpy.ndarray):
                Coherence of the signal and the reference signal
        """
        cpsd = calc_psd(seg_ref, seg, sro)
        psd_ref_sig = calc_psd(seg_ref, seg_ref)
        psd_sig = calc_psd(seg, seg)
        gamma = cpsd / (np.sqrt(psd_ref_sig * psd_sig) + 1e-13)
        return gamma


    # Maximum number of segments w.r.t. the reference signal (The actual
    # number of segments might be smaller due to the compensation of the
    # SRO-induced signal shift)
    num_segments = int(
        (len(ref_sig) - temp_dist - seg_len + seg_shift)
        // seg_shift
    )
    sro_estimates = np.zeros(num_segments)
    avg_coh_prod = np.zeros(fft_size)

    # The SRO-induced signal shift will be estimated based on the
    # SRO estimates
    tau_sro = 0

    sro_est = 0
    # Estimate the SRO segment-wisely every seg_shift samples
    for seg_idx in range(num_segments):
        # Estimate of the SRO-induced integer shift to be compensated
        shift = int(np.round(tau_sro))

        # Check if an acoustic source is active in all signal segments
        # needed to calculate the current product of complex conjugated
        # coherence functions. The average coherence product is only
        # updated if an acoustic source is active for at least
        # src_activity_th * seg_len samples within each considered signal
        # segment.
        start = seg_idx * seg_shift + temp_dist
        activity_seg = \
            activity_sig[start+shift:start+shift+seg_len]
        activity_seg_ref = activity_ref_sig[start:start+seg_len]
        start_delayed = seg_idx * seg_shift
        activity_seg_delayed = \
            activity_sig[start_delayed+shift:start+shift+seg_len]
        activity_seg_ref_delayed = \
            activity_ref_sig[start_delayed:start_delayed+seg_len]
        activity = (
                np.sum(activity_seg_ref_delayed) > src_activity_th
                and np.sum(activity_seg_ref) > src_activity_th
                and np.sum(activity_seg_delayed) > src_activity_th
                and np.sum(activity_seg) > src_activity_th
        )

        if activity:
            # Calculate the coherence Gamma(seg_idx*seg_shift,k). Note
            # that the segment taken from sig starts at
            # (seg_idx*seg_shift+shift) in order to coarsely compensate
            # the SRO induced delay.
            start = seg_idx * seg_shift + temp_dist
            seg_ref = ref_sig[start:start+seg_len]
            seg = sig[start+shift:start+shift+seg_len]
            coherence = calc_coherence(seg, seg_ref, sro_est)

            # Calculate the coherence Gamma(seg_idx*seg_shift-temp_dist,k).
            # Note that the segment taken from sig starts at
            # (seg_idx*seg_shift-temp_dist+shift) in order to coarsely
            # compensate the SRO induced delay.
            start_delayed = seg_idx * seg_shift
            seg_ref_delayed = \
                ref_sig[start_delayed:start_delayed+seg_len]
            seg_delayed = \
                sig[start_delayed+shift:start_delayed+shift+seg_len]
            coherence_delayed = \
                calc_coherence(seg_delayed, seg_ref_delayed, sro_est)

            # Calculate the complex conjugated product of consecutive
            # coherence functions for the considered frequency range
            coherence_product = coherence * np.conj(coherence_delayed)

            # Note that the used STFT exploits the symmetry of the FFT of
            # real valued input signals and computes only the non-negative
            # frequency terms. Therefore, the negative frequency terms
            # have to be added.
            coherence_product = np.concatenate(
                [coherence_product[:-1],
                    np.conj(coherence_product)[::-1][:-1]],
                -1
            )

            # Update the average coherence product
            avg_coh_prod = (alpha * avg_coh_prod
                            + (1 - alpha) * coherence_product)
                            
            # Interpret the coherence product as generalized cross power
            # spectral density, use an efficient golden section search
            # to find the time lag which maximizes the corresponding
            # generalized cross correlation and derive the SRO from the
            # time lag.
            sro_est = - max_time_lag_search(avg_coh_prod) / temp_dist
        if seg_idx > settling_time - 1:
            sro_estimates[seg_idx] = sro_est
        if seg_idx == settling_time - 1:
            sro_estimates[:seg_idx + 1] = sro_est

        # Use the current SRO estimate to update the estimate for the
        # SRO-induced time shift (The SRO-induced shift corresponds to
        # the average shift of the segment w.r.t. the center of the
        # segment).
        if seg_idx == settling_time - 1:
            # The center of the first segment is given by
            # (seg_len / 2 + temp_dist). The center of the other segments
            # is given by (seg_len / 2 + temp_dist + seg_idx * seg_shift).
            tau_sro += (.5 * seg_len + temp_dist) * sro_est
            tau_sro += seg_shift * sro_est * (settling_time - 1)
        elif seg_idx >= settling_time:
            # The center of the other segments is given by
            # (seg_len / 2 + temp_dist + seg_idx * seg_shift)
            tau_sro += seg_shift * sro_est

        # If the end of the next segment from sig is larger than the length
        # of sig stop SRO estimation.
        nxt_end = ((seg_idx + 1) * seg_shift
                    + temp_dist + int(np.round(tau_sro))
                    + seg_len)
        if nxt_end > len(sig):
            return sro_estimates[:seg_idx + 1] * 1e6
    return sro_estimates * 1e6

    
def owacd_sro_estimation():
    """Online Weighted Average Coherence Drift (WACD)-based SRO
    estimation.
    
    Parameters
    ----------

    Returns
    -------

    References
    ----------
    [1] Chinaev, Aleksej, et al. "Online Estimation of Sampling Rate Offsets
        in Wireless Acoustic Sensor Networks with Packet Loss."
        2021 29th European Signal Processing Conference (EUSIPCO). IEEE, 2021.
    """

    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 

    residualSROs = None

    return residualSROs
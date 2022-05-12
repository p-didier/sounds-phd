"""ABOUT
Script to test out an STFT implementation of Gburrek's DWACD SRO-estimation method,
originally implemented in the time-domain in the corresponding public Github repository. 
We use Pyroomasync-generated SRO-affected signals.

Ref: Gburrek, Tobias, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach. 
    "On Synchronization of Wireless Acoustic Sensor Networks in the 
    Presence of Time-Varying Sampling Rate Offsets and Speaker Changes."
    ICASSP 2022-2022 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP). IEEE, 2022.

Pyroomasync: https://github.com/ImperialCollegeLondon/sap-pyroomasync/tree/main/pyroomasync
DWACD: https://github.com/fgnt/paderwasn
"""

import sys
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pyroomasync import (
    ConnectedShoeBox,
    simulate
)
from paderwasn.synchronization.sro_estimation import DynamicWACD
from paderwasn.synchronization.utils import VoiceActivityDetector
from paderbox.transform import STFT
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search

@dataclass
class ExperimentParameters():
    """Small dataclass to easily store experiment parameters."""
    roomDims: np.ndarray
    micPos: np.ndarray
    fsOffsets: np.ndarray
    sourcePos: np.ndarray
    pathToSignals: str
    vadThreshold : float


SROset = np.linspace(start=0, stop=100, num=5)     # SROs to test
SROset = np.array([50])     # SROs to test
TEMPDIST = 8192


def main():
    """Main wrapper -- Generate signals and compute SRO estimates."""

    for ii in range(len(SROset)):
        print(f'Estimating SROs (set {ii+1}/{len(SROset)})...')

        # Get parameters
        params = gen_parameters(SROset[ii], baseFs=48000)
        vad = VoiceActivityDetector(params.vadThreshold)

        # Get signals
        signals = gen_signals(params)

        # (1) Apply DWACD (batch mode) in time-domain
        sroEstimator = DynamicWACD(temp_dist=TEMPDIST)
        tmp = sroEstimator(sig=signals[:, 0], ref_sig=signals[:, 1],
                                    activity_sig=vad(signals[:, 0]),
                                    activity_ref_sig=vad(signals[:, 1]))
        if ii == 0:
            sroEstimateDWACD = np.zeros((len(SROset), len(tmp)))
        sroEstimateDWACD[ii, :] = tmp

        # (2) Apply DWACD (batch mode) in STFT-domain
        sroEstimator = STFTDynamicWACD(temp_dist=TEMPDIST)
        tmp = sroEstimator(sig=signals[:, 0], ref_sig=signals[:, 1],
                                    activity_sig=vad(signals[:, 0]),
                                    activity_ref_sig=vad(signals[:, 1]))
        if ii == 0:
            sroEstimateDWACD_STFT = np.zeros((len(SROset), len(tmp)))
        sroEstimateDWACD_STFT[ii, :] = tmp

    print('ALL DONE.')

    # Plot
    plot_sro_est(sroEstimateDWACD, sroEstimateDWACD_STFT, SROset)


def gen_parameters(sro, baseFs=48000):
    """Generate parameters for current test."""

    fsOffset = sro * 1e-6 * baseFs

    params = ExperimentParameters(
        roomDims=np.array([4.47, 5.13, 3.18]),
        micPos=np.array([[3.40, 2.10, 0.72],
                        [3.35, 3.41, 0.72]]),
        fsOffsets=np.array([0, fsOffset]),   # sampling freq. offset [samples]
        sourcePos=np.array([1.98, 0.61, 0]),
        pathToSignals='97_tests/02_toolboxes_testing/01_pyroomasync/data',
        vadThreshold = 1
    )

    return params


def gen_signals(params: ExperimentParameters):
    """Generate asynchronized signals using Pyroomasync."""

    # Create room
    room = ConnectedShoeBox(params.roomDims)

    # Add microphones with their sampling frequencies and latencies
    for ii in range(params.micPos.shape[0]):
        room.add_microphone(params.micPos[ii, :], fs_offset=params.fsOffsets[ii], delay=0, id=f'mic{ii+1}')

    # Add a source
    room.add_source(params.sourcePos, "02_data/00_raw_signals/01_speech/speech1.wav", 'source1')

    # Add point to point room impulse responses (one for each source-microphone pair)
    room.add_rir(f"{params.pathToSignals}/ace/Chromebook_EE_lobby_1_RIR.wav", mic_id='mic1', source_id='source1')
    room.add_rir(f"{params.pathToSignals}/ace/Chromebook_EE_lobby_2_RIR.wav", mic_id='mic2', source_id='source1')

    # simulate and get the results recorded in the microphones
    signals = simulate(room)
    signals = signals.T 

    return signals


def plot_sro_est(sroEstimateDWACD, sroEstimateDWACD_STFT, trueSROs):
    """Plot the SRO estimates along with their ground truth."""

    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    for ii in range(sroEstimateDWACD.shape[0]):
        ax.plot(sroEstimateDWACD[ii, :], 'r', label=f'DWACD ($t$): $\\varepsilon={np.round(trueSROs[ii], 1)}$ ppm')
        ax.plot(sroEstimateDWACD_STFT[ii, :], 'b', label=f'DWACD (STFT): $\\varepsilon={np.round(trueSROs[ii], 1)}$ ppm')
        plt.axhline(y=trueSROs[ii], color='k', linestyle='--')
    ax.grid()
    ax.set_ylabel('[ppm]')
    ax.set_xlabel('Online SRO estimation iteration index')
    # ax.set_ylim([-5, 1.1 * np.amax(trueSROs)])
    plt.legend()
    plt.title(f'(In black: true values) $l_\\mathrm{{d}} = {int(TEMPDIST)}$ samples.')
    plt.tight_layout()	
    plt.show()

    return None


class STFTDynamicWACD:
    def __init__(self,
                 seg_len=8192,
                 seg_shift=2048,
                 frame_shift_welch=512,
                 fft_size=4096,
                 temp_dist=8192,
                 alpha=.95,
                 src_activity_th=.75,
                 settling_time=40):
        """Dynamic weighted average coherence drift (DWACD) method

        Sampling rate offset (SRO) estimator for dynamic scenarios with
        time-varying SROs and position changes of the acoustic source from
        "On Synchronization of Wireless Acoustic Sensor Networks in the
        presence of Time-Varying Sampling Rate Offsets and Speaker Changes"
        (Note that moving sources cannot be handled)

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
        """
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.frame_shift_welch = frame_shift_welch
        self.fft_size = fft_size
        self.temp_dist = temp_dist
        self.stft = STFT(shift=frame_shift_welch, size=fft_size,
                         window_length=fft_size, pad=False, fading=False)
        self.src_activity_th = src_activity_th * self.seg_len
        self.settling_time = settling_time
        self.alpha = alpha

    def __call__(self, sig, ref_sig, activity_sig, activity_ref_sig):
        """Estimate the SRO of the single channel signal sig w.r.t. the single
        channel reference signal ref_sig

        Args:
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

        # Convert to STFT domain
        sigSTFT = self.stft(sig)
        ref_sigSTFT = self.stft(ref_sig)
        
        # Useful quantities
        Nl = sigSTFT.shape[0]      # number of STFT frames (calculated using `self.frame_shift_welch` as frame shift)
        Nf = sigSTFT.shape[1]      # number of STFT bins
        Nl_segShift = self.seg_shift // self.frame_shift_welch      # number of STFT frames per segment shift 
        Nl_segLen   = self.seg_len // self.frame_shift_welch        # number of STFT frames per segment 
        Nl_cohDelay = self.temp_dist // self.frame_shift_welch      # number of STFT frames corresponding to the time 
                                                                    # interval between two consecutive coherence functions
        nSegs = (Nl - Nl_cohDelay) // Nl_segShift     # total number of segments

        # Init
        sro_estimates = np.zeros(nSegs)
        avg_coh_prod = np.zeros(self.fft_size)
        tau_sro = 0
        sro_est = 0

        # Loop
        activity = np.zeros(nSegs)
        for seg_idx in range(nSegs):
            print(f'Estimating SRO at segment {seg_idx+1}/{nSegs}...')

            # Estimate of the SRO-induced integer shift to be compensated
            shift = int(np.round(tau_sro))
            # Corresponding STFT frequency shift (Linear Phase Drift model)
            phaseShift = np.exp( 1j * 2 * np.pi / Nf * np.arange(Nf) * shift)

            # Define STFT frame indices for current segment
            lStart = seg_idx * Nl_segShift + Nl_cohDelay
            lEnd = lStart + Nl_segLen 
            # idxSTFTframeEnd = idxSTFTframeStart + 1

            # Segment activity
            nStart = lStart * self.frame_shift_welch
            nEnd = lEnd * self.frame_shift_welch + self.fft_size
            activity_seg = activity_sig[nStart:nEnd]
            activity_seg_ref = activity_ref_sig[nStart:nEnd]
            nStart_delayed = (lStart - Nl_cohDelay) * self.frame_shift_welch
            nEnd_delayed = (lEnd - Nl_cohDelay) * self.frame_shift_welch + self.fft_size
            activity_seg_delayed = activity_sig[nStart_delayed:nEnd_delayed]
            activity_seg_ref_delayed = activity_ref_sig[nStart_delayed:nEnd_delayed]

            thresholdVAD = self.src_activity_th / self.seg_len * (nEnd - nStart + 1)
            activity[seg_idx] = (
                    np.sum(activity_seg_ref_delayed) > thresholdVAD
                    and np.sum(activity_seg_ref) > thresholdVAD
                    and np.sum(activity_seg_delayed) > thresholdVAD
                    and np.sum(activity_seg) > thresholdVAD
            )

            if activity[seg_idx]:
                print(f'Activity detected at segment {seg_idx+1}, updating estimates...')

                # Compute coherence directly from STFT domain signals
                seg_ref = ref_sigSTFT[lStart:lEnd, :] 
                seg = sigSTFT[lStart:lEnd, :] * phaseShift
                coherence = self.calc_coherence_stft(seg, seg_ref, sro_est)

                seg_ref_delayed = ref_sigSTFT[(lStart - Nl_cohDelay):(lEnd - Nl_cohDelay), :]
                seg_delayed = sigSTFT[(lStart - Nl_cohDelay):(lEnd - Nl_cohDelay), :] * phaseShift
                coherence_delayed = self.calc_coherence_stft(seg_delayed, seg_ref_delayed, sro_est)

                # Compute complex conjugate product
                coherence_product = coherence * np.conj(coherence_delayed)

                # Prep for ISTFT (negative frequency bins too)
                coherence_product = np.concatenate(
                    [coherence_product[:-1],
                     np.conj(coherence_product)[::-1][:-1]],
                    -1
                )

                # Update the average coherence product
                avg_coh_prod = (self.alpha * avg_coh_prod
                                + (1 - self.alpha) * coherence_product)

                sro_est = - max_time_lag_search(avg_coh_prod) / self.temp_dist 
            if seg_idx > self.settling_time - 1:
                sro_estimates[seg_idx] = sro_est
            if seg_idx == self.settling_time - 1:
                sro_estimates[:seg_idx + 1] = sro_est
            
            
            # Use the current SRO estimate to update the estimate for the
            # SRO-induced time shift (The SRO-induced shift corresponds to
            # the average shift of the segment w.r.t. the center of the
            # segment).
            if seg_idx == self.settling_time - 1:
                # The center of the first segment is given by
                # (seg_len / 2 + temp_dist). The center of the other segments
                # is given by (seg_len / 2 + temp_dist + seg_idx * seg_shift).
                tau_sro += (.5 * self.seg_len + self.temp_dist) * sro_est
                tau_sro += self.seg_shift * sro_est * (self.settling_time - 1)
            elif seg_idx >= self.settling_time:
                # The center of the other segments is given by
                # (seg_len / 2 + temp_dist + seg_idx * seg_shift)
                tau_sro += self.seg_shift * sro_est

            # If the end of the next segment from sig is larger than the length
            # of sig stop SRO estimation.
            nxt_end = ((seg_idx + 1) * self.seg_shift
                       + self.temp_dist + int(np.round(tau_sro))
                       + self.seg_len)
            if nxt_end > len(sig):
                return sro_estimates[:seg_idx + 1] * 1e6

        return sro_estimates * 1e6

    def calc_psd_stft(self, stft_seg_i, stft_seg_j, sro=0.):
        """Estimate the (cross) power spectral density (PSD) from the given
        signal segments using a Welch method, starting from the STFT domain.

        Args:
            stft_seg_i (array-like):
                Matrix with seg_len elements corresponding to the segment taken
                from the i-th signal, in the STFT domain
            stft_seg_j (array-like):
                Matrix with seg_len elements corresponding to the segment taken
                from the j-th signal, in the STFT domain
            sro (float):
                SRO to be compensated in the Welch method
        Returns:
            psd_ij (numpy.ndarray):
                PSD of the the i-th signal and the j-th signal
        """
        shifts = sro * self.frame_shift_welch * np.arange(len(stft_seg_j))
        stft_seg_j *= \
            np.exp(1j * 2 * np.pi / self.fft_size
                   * np.arange(self.fft_size // 2 + 1)[None] * shifts[:, None])
        return np.mean(stft_seg_i * np.conj(stft_seg_j), axis=0)

    def calc_coherence_stft(self, seg, seg_ref, sro):
        """Estimate the coherence from the given signal segments, 
        directly in the STFT domain.

        Args:
            seg (array-like):
                Matrix with seg_len elements corresponding to the segment taken
                from signal whose SRO should be estimated, in the STFT domain
            seg_ref (array-like):
                Matrix with seg_len elements corresponding to the segment taken
                from the reference signal, in the STFT domain
            sro (float):
                SRO to be compensated when calculating the PSDs needed for
                coherence estimation
        Returns:
            gamma (numpy.ndarray):
                Coherence of the signal and the reference signal
        """
        cpsd = self.calc_psd_stft(seg_ref, seg, sro)
        psd_ref_sig = self.calc_psd_stft(seg_ref, seg_ref)
        psd_sig = self.calc_psd_stft(seg, seg)
        gamma = cpsd / (np.sqrt(psd_ref_sig * psd_sig) + 1e-13)
        return gamma



# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
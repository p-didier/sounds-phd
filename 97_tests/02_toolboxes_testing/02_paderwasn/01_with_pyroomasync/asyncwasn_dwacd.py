"""ABOUT
Script to test out Pyroomasync-generated SRO-affected signals 
in combination with Gburrek's DWACD SRO-estimation method.

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
from paderwasn.synchronization.sro_estimation import DynamicWACD, OnlineWACD
from paderwasn.synchronization.utils import VoiceActivityDetector

@dataclass
class ExperimentParameters():
    """Small dataclass to easily store experiment parameters."""
    roomDims: np.ndarray
    nodePos: np.ndarray
    fsOffsets: np.ndarray
    sourcePos: np.ndarray
    pathToSignals: str
    vadThreshold : float
    comp : dict
    randomGen : np.random.Generator

    def __post_init__(self):
        if len(self.comp['Mk']) != self.nodePos.shape[0]:
            raise ValueError(f'There must be as many Mk values as nodes ({self.nodePos.shape[0]}).')

# General parameters
roomDims = np.array([4.47, 5.13, 3.18])
nodePos = np.array([[3.40, 2.10, 0.72], [3.35, 3.41, 0.72]])
sourcePos = np.array([1.98, 0.61, 0])
pathToSignals ='97_tests/02_toolboxes_testing/01_pyroomasync/data'
vadThreshold = 1
baseFs=48000

SROset = np.linspace(start=0, stop=100, num=5)     # SROs to test
SROset = np.array([50])     # SROs to test
mode = 'batch'      # batch-mode simulation: giving entire signal as input to `paderwasn` functions
# mode = 'online'     # online-mode simulation: giving signal chunk by chunk as input to `paderwasn` functions
TEMPDIST = 8192
# TEMPDIST = int(8192 / 2)
# TEMPDIST = int(8192 / 4)
# TEMPDIST = int(8192 / 8)
COMPUTEONLINE = False
SEED = 12345
multichannelCompression = dict([
    ('active', True),
    ('Mk', [1, 5]),
    ('minSensorDistance', 0.05),
    ('maxSensorDistance', 0.2)
])


def main():
    """Main wrapper -- Generate signals and compute SRO estimates."""

    params = ExperimentParameters(
        roomDims=roomDims,
        nodePos=nodePos,
        fsOffsets=np.array([0, SROset[0] * 1e-6 * baseFs]),   # sampling freq. offset [samples]
        sourcePos=sourcePos,
        pathToSignals=pathToSignals,
        vadThreshold=vadThreshold,
        comp=multichannelCompression,
        randomGen=np.random.default_rng(SEED)
    )


    for ii in range(len(SROset)):
        print(f'Estimating SROs (set {ii+1}/{len(SROset)})...')

        # Set sampling frequency offsets
        params.fsOffsets = np.array([0, SROset[ii] * 1e-6 * baseFs])   # sampling freq. offset [samples]

        # Compute VAD
        vad = VoiceActivityDetector(params.vadThreshold)

        # Generate signals
        signals = gen_signals(params)

        if params.comp['active']:
            signalsCompressed = np.zeros((signals.shape[0], params.nodePos.shape[0]))
            for k in range(params.nodePos.shape[0]):
                linearCompressor = params.randomGen.uniform(size=(params.comp['Mk'][k],)) +\
                    1j * params.randomGen.uniform(size=(params.comp['Mk'][k],)) 
                signalsCompressed[:, k] = np.einsum('ij,ij->i', linearCompressor[np.newaxis, :], signals[:, sum(params.comp['Mk'][:k]):(sum(params.comp['Mk'][:k + 1]))])
            signals = signalsCompressed

        if mode == 'batch':
            # Apply DWACD (batch mode)
            sroEstimator = DynamicWACD(temp_dist=TEMPDIST)
            tmp = sroEstimator(sig=signals[:, 0], ref_sig=signals[:, 1],
                                        activity_sig=vad(signals[:, 0]),
                                        activity_ref_sig=vad(signals[:, 1]))
            if ii == 0:
                sroEstimateDWACD = np.zeros((len(SROset), len(tmp)))
            sroEstimateDWACD[ii, :] = tmp
            if COMPUTEONLINE:
                # Apply Online WACD
                sroEstimator = OnlineWACD(temp_dist=TEMPDIST)
                tmp = sroEstimator(sig=signals[:, 0], ref_sig=signals[:, 1])
                if ii == 0:
                    sroEstimateOWACD = np.zeros((len(SROset), len(tmp)))
                sroEstimateOWACD[ii, :] = tmp
            else:
                sroEstimateOWACD = None

        elif mode == 'online':
            # Apply DWACD (online mode)
            sroEstimator = DynamicWACD()
            num_segments = int(
                (signals.shape[0] - sroEstimator.temp_dist - sroEstimator.seg_len + sroEstimator.seg_shift)
                // sroEstimator.seg_shift
            )
            if ii == 0:
                sroEstimateDWACD = np.zeros((len(SROset), num_segments))

            for idxseg in range(num_segments):
                idxBeg = idxseg * sroEstimator.seg_shift
                idxEnd = idxBeg + sroEstimator.seg_len + sroEstimator.temp_dist + sroEstimator.seg_len - sroEstimator.seg_shift
                # SRO estimate over current segment
                tmp = sroEstimator(sig=signals[idxBeg:idxEnd, 0], ref_sig=signals[idxBeg:idxEnd, 1],
                                            activity_sig=vad(signals[idxBeg:idxEnd, 0]),
                                            activity_ref_sig=vad(signals[idxBeg:idxEnd, 1]))

                sroEstimateDWACD[ii, idxseg] = tmp[0]

    print('ALL DONE.')

    # Plot
    plot_sro_est(sroEstimateDWACD, sroEstimateOWACD, SROset)


def gen_signals(params: ExperimentParameters):
    """Generate asynchronized signals using Pyroomasync."""

    # Create room
    room = ConnectedShoeBox(params.roomDims)

    # Add microphones with their sampling frequencies and latencies
    for ii in range(params.nodePos.shape[0]):
        if params.comp['active']:
            for _ in range(params.comp['Mk'][ii]):
                posOffset = (params.comp['maxSensorDistance'] - params.comp['minSensorDistance'])\
                     * params.randomGen.uniform(size=(3,)) + params.comp['minSensorDistance']
                room.add_microphone(params.nodePos[ii, :] + posOffset, fs_offset=params.fsOffsets[ii], delay=0, id=f'mic{ii+1}')
        else:
            room.add_microphone(params.nodePos[ii, :], fs_offset=params.fsOffsets[ii], delay=0, id=f'mic{ii+1}')

    # Add a source
    room.add_source(params.sourcePos, "02_data/00_raw_signals/01_speech/speech1.wav", 'source1')

    # Add point to point room impulse responses (one for each source-microphone pair)
    room.add_rir(f"{params.pathToSignals}/ace/Chromebook_EE_lobby_1_RIR.wav", mic_id='mic1', source_id='source1')
    room.add_rir(f"{params.pathToSignals}/ace/Chromebook_EE_lobby_2_RIR.wav", mic_id='mic2', source_id='source1')

    # simulate and get the results recorded in the microphones
    signals = simulate(room)
    signals = signals.T 

    return signals


def plot_sro_est(sroEstimateDWACD, sroEstimateOWACD, trueSROs):
    """Plot the SRO estimates along with their ground truth."""

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    for ii in range(sroEstimateDWACD.shape[0]):
        ax.plot(sroEstimateDWACD[ii, :], 'r', label=f'DWACD: $\\varepsilon={np.round(trueSROs[ii], 1)}$ ppm')
        if COMPUTEONLINE:
            ax.plot(sroEstimateOWACD[ii, :], 'b', label=f'Online WACD: $\\varepsilon={np.round(trueSROs[ii], 1)}$ ppm')
        plt.axhline(y=trueSROs[ii], color='k', linestyle='--')
    ax.grid()
    ax.set_ylabel('[ppm]')
    ax.set_xlabel('Online SRO estimation iteration index')
    ax.set_ylim([-5, 1.1 * np.amax(trueSROs)])
    plt.legend()
    plt.title(f'(In black: true values) $l_\\mathrm{{d}} = {int(TEMPDIST)}$ samples.')
    plt.tight_layout()	
    plt.show()

    return None


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
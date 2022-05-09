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
from sqlalchemy import true

@dataclass
class ExperimentParameters():
    roomDims: np.ndarray
    micPos: np.ndarray
    fsOffsets: np.ndarray
    sourcePos: np.ndarray
    pathToSignals: str
    vadThreshold : float

SROset = np.linspace(start=0, stop=100, num=10)     # SROs to test


def main():

    for ii in range(len(SROset)):
        print(f'Estimating SROs (set {ii+1}/{len(SROset)})...')

        # Get parameters
        params = gen_parameters(SROset[ii], baseFs=48000)
        vad = VoiceActivityDetector(params.vadThreshold)

        # Get signals
        signals = gen_signals(params)

        # Apply DWACD
        sroEstimator = DynamicWACD()
        tmp = sroEstimator(sig=signals[:, 0], ref_sig=signals[:, 1],
                                    activity_sig=vad(signals[:, 0]),
                                    activity_ref_sig=vad(signals[:, 1]))
        if ii == 0:
            sroEstimateDWACD = np.zeros((len(SROset), len(tmp)))
        sroEstimateDWACD[ii, :] = tmp

    print('ALL DONE.')

    plot_sro_est(sroEstimateDWACD, SROset)
    

    stop = 1
    
    return None


def gen_parameters(sro, baseFs=48000):

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


def plot_sro_est(sroEsts, trueSROs):

    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    for ii in range(sroEsts.shape[0]):
        ax.plot(sroEsts[ii, :], label=f'$\\varepsilon={trueSROs[ii]}$ ppm')
        plt.axhline(y=trueSROs[ii], color='k', linestyle='--')
    ax.grid()
    ax.set_ylabel('[ppm]')
    plt.legend()
    plt.tight_layout()	
    plt.show()

    return None


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------

from cmath import exp
import sys
from pathlib import Path, PurePath

rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}')

PATHTOROOT = pathToRoot
SIGNALSPATH = f'{PATHTOROOT}/02_data/00_raw_signals'

import numpy as np
import random
import pyroomacoustics as pra
from danse.siggen.classes import *
import danse.siggen.utils as sig_ut
import danse.danse_toolbox.d_base as base
import danse.danse_toolbox.d_core as core
import danse.danse_toolbox.d_post as pp
from danse.danse_toolbox.d_eval import DynamicMetricsParameters
from danse.danse_toolbox.d_classes import *
from dataclasses import dataclass

@dataclass
class TestParameters:
    # TODO: vvv self-noise
    selfnoiseSNR: int = -50 # [dB] microphone self-noise SNR
    wasn: WASNparameters = WASNparameters()
    danseParams: DANSEparameters = DANSEparameters()
    exportFolder: str = Path(__file__).parent   # folder to export outputs
    seed: int = 12345   # random generator seed

    def __post_init__(self):
        np.random.seed(self.seed)  # set random seed
        random.seed(self.seed)  # set random seed
        #
        self.testid = f'J{self.wasn.nNodes}Mk{list(self.wasn.nSensorPerNode)}Nn{self.wasn.nNoiseSources}Nd{self.wasn.nDesiredSources}T60_{int(self.wasn.t60)}ms'
        # Check consistency
        if self.danseParams.nodeUpdating == 'sym' and\
            any(self.wasn.SROperNode != 0):
            raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')

# DEFINE TEST PARAMETERS
PARAMS = TestParameters(
    selfnoiseSNR=-99,  # TODO:
    wasn=WASNparameters(
        rd=np.array([5, 5, 5]),
        fs=16000,
        t60=0.2,
        nNodes=2,
        sigDur=5,
        nSensorPerNode=[1, 1],
        desiredSignalFile=[f'{SIGNALSPATH}/01_speech/{file}'\
            for file in [
                'speech1.wav',
                'speech2.wav'
            ]],
        noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}'\
            for file in [
                'whitenoise_signal_1.wav',
                'whitenoise_signal_2.wav'
            ]],
    ),
    danseParams=DANSEparameters(
        DFTsize=1024,
        WOLAovlp=.5,
        nodeUpdating='seq',
        # nodeUpdating='asy',
        broadcastType='fewSamples',
        # broadcastType='wholeChunk',
        dynMetrics=DynamicMetricsParameters()
    )
)
# Include WASN parameters to DANSE parameters
PARAMS.danseParams.get_wasn_info(PARAMS.wasn)
# Set export folder
PARAMS.exportFolder = f'{Path(__file__).parent}/{PARAMS.testid}'

def main():
    """
    Main function for TI-GEVD-DANSE testing.
    """

    # Build acoustic scenario (room)
    room, vad, wetSpeechAtRefSensor = sig_ut.build_room(PARAMS.wasn)

    # Build WASN
    wasn = sig_ut.build_wasn(room,  vad, wetSpeechAtRefSensor, PARAMS.wasn)

    # Run DANSE
    out, wasnUpdated = danse_it_up(wasn, PARAMS)

    # Post-processing (visualization, etc.)
    postprocess(out, wasnUpdated, room, PARAMS)


def danse_it_up(wasn: list[Node], p: TestParameters):
    """
    Container function for prepping signals and launching the DANSE algorithm.

    Parameters
    ----------
    wasn : list of `Node` objects
        WASN under consideration.
    p : `TestParameters` object
        TI-GEVD-DANSE testing parameters.

    Returns
    -------
    out : `DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasn : list of `Node` objects
        WASN under consideration after DANSE processing.
    """

    # Prep for FFTs (zero-pad)
    for k in range(p.wasn.nNodes):  # for each node
        # Derive exponential averaging factor for `Ryy` and `Rnn` updates
        wasn[k].beta = np.exp(np.log(0.5) / \
            (p.danseParams.t_expAvg50p * wasn[k].fs / p.danseParams.Ns))

    # Launch DANSE
    out, wasnUpdated = core.danse(wasn, p.danseParams)

    return out, wasnUpdated


def postprocess(out: pp.DANSEoutputs,
        wasn: list[Node],
        room: pra.room.ShoeBox,
        p: TestParameters):
    """
    Defines the post-processing steps to be undertaken after a DANSE run.
    Using the `danse.danse_toolbox.d_post` [abbrev. `pp`] functions.

    Parameters
    ----------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasn : list of `Node` objects
        WASN under consideration, after DANSE processing.
    room : `pyroomacoustics.room.ShoeBox` object
        Acoustic scenario under consideration.
    p : `TestParameters` object
        Test parameters.
    """

    # Default booleans
    runit = True   # by default, run
    # Check whether export folder exists
    if Path(p.exportFolder).is_dir():
        # Check whether the folder contains something
        if Path(p.exportFolder).stat().st_size > 0:
            inp = input(f'The folder\n"{p.exportFolder}"\ncontains data. Overwrite? [y/[n]]:  ')
            if inp not in ['y', 'Y']:
                runit = False   # don't run
                print('Aborting export.')
    else:
        print(f'Create export folder "{p.exportFolder}".')
        Path(p.exportFolder).mkdir()

    if runit:
        # Export .wav files
        out.export_sounds(wasn, p.exportFolder)

        # Plot (+ export) acoustic scenario (WASN)
        pp.plot_asc(room, p.wasn, p.exportFolder)

        # Plot performance metrics (+ export)
        out.plot_perf(wasn, p.exportFolder)

        # Plot signals at specific nodes (+ export)
        out.plot_sigs(wasn, p.exportFolder)

    stop = 1

    return None


if __name__ == '__main__':
    sys.exit(main())

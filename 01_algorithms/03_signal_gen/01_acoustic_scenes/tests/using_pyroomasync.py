import sys
from pathlib import Path

from pyroomasync import (
    ConnectedShoeBox,
    simulate
)

from pyroomasync.utils.visualization import (
    plot_microphone_signals,
    plot_room
)

# `pyroomasync` testing script
# -- Inspired by E. Grinstein's `pyroomasync_usage_example.ipynb` notebook | https://github.com/ImperialCollegeLondon/sap-pyroomasync/tree/main/examples

pathToCurrDir = Path(__file__).parent

def main():

    # Create room
    room = ConnectedShoeBox([4.47, 5.13, 3.18])

    # Add microphones with their sampling frequencies and latencies
    # room.add_microphone([3.40, 2.10, 0.72], fs_offset=32000-room.base_fs, delay=0.2, id="chromebook1")
    # room.add_microphone([3.35, 3.41, 0.72], fs_offset=31999-room.base_fs, delay=0.02, id="chromebook2")
    room.add_microphone([3.40, 2.10, 0.72], fs_offset=32000-room.base_fs, delay=0, id="chromebook1")
    room.add_microphone([3.35, 3.41, 0.72], fs_offset=31000-room.base_fs, delay=0, id="chromebook2")

    # Add a source
    room.add_source([1.98, 0.61, 0], f"{pathToCurrDir}/data/vctk/p225_002.wav", id="fostex")

    # Add point to point room impulse responses (one for each source-microphone pair)
    room.add_rir(f"{pathToCurrDir}/data/ace/Chromebook_EE_lobby_1_RIR.wav", "chromebook1", "fostex")
    room.add_rir(f"{pathToCurrDir}/data/ace/Chromebook_EE_lobby_2_RIR.wav", "chromebook2", "fostex")

    # simulate and get the results recorded in the microphones
    simulation_results = simulate(room)

    show_results(room, simulation_results)


def show_results(room, simulation_results):
    
    plot_room(room)
    plot_microphone_signals(simulation_results)


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
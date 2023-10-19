# Purpose of script:
# TI-DANSE trial from scratch.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
import pyroomacoustics as pra

SEED = 0
ROOM_DIM = 10  # [m]
FS = 16000
NSAMPLES = 1000
M = 10  # Number of microphones
K = 5   # Number of nodes
N = 1024  # STFT window length

def main():
    """Main function (called by default when running script)."""
    
    # Create acoustic scene
    room = create_scene()

    # Create WASN
    wasn = create_wasn(room)
    
    # Run TI-DANSE
    out = ti_danse(wasn)


def ti_danse(wasn):

    raise NotImplementedError


class Node:
    # Class to store node information (from create_wasn())
    def __init__(self, signal, neighbors):
        self.signal = signal
        self.neighbors = neighbors


class WASN:
    # Class to store WASN information (from create_wasn())
    def __init__(self):
        self.nodes: list[Node] = []

    def compute_stft_signals(self, L=N, hop=N // 2, transform=np.fft.rfft):
        self.ySTFT = np.zeros((len(self.nodes), L // 2 + 1, self.nodes[0].signal.shape[1]), dtype=np.complex)
        for k in range(len(self.nodes)):
            self.ySTFT[k, :, :] = pra.transform.stft(
                self.nodes[k].signal,
                L=L,
                hop=hop,
                transform=transform
            )


def create_wasn(room: pra.ShoeBox):

    # Randomly arrange the microphones across `K` nodes, such that each
    # node has at least one microphone.
    mics = room.mic_array.R.T
    np.random.shuffle(mics)
    mics = mics[:K, :]
    mics = mics.T

    # Define adjacency matrix such that each node is connected to its
    # closest neighbor.
    adj = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                adj[i, j] = np.linalg.norm(mics[:, i] - mics[:, j])

    # Create neighbors lists
    neighbors = []
    for i in range(K):
        neighbors.append(np.argsort(adj[i, :])[:2])

    # Create nodes and WASN
    wasn = WASN()
    for i in range(K):
        node = Node(room.mic_array.signals[:, i], neighbors[i])
        wasn.nodes.append(node)
    wasn.compute_stft_signals()

    return wasn


def create_scene():

    room = pra.ShoeBox(
        [ROOM_DIM, ROOM_DIM, ROOM_DIM], fs=FS, materials=pra.Material(0.5, 0.5)
    )

    # Add source
    room.add_source(
        np.random.rand(3) * ROOM_DIM, signal=np.random.randn(NSAMPLES)
    )

    # Add microphones
    for _ in range(M):
        room.add_microphone(np.random.rand(3) * ROOM_DIM)

    # Compute RIRs
    room.compute_rir()

    # Simulate
    room.simulate()

    return room



if __name__ == '__main__':
    sys.exit(main())
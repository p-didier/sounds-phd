# Purpose of script:
# Basic tests on a rank-1 data model for the DANSE algorithm and the MWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import resampy
import scipy.linalg as la
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

TARGET_SIGNAL = 'danse/tests/sigs/01_speech/speech2_16000Hz.wav'
N_SENSORS = 10
SELFNOISE_POWER = 1
DURATIONS = np.logspace(np.log10(1), np.log10(30), 50)
# DURATIONS = [20]
FS = 16e3
N_MC = 10

# FILTER_TYPE = 'mwf'  # 'mwf' or 'gevdmwf'
FILTER_TYPE = 'gevdmwf'  # 'mwf' or 'gevdmwf'

SEED = 0

def main(
        M=N_SENSORS,
        durations=DURATIONS,
        fs=FS,
        nMC=N_MC,
        selfNoisePower=SELFNOISE_POWER,
        seed=SEED
    ):
    """Main function (called by default when running script)."""

    # Set random seed
    np.random.seed(seed)

    diff = np.zeros((nMC, len(durations)))
    scalings = np.random.uniform(low=50, high=100, size=M)
    # scalings = np.random.uniform(low=0.5, high=1, size=M)
    # Get clean signals
    nSamplesMax = int(np.amax(durations) * fs)
    cleanSigs, latentSignal = get_clean_signals(M, np.amax(durations), scalings, fs)
    sigma_sr = np.sqrt(np.mean(cleanSigs ** 2, axis=0))
    for idxMC in range(nMC):
        print(f'Running Monte-Carlo iteration {idxMC+1}/{nMC}')

        # Generate noise signals
        noiseSignals = np.zeros((nSamplesMax, M))
        sigma_nr = np.zeros(M)
        for n in range(M):
            # Generate random sequence with unit power
            randSequence = np.random.normal(size=nSamplesMax)
            randSequence = randSequence / np.sqrt(np.mean(randSequence ** 2))
            # Scale to desired power
            noiseSignals[:, n] = randSequence * np.sqrt(selfNoisePower)
            # Check power
            sigma_nr[n] = np.sqrt(np.mean(noiseSignals[:, n] ** 2))
            if np.abs(sigma_nr[n] ** 2 - selfNoisePower) > 1e-6:
                raise ValueError(f'Noise signal power is {sigma_nr[n] ** 2} instead of {selfNoisePower}')
            
        # Loop over durations
        for ii in range(len(durations)):
            # Generate noisy signals
            nSamples = int(durations[ii] * fs)
            noisySignals = cleanSigs[:nSamples, :] + noiseSignals[:nSamples, :]

            # MWF
            filter = compute_filter(
                noisySignals,
                cleanSigs[:nSamples, :],
                noiseSignals[:nSamples, :],
                type=FILTER_TYPE
            )

            # Compute difference between normalized estimated filters
            # and normalized expected filters
            diffsPerSensor = np.zeros(M)
            for n in range(M):
                rtf = scalings / scalings[n]
                hs = np.sum(rtf ** 2) * sigma_sr[n] ** 2
                spf = hs / (hs + sigma_nr[n] ** 2)  # spectral post-filter
                fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter
                diffsPerSensor[n] = np.mean(np.abs(filter[:, n] - fasAndSPF))
            diff[idxMC, ii] = np.mean(diffsPerSensor)

        # Plots
        if 0:
            filteredSignals = np.zeros((nSamples, M))
            filteredSignal_RTFs = np.zeros((nSamples, M))
            for n in range(M):
                filteredSignals[:, n] = noisySignals @ filter[:, n]
                # Compute filtered signal using RTFs
                h = scalings / scalings[n]
                h2 = np.sum(h ** 2)
                hs = h2 * sigma_sr[n] ** 2
                spf = hs / (hs + sigma_nr[n] ** 2)
                filteredSignal_RTFs[:, n] = noisySignals @ h / h2 * spf
            # plot_results(
            #     cleanSigs[:nSamples, :],
            #     noisySignals,
            #     filteredSignals,
            #     filteredSignal_RTFs
            # )
            plot_filter(filter, scalings, sigma_sr, sigma_nr)
            stop = 1

    # Plot difference
    plt.figure()
    plt.loglog(durations, diff.T, '-', color='0.75')
    plt.loglog(durations, np.mean(diff, axis=0), '.-', color='k')
    plt.grid(which='both')
    plt.xlabel('Signal duration (s)')
    plt.ylabel(f'Abs. diff. bw. estimated and expected filter ({nMC} MC runs)')
    plt.show(block=False)


    stop = 1


def plot_filter(filter, scalings, sigma_sr, sigma_nr):
    fig, axs = plt.subplots(1, 1)
    for n in range(filter.shape[1]):
        # axs.plot(filter[:, n] * np.sum(scalings ** 2), '.-', label=f'MWF weights sensor {n}')
        axs.plot(filter[:, n], f'C{n}.-', label=f'MWF weights sensor {n}')
        # axs.plot(filter[:, n] / np.amax(np.abs(filter[:, n])), '.-', label=f'MWF weights sensor {n}')
        h = scalings / scalings[n]
        hs = np.sum(h ** 2) * sigma_sr[n] ** 2
        spectralPostFilter = hs / (hs + sigma_nr[n] ** 2)
        if n == 0:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--', label='Matched BF and spectral post-filter')
        else:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--')
    # axs.plot(scalings / np.amax(np.abs(scalings)), '.--', label='Signal scalings')
    # axs.hlines(1, 0, filter.shape[0] - 1, color='k', linestyle='--', linewidth=0.5)
    fig.legend()
    plt.show(block=False)


def plot_results(cleanSigs, noisySignals, filteredSignals, filteredSignal_RTFs):
    # Plot
    nSensors = cleanSigs.shape[1]
    nRows = int(np.floor(np.sqrt(nSensors)))
    nCols = int(np.ceil(nSensors / nRows))
    fig, axs = plt.subplots(nRows, nCols, sharex=True, sharey=True)
    delta = 2 * np.amax(np.abs(noisySignals))
    for n in range(nSensors):
        if nRows == 1:
            currAx = axs[n % nCols]
        else:
            currAx = axs[int(np.floor(n / nCols)), n % nCols]
        # overlay 1
        currAx.plot(noisySignals[:, n], 'k', label=f'Noisy signal')
        currAx.plot(filteredSignals[:, n], 'r', label=f'MWF-filtering')
        currAx.plot(cleanSigs[:, n], 'y', label=f'Clean signal')
        # overlay 2
        currAx.plot(noisySignals[:, n] + delta, 'k', label=f'Noisy signal')
        currAx.plot(filteredSignal_RTFs[:, n] + delta, 'r', label=f'RTF-filtering')
        currAx.plot(cleanSigs[:, n] + delta, 'y', label=f'Clean signal')
        if n == 0:
            currAx.legend()
        currAx.set_title(f'Sensor {n}')
    plt.show(block=False)


def compute_filter(
        noisySignals: np.ndarray,
        cleanSigs: np.ndarray,
        noiseOnlySigs: np.ndarray,
        type='mwf',
        rank=1
    ):
    """
    [1] Santiago Ruiz, Toon van Waterschoot and Marc Moonen, "Distributed
    combined acoustic echo cancellation and noise reduction in wireless
    acoustic sensor and actuator networks" - 2022
    """

    nSensors = cleanSigs.shape[1]
    Ryy = noisySignals.T @ noisySignals
    if type == 'mwf':
        RyyInv = np.linalg.inv(Ryy)
        w = np.zeros((nSensors, nSensors))
        for n in range(nSensors):
            Ryd = noisySignals.T @ cleanSigs[:, n]
            w[:, n] = RyyInv @ Ryd
    elif type == 'gevdmwf':
        Rnn = noiseOnlySigs.T @ noiseOnlySigs
        sigma, Xmat = la.eigh(Ryy, Rnn)
        idx = np.flip(np.argsort(sigma))
        sigma = sigma[idx]
        Xmat = Xmat[:, idx]
        Qmat = np.linalg.inv(Xmat.conj().T)
        Dmat = np.zeros((nSensors, nSensors))
        Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
        w = Xmat @ Dmat @ Qmat.T.conj()   # see eq. (24) in [1]
    return w


def get_clean_signals(M, dur, scalings, fsTarget):
    # Load target signal
    latentSignal, fs = sf.read(TARGET_SIGNAL)
    if fs != fsTarget:
        # Resample
        latentSignal = resampy.resample(
            latentSignal,
            fs,
            fsTarget
        )
    # Truncate
    latentSignal = latentSignal[:int(dur * fsTarget)]
    # Normalize
    # latentSignal = latentSignal / np.max(np.abs(latentSignal))
    # Normalize power
    latentSignal = latentSignal / np.sqrt(np.mean(latentSignal ** 2))
    
    # Generate clean signals
    mat = np.tile(latentSignal, (M, 1)).T
    # Random scalings
    cleanSigs = mat @ np.diag(scalings)  # rank-1 matrix (scalings only)

    return cleanSigs, latentSignal

if __name__ == '__main__':
    sys.exit(main())
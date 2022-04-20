import sys, copy
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
#
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("_general_fcts" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_general_fcts')
import VAD
if not any("_third_parties" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/_third_parties')
if not any("01_algorithms/01_NR/02_distributed" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
# Custom packages imports
from danse_utilities.classes import ProgramSettings
from danse_utilities.setup import generate_signals
import danse_utilities.danse_subfcns as subs


# General parameters
ascBasePath = f'{pathToRoot}/02_data/01_acoustic_scenarios'
signalsPath = f'{pathToRoot}/02_data/00_raw_signals'
# Set experiment settings
mySettings = ProgramSettings(
    samplingFrequency=16000,
    acousticScenarioPath=f'{ascBasePath}/tests/J2Mk[1, 1]_Ns1_Nn1/AS1_anechoic',
    desiredSignalFile=[f'{signalsPath}/01_speech/speech1.wav'],
    noiseSignalFile=[f'{signalsPath}/02_noise/whitenoise_signal_1.wav'],
    #
    signalDuration=20,
    baseSNR=5,
    chunkSize=2**10,            # DANSE iteration processing chunk size [samples]
    chunkOverlap=0.5,           # Overlap between DANSE iteration processing chunks [/100%]
    SROsppm=0,
    #
    selfnoiseSNR=-50
    )


def main():

    print(mySettings)

    # Generate base signals (and extract acoustic scenario)
    mySignals, asc = generate_signals(mySettings)
    # Compute STFTs
    y, f, t = get_the_stfts(mySignals, mySettings)

    # -------------------------------------------------
    # -------------------------------------------------
    # Useful variables
    N = len(mySettings.stftWin)     # number of samples per frame
    Ns = mySettings.stftEffectiveFrameLen   # number of new samples per frame
    K = asc.numNodes            # number of nodes in network
    nf = f.shape[0] # number of frequency lines
    nFrames = y.shape[1]     # expected number of frames
    # Hard-coded variables
    u = 0   # updating node index (if sequential updating)
    beta = 0.9889709163809315
    lambda_ext = beta ** 10
    alpha = 0.5
    extUpdateEvery = 3  # [s]
    #
    # bypassSmoothing = True
    bypassSmoothing = False
    #
    sequential = False
    # sequential = True
    # -------------------------------------------------
    # -------------------------------------------------
    
    if bypassSmoothing:
        # ↓↓↓↓ No smoothing of filter estimates ↓↓↓↓
        extUpdateEvery = 0  # [s]
        lambda_ext = 0
        alpha = 1
        # ↑↑↑↑ No smoothing of filter estimates ↑↑↑↑

    # Compute VAD
    thrsVAD = np.amax(mySignals.drySpeechSources[:, 0] ** 2) / mySettings.VADenergyFactor
    vad, _ = VAD.oracleVAD(mySignals.drySpeechSources[:, 0], mySettings.VADwinLength, thrsVAD, mySettings.samplingFrequency)
    # Compute frame-wise VAD
    oVADframes = np.zeros(nFrames, dtype=bool)
    for l in range(nFrames):
        VADinFrame = vad[l * Ns : l * Ns + N]
        nZeros = sum(VADinFrame == 0)
        oVADframes[l] = nZeros <= N / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

    # Init parameters
    _, _, _, frameSize, _, _, _, neighbourNodes = subs.danse_init(mySignals.sensorSignals, mySettings, asc)
    nSecondsPerFrame = frameSize / mySettings.samplingFrequency 

    # Initialize arrays
    Ryy = []
    Rnn = []
    w = []
    wExternal = []
    wExternalTarget = []
    d = []
    zmkfull = []
    dimYTilde = np.zeros(K, dtype=int)
    for k in range(K):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
        sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Ryy.append(np.tile(sliceTilde, (nf, 1, 1)))                    # noise only
        Rnn.append(np.tile(sliceTilde, (nf, 1, 1)))                    # speech + noise
        tmp = np.zeros((nf, dimYTilde[k]), dtype=complex)
        tmp[:, 0] = 1
        w.append(tmp)
        tmp = np.zeros((nf, asc.numSensorPerNode[k]), dtype=complex)
        tmp[:, 0] = 1
        wExternalTarget.append(tmp)
        tmp = np.zeros((nf, asc.numSensorPerNode[k]), dtype=complex)
        tmp[:, 0] = 1
        wExternal.append(tmp)
        d.append(np.zeros((nf, nFrames), dtype=complex))
        zmkfull.append(np.zeros((nf, nFrames), dtype=complex))
    nUpdatesRyy = np.zeros(K)
    nUpdatesRnn = np.zeros(K)
    nFilterUpdates = np.zeros(K)
    lastExtUpdateFrameIdx = np.zeros(K)


    # Online processing -- loop over time frames
    for l in range(nFrames):
        if l % 10 == 0:
            print(f'Processing frame {l+1}/{nFrames}...')

        ycurr = y[:, l, :]      # current signals frame

        zmk = [np.empty((nf, 0), dtype=complex) for _ in range(K)]

        # Generate compressed (`z`) signals
        z = np.empty((nf, 0), dtype=complex)
        for k in range(K):
            yk = ycurr[:, asc.sensorToNodeTags == k + 1]
            zk = np.einsum('ij,ij->i', wExternal[k].conj(), yk)     # zq = wqq^H * yq
            # zk = yk[:, 0]
            z = np.concatenate((z, zk[:, np.newaxis]), axis=1)
        
        for k in range(K):
            if not sequential:
                update = True
            elif sequential:
                if k == u:
                    update = True
                else:
                    update = False

            zmk = copy.copy(z)
            zmk = np.delete(zmk, k, axis=1)
            ytildecurr = np.concatenate((ycurr[:, asc.sensorToNodeTags == k + 1], zmk), axis=1)

            if update:
                yyH = np.einsum('ij,ik->ijk', ytildecurr, ytildecurr.conj())
                if oVADframes[l]:
                    Ryy[k] = beta * Ryy[k] + (1 - beta) * yyH
                    nUpdatesRyy[k] += 1
                else:
                    Rnn[k] = beta * Rnn[k] + (1 - beta) * yyH
                    nUpdatesRnn[k] += 1

                if nUpdatesRyy[k] > dimYTilde[k] and nUpdatesRnn[k] > dimYTilde[k]:
                    w[k], _, sigma = subs.perform_gevd_noforloop(Ryy[k], Rnn[k], rank=1, refSensorIdx=0)
                    wExternal[k] = lambda_ext * wExternal[k] + (1 - lambda_ext) * wExternalTarget[k]
                    nFilterUpdates[k] += 1
            else:
                pass    # do not update `w[k]`

            # External filters for compression
            if l - lastExtUpdateFrameIdx[k] >= np.ceil(extUpdateEvery / nSecondsPerFrame):
                wExternalTarget[k] = (1 - alpha) * wExternalTarget[k] + alpha * w[k][:, :asc.numSensorPerNode[k]]
                lastExtUpdateFrameIdx[k] = l

            # Desired signal
            d[k][:, l] = np.einsum('ij,ij->i', w[k].conj(), ytildecurr)

        u = (u + 1) % K     # update updating node index (for sequential processing)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Plot ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kshow = 0   # index of node to plot

    climHigh = np.amax(20 * np.log10(np.abs(y[:, :, kshow])))
    climLow = np.amin(20 * np.log10(np.abs(y[:, :, kshow])))
    exts = [t[0], t[-1], f[-1,kshow], f[0,kshow]]   # plot axes extents
    if climLow < -150:
        climLow = -150
    fig = plt.figure(figsize=(6,2))
    ax = fig.add_subplot(121)
    mapp = ax.imshow(20 * np.log10(np.abs(y[:, :, kshow])), vmin=climLow, vmax=climHigh, extent=exts)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('$f$ [Hz]')
    ax.set_title('Ref. mic. signal')
    plt.colorbar(mapp)
    ax = fig.add_subplot(122)
    mapp = ax.imshow(20 * np.log10(np.abs(d[kshow])), vmin=climLow, vmax=climHigh, extent=exts)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('$f$ [Hz]')
    ax.set_title('Estimated desired signal')
    plt.colorbar(mapp)
    plt.tight_layout()
    plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    stop = 1
    

def get_the_stfts(mySignals, mySettings):
    """Compute sensor signals STFTs given input settings."""
    # Compute STFTs
    nSensors = mySignals.sensorSignals.shape[-1]
    for channel in range(nSensors):

        fcurr, t, tmp = sig.stft(mySignals.sensorSignals[:, channel],
                            fs=mySettings.samplingFrequency,
                            window=mySettings.stftWin,
                            nperseg=mySettings.stftWinLength,
                            noverlap=int(mySettings.stftFrameOvlp * mySettings.stftWinLength),
                            return_onesided=True)
        if channel == 0:
            y = np.zeros((tmp.shape[0], tmp.shape[1], nSensors), dtype=complex)
            f = np.zeros((tmp.shape[0], nSensors))
        y[:, :, channel] = tmp
        f[:, channel] = fcurr

    return y, f, t


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------

import sys, time
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("01_algorithms/01_NR/02_distributed" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_utilities.classes import Signals, AcousticScenario, ProgramSettings
from danse_utilities import danse_subfcns as subs
from danse_utilities.setup import prep_for_ffts


def gevd_mwf_batch(signals: Signals, asc: AcousticScenario, settings: ProgramSettings, plotit=False):
    """Batch-mode GEVD-MWF.
    
    Parameters
    ----------
    signals : Signals object
        The microphone signals and their relevant attributes.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.


    Returns
    -------
    DfiltGEVD
    Dfilt
    """

    # y = np.zeros((signals.sensorSignals.shape[0], signals.numNodes))
    # for ii in range(signals.numNodes):
    #     tags = asc.sensorToNodeTags == ii+1
    #     idx = 0   # default
    #     for jj in range(len(tags)):
    #         if tags[jj] == True:
    #             idx = jj
    #     y[:, ii] = signals.sensorSignals[:, idx]

    y = signals.sensorSignals

    # fig = plt.figure(figsize=(8,4))
    # ax = fig.add_subplot(111)
    # for ii in range(y.shape[-1]):
    #     ax.plot(y[:, ii] + ii * 2 * np.amax(y))
    # ax.grid()
    # plt.tight_layout()	
    # plt.show()

    # Get STFTs
    for k in range(y.shape[-1]):
        f, t, out = sig.stft(y[:, k],
                            fs=asc.samplingFreq,
                            window=settings.stftWin,
                            nperseg=settings.stftWinLength,
                            noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
                            return_onesided=True)
        if k == 0:
            ystft = np.zeros((out.shape[0], out.shape[1], y.shape[-1]), dtype=complex)
        ystft[:, :, k] = out

    # Get VAD per frame
    nFrames = len(t)
    oVADframes = np.zeros(nFrames)
    for l in range(nFrames):
        idxBegChunk = int(l * settings.stftWinLength * (1 - settings.stftFrameOvlp))
        idxEndChunk = int(idxBegChunk + settings.stftWinLength)
        # Compute VAD
        VADinFrame = signals.VAD[idxBegChunk:idxEndChunk]
        oVADframes[l] = sum(VADinFrame == 0) <= settings.stftWinLength / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

    # Batch-mode spatial autocorrelation matrices
    w = np.zeros((len(f), y.shape[-1]), dtype=complex)
    Dfilt = np.zeros((len(f), len(t)), dtype=complex)
    DfiltGEVD = np.zeros((len(f), len(t)), dtype=complex)
    for kappa in range(len(f)):
        Ryy = np.mean(np.einsum('ij,ik->ijk', ystft[kappa, oVADframes==1, :], ystft[kappa, oVADframes==1, :].conj()), axis=0)
        Rnn = np.mean(np.einsum('ij,ik->ijk', ystft[kappa, oVADframes==0, :], ystft[kappa, oVADframes==0, :].conj()), axis=0)

        # NO GEVD
        ed = np.zeros(Ryy.shape[-1])
        ed[0] = 1
        w[kappa, :] = np.linalg.pinv(Ryy) @ (Ryy - Rnn) @ ed
        Dfilt[kappa,:] = ystft[kappa,:,:] @ w[kappa, :].conj()

        # GEVD
        sigma, Xmat = scipy.linalg.eigh(Ryy, Rnn)
        # Flip Xmat to sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigma))
        sigma = sigma[idx]
        Xmat = Xmat[:, idx]

        # Reference sensor selection vector 
        Evect = np.zeros((Ryy.shape[-1],))
        Evect[settings.referenceSensor] = 1

        Qmat = np.linalg.inv(Xmat.conj().T)
        # Sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigma))
        GEVLs_yy = np.flip(np.sort(sigma))
        Qmat = Qmat[:, idx]
        diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:settings.GEVDrank]])   # rank <GEVDrank> approximation
        diagveig = np.append(diagveig, np.zeros(Ryy.shape[0] - settings.GEVDrank))
        # LMMSE weights
        w[kappa, :] = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect

        # Apply filter
        DfiltGEVD[kappa,:] = ystft[kappa,:,:] @ w[kappa, :].conj()

    if plotit:
        # Get colorbar limits
        climHigh = np.amax(np.concatenate((20*np.log10(np.abs(ystft[:,:,0])), 20*np.log10(np.abs(Dfilt)), 20*np.log10(np.abs(DfiltGEVD))), axis=0))
        climLow = np.amin(np.concatenate((20*np.log10(np.abs(ystft[:,:,0])), 20*np.log10(np.abs(Dfilt)), 20*np.log10(np.abs(DfiltGEVD))), axis=0))

        # Plot
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(311)
        mapp = plt.imshow(20*np.log10(np.abs(ystft[:,:,0])), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title('Original reference mic signal')
        ax = fig.add_subplot(312)
        mapp = plt.imshow(20*np.log10(np.abs(Dfilt)), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title('After batch-MWF')
        ax = fig.add_subplot(313)
        mapp = plt.imshow(20*np.log10(np.abs(DfiltGEVD)), extent=[t[0], t[-1], f[-1], f[0]], vmin=climLow, vmax=climHigh)
        ax.invert_yaxis()
        ax.set_aspect('auto')
        plt.colorbar(mapp)
        plt.title(f'After batch-GEVD-MWF (rank {settings.GEVDrank})')
        plt.tight_layout()	
        plt.show()

        stop = 1

    return DfiltGEVD, Dfilt


# def danse_batch(signals: Signals, asc: AcousticScenario, settings: ProgramSettings):


#     raise ValueError('NOT FINISHED -- 20220704')

#     # Prepare signals for Fourier transforms
#     y, t, _ = prep_for_ffts(signals, asc, settings)
    
#     # Initialization (extracting/defining useful quantities)
#     _, winWOLAanalysis, winWOLAsynthesis, frameSize, nExpectedNewSamplesPerFrame, numIterations, _, neighbourNodes = subs.danse_init(y, settings, asc)

#     # Get STFTs
#     for k in range(y.shape[-1]):
#         f, t, out = sig.stft(y[:, k],
#                             fs=asc.samplingFreq,
#                             window=settings.stftWin,
#                             nperseg=settings.stftWinLength,
#                             noverlap=int(settings.stftFrameOvlp * settings.stftWinLength),
#                             return_onesided=True)
#         if k == 0:
#             ystft = np.zeros((out.shape[0], out.shape[1], y.shape[-1]), dtype=complex)
#         ystft[:, :, k] = out

#     # Get VAD per frame
#     nFrames = len(t)
#     oVADframes = np.zeros(nFrames)
#     for l in range(nFrames):
#         idxBegChunk = int(l * settings.stftWinLength * (1 - settings.stftFrameOvlp))
#         idxEndChunk = int(idxBegChunk + settings.stftWinLength)
#         # Compute VAD
#         VADinFrame = signals.VAD[idxBegChunk:idxEndChunk]
#         oVADframes[l] = sum(VADinFrame == 0) <= settings.stftWinLength / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

#     # ---------------------- Arrays initialization ----------------------
#     lk = np.zeros(asc.numNodes, dtype=int)                      # node-specific broadcast index
#     i = np.zeros(asc.numNodes, dtype=int)                       # !node-specific! DANSE iteration index
#     #
#     wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
#     Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
#     Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
#     ryd = []                                        # cross-correlation between observations and estimations
#     ytilde = []                                     # local full observation vectors, time-domain
#     ytildeHat = []                                  # local full observation vectors, frequency-domain
#     z = []                                          # current-iteration compressed signals used in DANSE update
#     zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
#     dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
#     oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
#     numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
#     if settings.computeLocalEstimate:
#         wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
#         Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
#         Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
#         dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
        
#     for k in range(asc.numNodes):
#         dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])
#         wtmp = np.zeros((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex)
#         wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
#         wTilde.append(wtmp)
#         ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
#         ytildeHat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
#         #
#         sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
#         Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
#         Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
#         ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
#         #
#         z.append(np.empty((frameSize, 0), dtype=float))
#         zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
#         #
#         if settings.computeLocalEstimate:
#             dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
#             wtmp = np.zeros((numFreqLines, numIterations + 1, dimYLocal[k]), dtype=complex)
#             wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
#             wLocal.append(wtmp)
#             sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
#             Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
#             Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
#     # Desired signal estimate [frames x frequencies x nodes]
#     dhat = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
#     d = np.zeros((y.shape[0], asc.numNodes))  # time-domain version of `dhat`
#     dhatLocal = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)
#     dLocal = np.zeros((y.shape[0], asc.numNodes))  # time-domain version of `dhatLocal`

#     # Autocorrelation matrices update counters
#     numUpdatesRyy = np.zeros(asc.numNodes)
#     numUpdatesRnn = np.zeros(asc.numNodes)
#     minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
#     # Booleans
#     startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
#     # ------------------------------------------------------------------


#     t0 = time.perf_counter()    # loop timing
#     # Parse event matrix (+ inform user)
#     t, eventTypes, nodesConcerned = subs.events_parser(events, startUpdates, printouts=True)

#     # Loop over events
#     for idxEvent in range(len(eventTypes)):

#         # Node index
#         k = int(nodesConcerned[idxEvent])
#         event = eventTypes[idxEvent]

#         # Extract current local data chunk
#         idxEndChunk = int(np.floor(t * fs[k]))
#         idxBegChunk = idxEndChunk - frameSize     # <N> samples long chunk
#         yLocalCurr = y[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]
        
#         # Compute VAD
#         VADinFrame = oVAD[idxBegChunk:idxEndChunk]
#         oVADframes[i[k]] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1

#         # Count number of spatial covariance matrices updates
#         if oVADframes[i[k]]:
#             numUpdatesRyy[k] += 1
#         else:
#             numUpdatesRnn[k] += 1
        
#         # Process buffers
#         z[k], _ = subs.process_incoming_signals_buffers(
#             zBuffer[k],
#             z[k],
#             neighbourNodes[k],
#             i[k],
#             frameSize,
#             N=nExpectedNewSamplesPerFrame,
#             L=settings.broadcastLength,
#             lastExpectedIter=numIterations - 1)

#         # Wipe local buffers
#         zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]

#         # Build full available observation vector
#         yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
#         ytilde[k][:, i[k], :] = yTildeCurr

#         # --------------------- Spatial covariance matrices updates ---------------------
#         # Go to frequency domain
#         ytildeHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(ytilde[k][:, i[k], :] * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
#         ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]      # Keep only positive frequencies

#         Ryytilde[k], Rnntilde[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :],
#                                         Ryytilde[k], Rnntilde[k], settings.expAvgBeta, oVADframes[i[k]])
#         if settings.computeLocalEstimate:
#             # Local observations only
#             Ryylocal[k], Rnnlocal[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :dimYLocal[k]],
#                                             Ryylocal[k], Rnnlocal[k], settings.expAvgBeta, oVADframes[i[k]])
#         # -------------------------------------------------------------------------------
        
#         # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
#         if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
#             startUpdates[k] = True

#         if startUpdates[k] and not settings.bypassFilterUpdates:
#             # No `for`-loop versions
#             if settings.performGEVD:    # GEVD update
#                 wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
#                 if settings.computeLocalEstimate:
#                     wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)

#             else:                       # regular update (no GEVD)
#                 wTilde[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryytilde[k], Rnntilde[k], settings.referenceSensor)
#                 if settings.computeLocalEstimate:
#                     wLocal[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryylocal[k], Rnnlocal[k], settings.referenceSensor)

#         else:
#             # Do not update the filter coefficients
#             wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
#             if settings.computeLocalEstimate:
#                 wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
#         if settings.bypassFilterUpdates:
#             print('!! User-forced bypass of filter coefficients updates !!')
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         # ----- Compute desired signal chunk estimate -----
#         dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
#         # dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :], ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
#         dhat[:, i[k], k] = dhatCurr
#         # Transform back to time domain (WOLA processing)
#         dChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatCurr, frameSize)
#         d[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunk)   # overlap and add construction of output time-domain signal
        
#         if settings.computeLocalEstimate:
#             # Local observations only
#             dhatLocalCurr = np.einsum('ij,ij->i', wLocal[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :dimYLocal[k]])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
#             dhatLocal[:, i[k], k] = dhatLocalCurr
#             # Transform back to time domain (WOLA processing)
#             dLocalChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatLocalCurr, frameSize)
#             dLocal[idxBegChunk:idxEndChunk, k] += np.real_if_close(dLocalChunk)   # overlap and add construction of output time-domain signal
#         # -----------------------------------------------------------------------
        
#         # Increment DANSE iteration index
#         i[k] += 1


#     return 0



import numpy as np
import copy, sys
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
if not any("01_algorithms/01_NR/02_distributed" in s for s in sys.path):
    sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
# Custom packages imports
import danse_utilities.danse_subfcns as subs


def run_danse(y, K, asc, neighbourNodes, nf, nFrames, applyCompression, sequential, oVADframes, beta, lambda_ext, extUpdateEvery, nSecondsPerFrame, alpha):

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
        tmp = np.ones((nf, asc.numSensorPerNode[k]), dtype=complex)     # in bertrand's MATLAB scripts: initialized as all-ones
        # tmp[:, 0] = 1
        wExternalTarget.append(tmp)
        tmp = np.ones((nf, asc.numSensorPerNode[k]), dtype=complex)     # in bertrand's MATLAB scripts: initialized as all-ones
        # tmp[:, 0] = 1
        wExternal.append(tmp)
        d.append(np.zeros((nf, nFrames), dtype=complex))
        zmkfull.append(np.zeros((nf, nFrames), dtype=complex))
    nUpdatesRyy = np.zeros(K)
    nUpdatesRnn = np.zeros(K)
    nFilterUpdates = np.zeros(K)
    lastExtUpdateFrameIdx = np.zeros(K)

    wThroughFrames = []
    sroresidual = []
    for k in range(K):
        wThroughFrames.append(np.zeros((nFrames, w[0].shape[0], w[0].shape[1]), dtype=complex))
        sroresidual.append(np.zeros((nFrames, dimYTilde[k])))

    u = 0   # initialize updating node index
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
            if applyCompression:
                zk = np.einsum('ij,ij->i', wExternal[k].conj(), yk)     # zq = wqq^H * yq
            else:
                zk = yk[:, 0]
            z = np.concatenate((z, zk[:, np.newaxis]), axis=1)
        
        for k in range(K):

            zmk = copy.copy(z)
            zmk = np.delete(zmk, k, axis=1)
            ytildecurr = np.concatenate((ycurr[:, asc.sensorToNodeTags == k + 1], zmk), axis=1)
            
            yyH = np.einsum('ij,ik->ijk', ytildecurr, ytildecurr.conj())
            if oVADframes[l]:
                Ryy[k] = beta * Ryy[k] + (1 - beta) * yyH
                nUpdatesRyy[k] += 1
            else:
                Rnn[k] = beta * Rnn[k] + (1 - beta) * yyH
                nUpdatesRnn[k] += 1

            update = True
            if sequential and k != u:
                update = False

            if update and nUpdatesRyy[k] > dimYTilde[k] and nUpdatesRnn[k] > dimYTilde[k]:
                wapriori = copy.copy(w[k])  # a priori filter
                w[k], _ = subs.perform_gevd_noforloop(Ryy[k], Rnn[k], rank=1, refSensorIdx=0)
                # w[k] = subs.perform_update_noforloop(Ryy[k], Rnn[k], refSensorIdx=0)  # no GEVD method
                nFilterUpdates[k] += 1
            else:
                pass    # do not update `w[k]`
            
            if nFilterUpdates[k] >= 10:
                sroresidual[k][l,:] = subs.cohdrift_sro_estimation(w[k], wapriori, asc.numSensorPerNode[k], 1024, 512)

            # (smoothly) update broadcast filters
            wExternal[k] = lambda_ext * wExternal[k] + (1 - lambda_ext) * wExternalTarget[k]

            wThroughFrames[k][l, :, :] = w[k]

            # External filters for compression
            if l - lastExtUpdateFrameIdx[k] >= np.ceil(extUpdateEvery / nSecondsPerFrame):
                wExternalTarget[k] = (1 - alpha) * wExternalTarget[k] + alpha * w[k][:, :asc.numSensorPerNode[k]]
                lastExtUpdateFrameIdx[k] = l

            # Desired signal
            d[k][:, l] = np.einsum('ij,ij->i', w[k].conj(), ytildecurr)

        u = (u + 1) % K     # update updating node index (for sequential processing)

    return d, wThroughFrames
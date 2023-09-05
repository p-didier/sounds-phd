import numpy as np
from scipy import linalg as la
from .common import *

MAX_ITER_BATCH_DANSE = 100
TOL_BATCH_DANSE = 1e-4

def run_batch_mwf(
        x: np.ndarray,
        n: np.ndarray,
        filterType='regular',
        rank=1,
        vad=None
    ):
    """Batch-mode (GEVD-)MWF."""

    # Get noisy signal (time-domain)
    y: np.ndarray = x + n

    if vad is None:
        nSamples = x.shape[0]
        Ryy = 1 / nSamples * y.T @ y.conj()
        Rnn = 1 / nSamples * n.T @ n.conj()
    else:
        # Include VAD knowledge in batch-mode estimates of `Ryy` and `Rnn`.
        # Only keep frames where all sensors VADs are active.
        vadBool = np.all(vad.astype(bool), axis=1)
        Ryy = 1 / np.sum(vadBool) *\
            y[vadBool, :].T @ y[vadBool, :].conj()
        Rnn = 1 / np.sum(~vadBool) *\
            y[~vadBool, :].T @ y[~vadBool, :].conj()

    # Compute filter
    w = update_filter(
        Ryy, Rnn,
        filterType=filterType,
        rank=rank
    )

    return w


def run_batch_mwf_wola(
        x,
        n,
        filterType='regular',
        rank=1,
        p: WOLAparameters=WOLAparameters(),
        verbose=False,
        vad=None
    ):
    """Batch-mode WOLA-(GEVD-)MWF."""
    # Get noisy signal (time-domain)
    y = x + n
    # Compute WOLA domain signal
    yWola, nWola, vadFramewise = to_wola(p, y, n, vad, verbose)

    # Compute batch-mode covariance matrices
    if vad is None:
        Ryy = compute_scm(yWola)
        Rnn = compute_scm(nWola)
    else:
        # Check if there are enough frames to compute batch-mode estimates
        # using VAD information
        if np.sum(vadFramewise.astype(bool)) < y.shape[1] or\
            np.sum(~vadFramewise.astype(bool)) < y.shape[1]:
            print('[WARNING] VAD-BASED BATCH FILTERS ESTIMATION: not enough VAD=0 and VAD=1 frames to update batch filters.')
            w = np.zeros((y.shape[1], yWola.shape[1], y.shape[1]))
            for k in range(y.shape[1]):
                w[0, :, k] = 1
            return w
        Ryy = compute_scm(yWola[vadFramewise.astype(bool), :, :])
        Rnn = compute_scm(yWola[~vadFramewise.astype(bool), :, :])

    # Compute filter
    w = update_filter(
        Ryy, Rnn,
        filterType=filterType,
        rank=rank
    )

    return w


def run_batch_danse(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        referenceSensorIdx=0,
        verbose=True,
        vad=None
    ):

    maxIter = MAX_ITER_BATCH_DANSE
    tol = TOL_BATCH_DANSE
    # Get noisy signal
    y = x + n
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    # Determine data type (complex or real)
    myDtype = np.complex128 if np.iscomplex(x).any() else np.float64
    # Initialize
    w = []
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        wCurr = np.zeros((nSensorsPerNode + nNodes - 1, maxIter), dtype=myDtype)
        wCurr[referenceSensorIdx, :] = 1
        w.append(wCurr)
    idxNodes = np.arange(nNodes)
    idxUpdatingNode = 0

    # Initialize network-wide filter
    wNet = np.zeros((x.shape[1], maxIter, nNodes), dtype=myDtype)
    for k in range(nNodes):
        idxRef = np.where(channelToNodeMap == k)[0][referenceSensorIdx]
        wNet[idxRef, 0, k] = 1  # set reference sensor weight to 1
    
    # Run DANSE
    if nodeUpdatingStrategy == 'sequential':
        label = 'Batch DANSE [seq NU]'
    else:
        label = 'Batch DANSE [sim NU]'
    if filterType == 'gevd':
        label += ' [GEVD]'
    for iter in range(maxIter):
        if verbose:
            print(f'{label} iteration {iter+1} (max. {maxIter})')
        # Compute fused signals from all sensors
        fusedSignals = np.zeros((x.shape[0], nNodes), dtype=myDtype)
        fusedSignalsNoiseOnly = np.zeros((x.shape[0], nNodes), dtype=myDtype)
        for q in range(nNodes):
            yq = y[:, channelToNodeMap == q]
            fusedSignals[:, q] = yq @ w[q][:yq.shape[1], iter].conj()
            nq = n[:, channelToNodeMap == q]
            fusedSignalsNoiseOnly[:, q] = nq @ w[q][:nq.shape[1], iter].conj()
            
        for k in range(nNodes):
            # Get y tilde
            yTilde: np.ndarray = np.concatenate((
                y[:, channelToNodeMap == k],
                fusedSignals[:, idxNodes != k]
            ), axis=1)
            nTilde: np.ndarray = np.concatenate((
                n[:, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, idxNodes != k]
            ), axis=1)

            # Compute covariance matrices
            if vad is None:
                Ryy = 1 / yTilde.shape[0] * yTilde.T @ yTilde.conj()
                Rnn = 1 / nTilde.shape[0] * nTilde.T @ nTilde.conj()
            else:
                vadBool = np.all(vad.astype(bool), axis=1)
                Ryy = 1 / np.sum(vadBool) *\
                    yTilde[vadBool, :].T @ yTilde[vadBool, :].conj()
                Rnn = 1 / np.sum(~vadBool) *\
                    yTilde[~vadBool, :].T @ yTilde[~vadBool, :].conj()
            
            if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                updateFilter = True
            elif nodeUpdatingStrategy == 'simultaneous':
                updateFilter = True
            else:
                updateFilter = False

            if updateFilter:
                # Compute filter
                if filterType == 'regular':
                    e = np.zeros(w[k].shape[0])
                    e[referenceSensorIdx] = 1  # selection vector
                    ryd = (Ryy - Rnn) @ e
                    w[k][:, iter + 1] = np.linalg.inv(Ryy) @ ryd
                elif filterType == 'gevd':
                    sigma, Xmat = la.eigh(Ryy, Rnn)
                    idx = np.flip(np.argsort(sigma))
                    sigma = sigma[idx]
                    Xmat = Xmat[:, idx]
                    Qmat = np.linalg.inv(Xmat.T.conj())
                    Dmat = np.zeros((Ryy.shape[0], Ryy.shape[0]))
                    Dmat[:rank, :rank] = np.diag(1 - 1 / sigma[:rank])
                    e = np.zeros(Ryy.shape[0])
                    e[referenceSensorIdx] = 1
                    w[k][:, iter + 1] = Xmat @ Dmat @ Qmat.T.conj() @ e
            else:
                w[k][: , iter + 1] = w[k][:, iter]  # keep old filter
            
        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes

        # Compute network-wide filters
        for k in range(nNodes):
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(x.shape[1]):
                # Node index corresponding to channel `m`
                currNode = channelToNodeMap[m]
                # Count channel index within node
                c = channelCount[currNode]
                if currNode == k:
                    # Use local filter coefficient
                    wNet[m, iter + 1, k] = w[currNode][c, iter + 1]
                else:
                    nChannels_k = np.sum(channelToNodeMap == k)
                    gkq = w[k][nChannels_k + neighborCount, iter + 1]
                    wNet[m, iter + 1, k] = w[currNode][c, iter] * gkq
                channelCount[currNode] += 1

                if currNode != k and c == np.sum(channelToNodeMap == currNode) - 1:
                    neighborCount += 1
        
        # Check convergence
        if iter > 0:
            diff = 0
            for k in range(nNodes):
                diff += np.mean(np.abs(w[k][:, iter + 1] - w[k][:, iter]))
            if diff < tol:
                if verbose:
                    print(f'Convergence reached after {iter+1} iterations')
                break

    # Format for output: just keep the iterations that were actually run
    wOut = np.zeros((x.shape[1], iter + 2, nNodes), dtype=myDtype)
    for k in range(nNodes):
        wOut[:, :, k] = wNet[:, :(iter + 2), k]

    return wOut


def run_batch_danse_wola(
        x,
        n,
        channelToNodeMap,      
        filterType='regular',  # 'regular' or 'gevd'
        rank=1,
        nodeUpdatingStrategy='sequential',  # 'sequential' or 'simultaneous'
        referenceSensorIdx=0,
        p: WOLAparameters=WOLAparameters(),
        verbose=True,
        vad=None
    ):

    maxIter = MAX_ITER_BATCH_DANSE
    tol = TOL_BATCH_DANSE

    nSensors = x.shape[1]
    # Get noisy signal
    y = x + n
    # Compute WOLA domain signal
    yWola, nWola, vadFramewise = to_wola(p, y, n, vad, verbose)
    nFrames = yWola.shape[0]
    nPosFreqs = yWola.shape[1]
    # Get number of nodes
    nNodes = np.amax(channelToNodeMap) + 1
    # Initialize
    w = []
    for k in range(nNodes):
        nSensorsPerNode = np.sum(channelToNodeMap == k)
        wCurr = np.zeros((nPosFreqs, maxIter, nSensorsPerNode + nNodes - 1), dtype=complex)
        wCurr[:, :, referenceSensorIdx] = 1
        w.append(wCurr)
    idxNodes = np.arange(nNodes)
    idxUpdatingNode = 0

    # Initialize network-wide filter
    wNet = np.zeros((nSensors, maxIter, nPosFreqs, nNodes), dtype=complex)
    for k in range(nNodes):
        idxRef = np.where(channelToNodeMap == k)[0][referenceSensorIdx]
        wNet[idxRef, 0, :, k] = 1  # set reference sensor weight to 1 at first iteration
    # Run DANSE
    if nodeUpdatingStrategy == 'sequential':
        label = 'Batch WOLA-based DANSE [seq NU]'
    else:
        label = 'Batch WOLA-based DANSE [sim NU]'
    if filterType == 'gevd':
        label += ' [GEVD]'
    for iter in range(maxIter):
        if verbose:
            print(f'{label} iteration {iter+1} (max. {maxIter})')
        # Compute fused signals from all sensors
        fusedSignals = np.zeros(
            (nFrames, nPosFreqs, nNodes),
            dtype=complex
        )
        fusedSignalsNoiseOnly = np.zeros(
            (nFrames, nPosFreqs, nNodes),
            dtype=complex
        )
        for q in range(nNodes):
            yq = yWola[:, :, channelToNodeMap == q]
            fusedSignals[:, :, q] = np.einsum(
                'ijk,jk->ij',
                yq,
                w[q][:, iter, :yq.shape[2]].conj()
            )
            nq = nWola[:, :, channelToNodeMap == q]
            fusedSignalsNoiseOnly[:, :, q] = np.einsum(
                'ijk,jk->ij',
                nq,
                w[q][:, iter, :yq.shape[2]].conj()
            )
    
        for k in range(nNodes):
            # Get y tilde
            yTilde: np.ndarray = np.concatenate((
                yWola[:, :, channelToNodeMap == k],
                fusedSignals[:, :, idxNodes != k]
            ), axis=2)
            nTilde: np.ndarray = np.concatenate((
                nWola[:, :, channelToNodeMap == k],
                fusedSignalsNoiseOnly[:, :, idxNodes != k]
            ), axis=2)

            # Compute covariance matrices
            updateFilter = True
            if vad is None:
                Ryy = compute_scm(yTilde)
                Rnn = compute_scm(nTilde)
            else:
                # Check if there are enough frames to compute batch-mode estimates
                # using VAD information
                if np.sum(vadFramewise.astype(bool)) < yTilde.shape[-1] or\
                    np.sum(~vadFramewise.astype(bool)) < yTilde.shape[-1]:
                    print('[WARNING] VAD-BASED BATCH DANSE FILTERS ESTIMATION: not enough VAD=0 and VAD=1 frames to update batch filters.')
                    updateFilter = False
                else:
                    Ryy = 1 / np.sum(vadFramewise) *\
                        compute_scm(yTilde[vadFramewise.astype(bool), :, :])
                    Rnn = 1 / np.sum(~vadFramewise) *\
                        compute_scm(yTilde[~vadFramewise.astype(bool), :, :])
            
            if updateFilter:
                if nodeUpdatingStrategy == 'sequential' and k == idxUpdatingNode:
                    updateFilter = True
                elif nodeUpdatingStrategy == 'simultaneous':
                    updateFilter = True
                else:
                    updateFilter = False

            if updateFilter:
                # Compute filter
                w[k][:, iter + 1, :] = update_filter(
                    Ryy, Rnn,
                    filterType=filterType,
                    rank=rank,
                    referenceSensorIdx=referenceSensorIdx
                )
            else:
                w[k][:, iter + 1, :] = w[k][:, iter, :]  # keep old filter
            
        # Update node index
        if nodeUpdatingStrategy == 'sequential':
            idxUpdatingNode = (idxUpdatingNode + 1) % nNodes

        # Compute network-wide filters
        for k in range(nNodes):
            Mk = np.sum(channelToNodeMap == k)
            channelCount = np.zeros(nNodes, dtype=int)
            neighborCount = 0
            for m in range(nSensors):
                # Node index corresponding to channel `m`
                currNwIdx = channelToNodeMap[m]
                # Count channel index within node
                cIdx = channelCount[currNwIdx]
                if currNwIdx == k:  # current node channel
                    wNet[m, iter + 1, :, k] = w[currNwIdx][:, iter + 1, cIdx]
                else:  # neighbor node channel
                    gkq = w[k][:, iter + 1, Mk + neighborCount]
                    wNet[m, iter + 1, :, k] = w[currNwIdx][:, iter, cIdx] * gkq
                    # If we have reached the last channel of the current 
                    # neighbor node, increment neighbor count
                    if cIdx == np.sum(channelToNodeMap == currNwIdx) - 1:
                        neighborCount += 1
                channelCount[currNwIdx] += 1
        
        # Check convergence
        if iter > 0:
            diff = 0
            for k in range(nNodes):
                diff += np.mean(
                    np.abs(w[k][:, iter + 1, :] - w[k][:, iter, :]) /\
                        np.amax(np.abs(w[k][:, iter, :]))
                )
            if diff < tol:  # normalize by number of frequency bins
                if verbose:
                    print(f'Convergence reached after {iter+1} iterations')
                break
            if iter == maxIter - 2:
                print(f'WARNING: Batch DANSE did not converge after {maxIter} iterations')
                break

    # Format for output: just keep the iterations that were actually run
    wOut = np.zeros((x.shape[1], iter + 2, nPosFreqs, nNodes), dtype=complex)
    for k in range(nNodes):
        wOut[:, :, :, k] = wNet[:, :(iter + 2), :, k]

    return wOut


def compute_scm(a: np.ndarray):
    """Compute spatial covariance matrix (SCM) based on 
    WOLA-domain signal `a`."""
    return np.mean(np.einsum('ijk,ijl->ijkl', a, np.conj(a)), axis=0)
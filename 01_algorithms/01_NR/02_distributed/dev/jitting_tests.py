from numba import njit
import numpy as np
import scipy.linalg
import time

@njit
def invert_jitted(A):
    B = np.linalg.inv(A)
    return B

@njit
def sortevls_jitted(Qmat, sigma, rank):
    idx = np.flip(np.argsort(sigma))
    GEVLs_yy = np.flip(np.sort(sigma))
    Sigma_yy = np.diag(GEVLs_yy)
    Qmat = Qmat[:, idx]
    diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
    diagveig = np.append(diagveig, np.zeros(Sigma_yy.shape[0] - rank))
    return diagveig, Qmat

@njit
def getw_jitted(Qmat, diagveig, Evect):
    diagveig = diagveig.astype(np.complex128)
    Evect = Evect.astype(np.complex128)
    return np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect



def perform_gevd(Ryy, Rnn, rank=1, refSensorIdx=0, jitted=False, showtimings=False):
    """GEVD computations for DANSE.
    
    Parameters
    ----------
    Ryy : [N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    rank : int
        GEVD rank approximation.
    refSensorIdx : int
        Index of the reference sensor (>=0).

    Returns
    -------
    w : [N x 1] np.ndarray (complex)
        GEVD-DANSE filter coefficients.
    Qmat : [N x N] np.ndarray (complex)
        Hermitian conjugate inverse of the generalized eigenvectors matrix of the pencil {Ryy, Rnn}.
    """
    # Reference sensor selection vector 
    t0 = time.perf_counter()
    Evect = np.zeros((Ryy.shape[0],))
    Evect[refSensorIdx] = 1
    if showtimings:
        print(f'Step 1: {(time.perf_counter() - t0)*1e3}ms')
    # Perform generalized eigenvalue decomposition
    t0 = time.perf_counter()
    sigma, Xmat = scipy.linalg.eigh(Ryy, Rnn)
    if showtimings:
        print(f'Step 2.1: {(time.perf_counter() - t0)*1e3}ms')
    t0 = time.perf_counter()
    if jitted:
        Qmat = invert_jitted(Xmat.conj().T)     # <--- JITTED
    else:
        Qmat = np.linalg.inv(Xmat.conj().T)
    if showtimings:
        print(f'Step 2.2: {(time.perf_counter() - t0)*1e3}ms')
    # Sort eigenvalues in descending order
    t0 = time.perf_counter()
    if jitted:
        diagveig, Qmat = sortevls_jitted(Qmat, sigma, rank)
    else:
        idx = np.flip(np.argsort(sigma))
        GEVLs_yy = np.flip(np.sort(sigma))
        Sigma_yy = np.diag(GEVLs_yy)
        Qmat = Qmat[:, idx]
        diagveig = np.array([1 - 1/sigma for sigma in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
        diagveig = np.append(diagveig, np.zeros(Sigma_yy.shape[0] - rank))
    if showtimings:
        print(f'Step 3: {(time.perf_counter() - t0)*1e3}ms')
    # LMMSE weights
    t0 = time.perf_counter()
    if jitted:
        w = getw_jitted(Qmat, diagveig, Evect)
    else:
        w = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect
    if showtimings: 
        print(f'Step 5: {(time.perf_counter() - t0)*1e3}ms')
    return w, Qmat


n = 10      # full rank
r = 1       # rank approximation
seed = 213532
rng = np.random.default_rng(seed)
mat = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
Ryy = np.dot(mat, mat.conj().T)     # positive definite, full-rank matrix
mat2 = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
Rnn = np.dot(mat2, mat2.conj().T)   # positive definite, full-rank matrix


nMC = 1000

times = np.zeros(nMC)
for ii in range(nMC):
    t0 = time.perf_counter()
    w, Qmat = perform_gevd(Ryy, Rnn, rank=r, jitted=0)
    times[ii] = time.perf_counter() - t0
    if ii == 0 or ii == 1:
        print(f'MC run {ii+1}/{nMC} ({int(times[ii] * 1e6)} microseconds)')
print(f'jitted=0, n={n}: Average timing (over the last {int(nMC/2)} runs): {int(1e6*np.mean(times[int(len(times)/2):]))} microseconds')

times = np.zeros(nMC)
for ii in range(nMC):
    # print(f'MC run {ii+1}/{nMC}')
    t0 = time.perf_counter()
    w, Qmat = perform_gevd(Ryy, Rnn, rank=r, jitted=1)
    times[ii] = time.perf_counter() - t0
    if ii == 0 or ii == 1:
        print(f'MC run {ii+1}/{nMC} ({int(times[ii] * 1e6)} microseconds)')
print(f'jitted=1, n={n}: Average timing (over the last {int(nMC/2)} runs): {int(1e6*np.mean(times[int(len(times)/2):]))} microseconds')

stop = 1
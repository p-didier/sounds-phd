
#%%
import numpy as np
import matplotlib.pyplot as plt


Mk = 3
Nf = 512

w = np.random.random((Nf, Mk)) + 1j * np.random.random((Nf, Mk))
y = np.random.random((Nf, Mk)) + 1j * np.random.random((Nf, Mk))

z1 = np.einsum('ij,ij->i', w.conj(), y)

z2 = np.zeros(Nf, dtype=complex)
for ii in range(Nf):
    z2[ii] = np.dot(w[ii,:].conj().T, y[ii,:])

print(f'z1 shape: {z1.shape}')
print(f'z2 shape: {z2.shape}')
print(f'z1 == z2: {(z1 == z2).all()}')
print(f'z1[1:4]: {z1[1:4]}')
print(f'z2[1:4]: {z2[1:4]}')

plt.plot(np.real(z1), 'r-')
plt.plot(np.imag(z1 + 1), 'r-')
plt.plot(np.real(z2), 'b--')
plt.plot(np.imag(z2 + 1), 'b--')

#%%

import numpy as np
import scipy.signal

L = 1024
R = 0.5
a = np.hanning(L)
scipy.signal.check_NOLA(a, L, L*(1-R))

#%% -- Outer product along rows of 2D matrix to form 3D tensor

import numpy as np

seed = 123421
rng = np.random.default_rng(seed)
n1 = 10
n2 = 15
my2Dmat = rng.random(size=(n1, n2)) + 1j * rng.random(size=(n1, n2))
test = np.einsum('ij,ik->ijk', my2Dmat, my2Dmat.conj())  # update signal + noise matrix

truth = np.zeros((n1,n2,n2), dtype=complex)
for ii in range(n1):
    truth[ii, :, :] = np.outer(my2Dmat[ii, :], my2Dmat[ii, :].conj())

print(test.shape)
print(truth.shape)
print((test == truth).all())

#%% -- slice-wise GEVD on 3D matrix

import numpy as np
import time
import scipy.linalg

seed = 123421
rng = np.random.default_rng(seed)

n = 5
nKappas = 500
rank = 1
Ryy = np.zeros((nKappas, n, n), dtype=complex)
Rnn = np.zeros((nKappas, n, n), dtype=complex)
for kappa in range(nKappas):
    mat = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
    Ryy[kappa, :, :] = np.dot(mat, mat.conj().T)     # positive definite, full-rank matrix
    mat2 = rng.random(size=(n, n)) + 1j * rng.random(size=(n, n))
    Rnn[kappa, :, :] = np.dot(mat2, mat2.conj().T)   # positive definite, full-rank matrix
refSensorIdx = 0

# ------------ Ground truth ------------
t0 = time.perf_counter()
truth = np.zeros((nKappas, n), dtype=complex)
for kappa in range(nKappas):
    # Reference sensor selection vector 
    Evect = np.zeros((Ryy[kappa, :, :].shape[0],))
    Evect[refSensorIdx] = 1
    # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted
    sigma, Xmat = scipy.linalg.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :])

    Qmat = np.linalg.inv(Xmat.conj().T)
    # Sort eigenvalues in descending order
    idx = np.flip(np.argsort(sigma))
    GEVLs_yy = np.flip(np.sort(sigma))
    Sigma_yy = np.diag(GEVLs_yy)
    Qmat = Qmat[:, idx]
    diagveig = np.array([1 - 1/s for s in GEVLs_yy[:rank]])   # rank <GEVDrank> approximation
    diagveig = np.append(diagveig, np.zeros(Sigma_yy.shape[0] - rank))
    # LMMSE weights
    w = np.linalg.inv(Qmat.conj().T) @ np.diag(diagveig) @ Qmat.conj().T @ Evect
    truth[kappa, :] = w
print(f'Time with for-loop ({nKappas} freqs.): {time.perf_counter() - t0}s')
    
# ------------ for-loop-free estimate ------------
t0 = time.perf_counter()
# Reference sensor selection vector 
Evect = np.zeros((Ryy[kappa, :, :].shape[0],))
Evect[refSensorIdx] = 1

sigma = np.zeros((nKappas, n))
Xmat = np.zeros((nKappas, n, n), dtype=complex)
for kappa in range(nKappas):
    # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted
    sigmacurr, Xmatcurr = scipy.linalg.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :])
    # Flip Xmat to sort eigenvalues in descending order
    idx = np.flip(np.argsort(sigmacurr))
    sigma[kappa, :] = sigmacurr[idx]
    Xmat[kappa, :, :] = Xmatcurr[:, idx]
Qmat = np.linalg.inv(np.transpose(Xmat.conj(), axes=[0,2,1]))
# GEVLs tensor
Dmat = np.zeros((nKappas, n, n))
Dmat[:, 0, 0] = np.squeeze(1 - 1/sigma[:, :rank])
# LMMSE weights
QH = np.transpose(Qmat.conj(), axes=[0,2,1])
QmH = np.linalg.inv(QH)
test = np.matmul(np.matmul(np.matmul(QmH, Dmat), QH), Evect)
print(f'Time without for-loop ({nKappas} freqs.): {time.perf_counter() - t0}s')

print('done')
print(truth.shape)
print(test.shape)
print((test == truth).all())


# %%

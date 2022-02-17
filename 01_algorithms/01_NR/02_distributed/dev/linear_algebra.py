
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

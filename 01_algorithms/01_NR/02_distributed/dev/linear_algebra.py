
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



# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('default')  # <-- for Jupyter: white figures background


# Define random generator
seed = 12345
rng = np.random.default_rng(seed)

n = 20
z = rng.random(size=(n,))# + 1j * np.random.random(size=(n,))

Z = np.fft.fft(z, n, axis=0)

# Z2 = Z[:int(n/2)+1]
# Z2 = np.concatenate((Z2, [Z[int(n/2)]], np.flip(Z2.conj())[:-1]))

# Z2 = Z[1:int(n/2)+1]
# Z2 = np.concatenate(([Z[0]],Z2, np.flip(Z2[:-1].conj())[:-1]))

Z2 = Z[:int(n/2+1)]
Z2 = np.concatenate((Z2, np.flip(Z2[:-1].conj())[:-1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Z2.real, 'r')
ax.plot(Z2.imag, 'b')
ax.plot(Z.real, 'g--')
# plt.ylim([-10,10])
ax.plot(Z.imag, 'y--')

z2 = np.fft.ifft(Z2, n)

# print(np.amax(z2.imag))

Z3 = rng.random(size=(int(n/2)+1,)) + 1j * rng.random(size=(int(n/2)+1,))
Z3real = np.concatenate((Z3.real, np.flip(Z3[:-1].real)[:-1]))
Z3imag = np.concatenate((Z3.imag, -np.flip(Z3.imag)))
Z3imag = Z3.imag

Z3 = np.concatenate((Z3, np.flip(Z3[:-1].conj())[:-1]))
# Z3p = Z3real + 1j * Z3imag

z3 = np.fft.ifft(Z3p, n)
print(np.amax(z3.imag))
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(Z3real, 'r')
plt.plot(Z3imag, 'g')
ax = fig.add_subplot(212)
# plt.plot(z3.imag, 'k')



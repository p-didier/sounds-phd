
import numpy as np
import  matplotlib.pyplot as plt
import scipy

# a = np.random.rand(10,15,5)

# dimsidx = range(len(a.shape))
# t = np.argsort(a.shape)
# d = np.take(dimsidx,t)
# x = np.transpose(a, tuple(d))

# print(x.shape)

# for ii in range(x.ndim-1):
#     x = x[0]

# print(len(x))

# a = np.array([1,2,3])
# print(a**2)

# N = 3
# Z = np.random.random((N,N)) + np.random.random((N,N)) * 1j
# print(Z)
# print(Z.conj().T)

# fig, ax = plt.subplots(2,2)
# fig.set_size_inches(4, 4, forward=True)
# plt.show()

# N = int(16e3)
# x = np.random.random(N)
# X = scipy.fft.fft(x)
# x2 = scipy.fft.ifft(X)

N = 5
a = np.identity(N,dtype=complex)
a = np.repeat(a[:, :, np.newaxis], N, axis=2)

stop = 1 
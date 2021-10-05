
import random
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

# N = 5
# a = np.identity(N,dtype=complex)
# a = np.repeat(a[:, :, np.newaxis], N, axis=2)

# stop = 1 



# def mygevd(A,B):

#     # Cholesky factorization
#     G = np.linalg.cholesky(np.real(B))       # B = G*G^H

#     C1 = np.linalg.solve(G,A)
#     C = np.linalg.solve(G.conj(),C1.T)    # C = G^(-1)*A*G^(-H)

#     s, Qa = np.linalg.eig(C)

#     # Sort eigenvalues in descending order
#     idx = np.flip(np.argsort(s))
#     s = np.flip(np.sort(s))
#     S = np.real(np.diag(s))
#     Qa = Qa[:,idx]

#     X = np.linalg.solve(G.conj().T, Qa)   # X = G^(-H)*Qa
#     Q = np.linalg.pinv(X.conj().T)

#     return S,Q

# # COMPARING MATLAB AND PYTHON EIG()
# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# Q = np.array([[5,6,3],[2,3,4],[1,3,4]])

# B = np.dot(Q, Q.transpose())

# S,Q = mygevd(A,B)
# print(S)
# print(Q)



import winsound

# winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
winsound.PlaySound('tmp.wav', winsound.SND_FILENAME)
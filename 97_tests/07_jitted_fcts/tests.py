# Purpose of script:
# Test JIT-ed functions.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from numba import jit

def main():
    """Main function (called by default when running script)."""

    
    
    # Test 1: JIT-ed function for end of GEVD-update
    nFreqs = 1024
    dim = 8
    Xmat = get_random_complex_matrix((nFreqs, dim, dim))
    sigma = get_random_complex_matrix((nFreqs, dim))
    Evect = np.zeros((dim,))
    Evect[0] = 1
    GEVDrank = 1
    # Sanity check
    print('Sanity check...')
    w1 = update_gevd_endbit(Xmat, sigma, Evect, GEVDrank)
    w2 = update_gevd_endbit2(Xmat, sigma, Evect, GEVDrank)
    raise ValueError('[28.04.2023 18:21] Incorrect implemenation --> np.multiply != np.matmul')
    print('w1 == w2: {}'.format(np.allclose(w1, w2)))
    w = jit_update_gevd_endbit(Xmat, sigma, Evect, GEVDrank)

    print('Done.')


def update_gevd_endbit(Xmat, sigma, Evect, GEVDrank):
    XmatT = np.transpose(Xmat.conj(), axes=(0, 2, 1))
    Qmat = np.zeros_like(Xmat)
    for ii in range(XmatT.shape[0]):
        Qmat[ii, :, :] = np.linalg.inv(XmatT[ii, :, :])
    # GEVLs tensor
    Dmat = np.zeros_like(Xmat)
    for ii in range(GEVDrank):
        Dmat[:, ii, ii] = 1 - 1/sigma[:, ii]
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=(0, 2, 1))
    w = np.multiply(np.multiply(np.multiply(Xmat, Dmat), Qhermitian), Evect)

    return w


def update_gevd_endbit2(Xmat, sigma, Evect, GEVDrank):
    XmatT = np.transpose(Xmat.conj(), axes=(0, 2, 1))
    Qmat = np.zeros_like(Xmat)
    for ii in range(XmatT.shape[0]):
        Qmat[ii, :, :] = np.linalg.inv(XmatT[ii, :, :])
    # GEVLs tensor
    Dmat = np.zeros_like(Xmat)
    for ii in range(GEVDrank):
        Dmat[:, ii, ii] = 1 - 1/sigma[:, ii]
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=(0, 2, 1))

    w = np.matmul(np.matmul(np.matmul(Xmat, Dmat), Qhermitian), Evect)

    return w


@jit(nopython=True)
def jit_update_gevd_endbit(Xmat, sigma, Evect, GEVDrank):
    
    XmatT = np.transpose(Xmat.conj(), axes=(0, 2, 1))
    Qmat = np.zeros_like(Xmat)
    for ii in range(XmatT.shape[0]):
        Qmat[ii, :, :] = np.linalg.inv(XmatT[ii, :, :])
    # GEVLs tensor
    Dmat = np.zeros_like(Xmat)
    for ii in range(GEVDrank):
        Dmat[:, ii, ii] = 1 - 1/sigma[:, ii]
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=(0, 2, 1))
    w = np.multiply(np.multiply(np.multiply(Xmat, Dmat), Qhermitian), Evect)

    return w

# Function to get a random complex matrix of arbitrary size
def get_random_complex_matrix(size):
    return np.random.rand(*size) + 1j * np.random.rand(*size)


if __name__ == '__main__':
    sys.exit(main())
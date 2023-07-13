# Purpose of script:
# Test linear algebra property:
# is the following true?
#
# (A * h) / (h^H * A * h) = h / (h^H * h) 
# if A is diagonal and real-valued
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np

SEED = 12345
SIZE = 3

def main():
    """Main function (called by default when running script)."""
    
    np.random.seed(SEED)

    # Generate random matrix A
    Amat = np.random.randn(SIZE, SIZE)
    Amat = np.diag(np.diag(Amat))

    # Generate random complex vector h
    h = np.random.randn(SIZE) + 1j * np.random.randn(SIZE)

    # Compute left hand side
    lhs = (Amat @ h) / (h.conj().T @ Amat @ h)

    # Compute right hand side
    rhs = h / (h.conj().T @ h)

    # Compare
    print('lhs =', lhs)
    print('rhs =', rhs)
    print('lhs == rhs:', np.allclose(lhs, rhs))

    stop = 1

if __name__ == '__main__':
    sys.exit(main())
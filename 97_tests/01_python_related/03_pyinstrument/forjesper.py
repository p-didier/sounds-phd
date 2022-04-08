"""
Timing test for scipy.linalg.eigh().
Requires installed modules scipy, numpy, and pyinstrument.
"""

from pyinstrument import Profiler
import scipy.linalg
import sys
import numpy as np

N_RUNS = 10000
MAT_SIZE = 20
SEED = 12345

def main():

    profiler = Profiler()
    profiler.start()

    run(n=N_RUNS, m=MAT_SIZE, seed=SEED)

    profiler.stop()
    profiler.print()


def run(n, m, seed):

    rng = np.random.default_rng(seed)

    # Generate pos. semi-def. matrices
    myMat = rng.uniform(-10, 10, (m, m))
    myMat1 = np.dot(myMat, myMat.transpose())
    myMat = rng.uniform(-10, 10, (m, m))
    myMat2 = np.dot(myMat, myMat.transpose())

    for ii in range(n):
        a, b = scipy.linalg.eigh(myMat1, myMat2)

    print('Done computing.\n')


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
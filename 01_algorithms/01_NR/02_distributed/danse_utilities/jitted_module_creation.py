from numba.pycc import CC
import numpy as np
# from submodule import my_module

# print(my_module.multf(1, 2))

cc = CC('dist_fct_module')
# Uncomment the following line to print out the compilation steps
cc.verbose = True

@cc.export('get_trace_jitted', 'f8(f8[:,:], i4)')
def get_trace_jitted(A, ofst):
    return np.trace(A, ofst)

@cc.export('get_Amat_jitted', 'f8[:,:](f8[:], f8[:,:], f8[:])')
def get_Amat_jitted(f, H, h):
    return np.diag(f) @ H @ np.diag(h)

if __name__ == "__main__":
    cc.compile()
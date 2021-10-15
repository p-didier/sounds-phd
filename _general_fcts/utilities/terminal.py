import numpy as np

def loop_progress(indices,limits,printit=False):
    # % loop_progress -- Returns the percentage of progress of a given
    # % n-dimensional loop defined by indices and limits.
    # %
    # % >>> Inputs:
    # % -indices [N*1 int array, -] - Indices of the loops, in order from outter
    # %                                loop to inner loop.
    # % -limits [N*1 int array, -] - Corresponding limits (end indices).
    # % -print [bool] - If true, print the progress in the Command Window (default: true).
    # % -prefix [string] - To add at the beginning of every printout if <print>==1 (default: '').
    # % -varargin:
    # %       --'LoopTiming' [float, s] - Time one loop iteration takes (can be
    # %       easily retrieved via <tic> and <toc>.
    # % >>> Outputs:
    # % -p [float, -] - Percentage of completion.

    # (c) Paul Didier - Translated from MATLAB on 12-Oct-2021


    # Total number of iterations
    total_niter = np.prod(limits)

    # Current number of iterations done
    curr_niter = indices[-1] + 1
    for ii in range(len(indices)-1):
        limits_curr = limits[-1-ii:]
        curr_niter += indices[-1-(ii+1)] * np.prod(limits_curr)

    p = curr_niter/total_niter*100

    if printit:
        print('Progress: %.2f%%' % p)

    return p

# TESTS
# limits = [10,20,30]
# for ii in range(limits[0]):
#     for jj in range(limits[1]):
#         for kk in range(limits[2]):
#             loop_progress([ii,jj,kk],limits,True)

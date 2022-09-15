import numpy as np

def extract_few_samples_from_convolution(idDesired, a, b):
    """
    Manually computes convolution between `a` and `b`
    only for the output indices desired (`idDesired`). 

    Parameters
    ----------
    idDesired : [L x 1] np.ndarray (float)
        Indices desired from convolution output.
    a : [N x 1] np.ndarray (float)
        FIR filter (time-domain).
    b : [M x 1] np.ndarray (float)
        Signal to be used for convolution.

    Returns
    -------
    out : [L x 1] np.ndarray (float)
        Output samples from convolution between `a` and `b`.
    """

    out = np.zeros(len(idDesired))
    yqzp = np.concatenate((np.zeros(len(a)), b, np.zeros(len(a))))
    for ii in range(len(idDesired)):
        out[ii] = np.dot(yqzp[idDesired[ii] + 1:idDesired[ii] + 1 + len(a)], np.flip(a))
    
    return out
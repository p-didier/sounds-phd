import numpy as np

def extract_few_samples_from_convolution(idDesired, w, y):
    """
    Manually computes convolution between a filter `w` and a signal `y`
    for just the output indices desired (`idDesired`). 

    Parameters
    ----------
    idDesired : [L x 1] np.ndarray (float)
        Indices desired from convolution output.
    w : [N x 1] np.ndarray (float)
        FIR filter (time-domain).
    y : [M x 1] np.ndarray (float)
        Signal to be used for convolution.

    Returns
    -------
    out : [L x 1] np.ndarray (float)
        Output samples from convolution between `w` and `y`.
    """

    out = np.zeros(len(idDesired))
    yqzp = np.concatenate((np.zeros(len(w)), y, np.zeros(len(w))))
    for ii in range(len(idDesired)):
        out[ii] = np.dot(yqzp[idDesired[ii] + 1:idDesired[ii] + 1 + len(w)], np.flip(w))
    
    return out
import numpy as np

def get_closest(array, values):
    # CREDIT: https://stackoverflow.com/a/46184652/16870850

    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    if not isinstance(idxs, np.int64):
        # find indexes where previous index is closer
        prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
        idxs[prev_idx_is_less] -= 1
    
    return idxs


def normalize_toint16(nparray):
    """Normalizes a NumPy array to integer 16.
    Parameters
    ----------
    nparray : np.ndarray
        Input array to be normalized.

    Returns
    ----------
    nparrayNormalized : np.ndarray
        Normalized array.
    """
    amplitude = np.iinfo(np.int16).max
    nparrayNormalized = (amplitude * nparray / \
        np.amax(nparray) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
    return nparrayNormalized
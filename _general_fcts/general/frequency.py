import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '_general_fcts')))
from general.arrays import get_closest

def noctfr(n,fll,ful,type='exact'):
    # noctfr -- Computation of 1/n octave band center and cutoff frequencies.
    #   Center frequency is 1000 Hz.
    #
    # f,fl,fu = noctfr(n,fll,ful,type)
    #
    # f     center frequencies
    # fl    lower cutoff frequencies
    # fu    upper cutoff frequencies
    # n     octave band division, e.g., when n = 3, one-third octave band
    #       frequencies are computed
    # fll   lower limit of the frequency range of interest. fl <= fll
    # ful   upper limit of the frequency range of interest, fu >= ful
    # type  representation type of band center frequencies. Two valid options:
    #       'exact' for exact center frequency values, and 'nominal' for
    #       nominal center frequency values. The option 'nominal' can only be
    #       chosen when n = 1 or n = 3.
    #
    # (c) Edwin Reynders, KU Leuven, 2014
    # Translation from MATLAB by Paul Didier (Oct. 2021).
    #
    # References:
    # G. Vermeir. Bouwakoestiek, 2e ed., Acco, Leuven, 1999, p. 15.

    fc = 1000 # center frequency

    if fll == 0:
        print('<fll> cannot equal DC (0 Hz): replacing value by 20 Hz.')
        fll = 20

    x = np.floor(n * np.log(fll*2**(1/(2*n))/fc)/np.log(2))
    y = np.ceil(n * np.log(ful*2**(-1/(2*n))/fc)/np.log(2))

    f  = 1000*2**(np.arange(x,y+1)/n)
    fl = f*2**(-1/(2*n))
    fu = f*2**( 1/(2*n))

    if type == 'nominal':
        fn3 = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800] # Nominal 1/3-octave band center frequencies between 100 and 999 Hz. 
                                                                 # To be multiplied or divided by 10 for getting additional nominal frequencies. 
        x = len(str(np.floor(f( 1)))) # count number of significant digits of lowest center frequency, rounded to the nearest lower integer
        y = len(str(np.floor(f(-1)))) # same for the highest center frequency
        fn = []
        for par in np.arange(x,y):
            fn.append(fn3 / 1000*10**par)
        indl = np.argmin(np.abs(fn - f( 1)))
        indu = np.argmin(np.abs(fn - f(-1)))
        if n == 3:         
            f = fn[indl:indu]
        elif n == 1:
            f = fn[indl:3:indu]
        else:
            raise ValueError('<n> should equal 1 or 3 when <type> == "nominal"')

    return f, fl, fu


def divide_in_bands(bandtype, freqs):
    all_f_flag = False   
    if bandtype == 'OTOB':
        fc, fl, fu = noctfr(3, freqs[0], freqs[-1], type='exact')
    elif bandtype == 'OB':
        fc, fl, fu = noctfr(1, freqs[0], freqs[-1], type='exact')
    elif bandtype == None or bandtype == 'all':
        bands_indices = [np.arange(len(freqs))]
        all_f_flag = True
        fc, fl, fu = [0],[0],[0]

    if not all_f_flag:
        # Divide in bands
        bands_indices = []
        for ii, fc_curr in enumerate(fc):
            idx_start = get_closest(freqs, fl[ii])
            idx_end = get_closest(freqs, fu[ii])
            if not isinstance(idx_start, np.int64):
                idx_start = idx_start[0]
            if not isinstance(idx_start, np.int64):
                idx_end = idx_end[0]
            bands_indices.append(np.arange(idx_start, idx_end))

    return bands_indices, fc, fl, fu


def exact_to_norm_OTOBs(fc):
    # exact_to_norm_OTOBs -- Converts exact OTOB centre frequencies to normalised
    # OTOB centre frequencies according to the relevant ISO standards.
    #
    # >>> Inputs:
    # -fc [N*1 float vector, Hz] - OTOB centre frequencies to normalise.
    # >>> Outputs:
    # -fcn [N*1 float vector, Hz] - Normalised centre frequencies.

    # (c) Paul Didier - 10-Nov-2021

    fn = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,\
        1000, 125, 1600, 2000, 3150, 4000, 5000, 6300, 8000, 10e3, 12500, 16e3, 20e3])

    idx = get_closest(fn, fc)
    fcn = np.empty(len(idx))
    fcn[~np.isnan(idx)] = fn[idx[~np.isnan(idx)].tolist()]

    if fc.shape[0] == 1:
        fcn = fcn.T

    return fcn


def norm_to_exact_OTOBs(fcn):
    # norm_to_exact_OTOBs -- Converts normalized OTOB centre frequencies to exact
    # OTOB centre frequencies according to the relevant ISO standards.
    #
    # >>> Inputs:
    # -fc [N*1 float vector, Hz] - Normalized OTOB centre frequencies.
    # >>> Outputs:
    # -fcn [N*1 float vector, Hz] - Exact centre frequencies.

    # (c) Paul Didier - 10-Nov-2021

    fn = noctfr(3, np.amin(fcn)*2**(-1/6), np.amax(fcn)*2**(1/6), 'exact')[0]

    fc = fn[get_closest(fn,fcn)]

    return fc


# TESTS
# freqs = np.arange(0, 8000)
# bands_indices, fc, fl, fu = divide_in_bands('OTOB', freqs)

# fcn = exact_to_norm_OTOBs(fc)
# fc2 = norm_to_exact_OTOBs(fcn)
# print(fc)
# print(fcn)
# print(fc2)

# stop = 1
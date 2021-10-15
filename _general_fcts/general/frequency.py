import numpy as np

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

    f  = 1000*2**(np.arange(x,y)/n)
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


# TESTS
# freqs = np.arange(0, 8000)
# fc, fl, fu = noctfr(n=3, fll=freqs[0], ful=freqs[-1], type='exact')
# stop = 1
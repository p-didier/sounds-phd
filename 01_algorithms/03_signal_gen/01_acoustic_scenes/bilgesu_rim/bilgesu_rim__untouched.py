import numpy as np
from numpy import linalg as LA
import time
import random


def ISM(xr, xs, L, beta, N, Nt, Rd, Sr, Tw, Fc, Fs, c):

    S = 2 * (L[0] * L[1] + L[0] * L[2] + L[1] * L[2])
    V =L[0]*L[1]*L[2]
    alpha = 1 - np.exp(-(24 * V * np.log(10)) / (c * beta * S))
    beta = -np.sqrt(abs(1 - alpha))* np.ones((6, 1))

    L = L / c * Fs * 2
    xr = xr / c * Fs
    xs = xs / c * Fs
    Rd = Rd / c * Fs

    #assert (np.size(xr,0) == 3)
    K = np.size(xr, 1)

    h = np.zeros((Nt, K))

    #if (N.all == 0):
    N = np.floor(Nt/ L) + 1

    if (len(Sr)==0):
        Sr = sum(time.localtime() * 100)

    for k in range (1,K+1):
        #np.random('state', Sr)
        random.seed()
        for u in range (0,2):
            for v in range (0,2):
                for w in range (0,2):
                    for l in range (int(-N[0]),int(N[0])+1):
                        for m in range  (int(-N[1]),int(N[1])+1):
                            for n in range (int(-N[2]),int(N[2])+1):

                                pos_is = [xs[0] - 2 * u * xs[0] + l * L[0],xs[1] - 2 * v * xs[1] + m * L[1],xs[2] - 2 * w * xs[2] + n * L[2]]


                                rand_disp = Rd * (2 * np.random.rand(3) - 1) * np.count_nonzero(sum(np.abs([u,v,w,l,m,n])))
                                d = np.linalg.norm(pos_is + rand_disp - xr[:, k-1])+1



                                if (round(d) > Nt or round(d) < 1):
                                    continue


                                if (Tw == 0):
                                    indx = round(d)
                                    s = 1
                                else:
                                    tmp=int(max(np.ceil(d - Tw / 2), 1))
                                    indx = range(int(max(np.ceil(d - Tw / 2), 1)),int(min(np.floor(d+Tw / 2), Nt)))

                                    s = (1 + np.cos(2 * np.pi * (indx - d) / Tw)) * np.sinc(Fc * (indx - d)) / 2
                                cc=np.abs([l - u, l, m - v, m, n - w, n])
                                cc=cc.reshape(-1,1)
                                bb=np.power(beta ,cc )
                                dd=np.prod(bb)
                                #ccc=np.prod(np.power(beta , np.abs([l - u, l, m - v, m, n - w, n])))
                                A = dd / (4 * np.pi * (d - 1))
                                h[indx, k-1] = h[indx, k-1] + np.transpose(s * A)


    kk=1
    h = np.multiply(h, (Fs/c))


    return (h, Sr)


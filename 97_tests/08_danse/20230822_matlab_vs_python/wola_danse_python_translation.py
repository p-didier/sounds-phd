import numpy as np
from scipy.linalg import pinv
from scipy.signal import hann
from scipy.fftpack import fft, ifft, fftshift

import sys

def main():
    # Test code

    # Parameters
    fs = 16000
    frameLen = 512
    simultaneous = 1
    alpha = 0.7
    mu = 1

    # Generate input
    inp = [np.random.randn(16000, 2) for _ in range(3)]
    onoff = np.zeros(16000)
    onoff[fs * 2:fs * 3] = 1
    onoff[fs * 5:fs * 6] = 1
    onoff[fs * 8:fs * 9] = 1

    # Run DANSE
    out = WOLA_DANSE1(inp, fs, onoff, frameLen, simultaneous, alpha, mu)

    stop =1


def WOLA_DANSE1(Y, fs, onoff, L=None, simultaneous=True, alpha=0.7, mu=1):
    # WRITTEN BY CHATGPT ON 22.08.2023, given the input of the MATLAB code
    
    if L is None:
        L = int(512 * fs / 16000)

    # Hardcoded parameters
    nbsamples_per_sec = fs / (L / 2)
    lambda_val = np.exp(np.log(0.5) / (2 * nbsamples_per_sec))
    lambda_ext = np.exp(np.log(0.5) / (0.2 * nbsamples_per_sec))
    min_nb_samples = int(3 * nbsamples_per_sec)
    saveperiod = 20 * fs
    plotresults = 1
    shownode = 1
    
    # Initialization
    nbnodes = len(Y)
    lengthsignal = Y[0].shape[0]
    nbmicsnode = [Y[k].shape[1] for k in range(nbnodes)]
    dimnode = [nbmicsnode[k] + nbnodes - 1 for k in range(nbnodes)]
    Ryysamples = 0
    Rnnsamples = 0
    
    Ryy = [np.zeros((dimnode[k], dimnode[k], L // 2 + 1), dtype=complex) for k in range(nbnodes)]
    Rnn = [np.zeros((dimnode[k], dimnode[k], L // 2 + 1), dtype=complex) for k in range(nbnodes)]
    Wext = [np.ones((nbmicsnode[k], L // 2 + 1), dtype=complex) for k in range(nbnodes)]
    Wint = [np.zeros((dimnode[k], L // 2 + 1), dtype=complex) for k in range(nbnodes)]
    Wint[0][:, :] = 1  # Initializing Wint of first node
    Wext_target = [np.ones((nbmicsnode[k], L // 2 + 1), dtype=complex) for k in range(nbnodes)]
    estimation = [np.zeros(lengthsignal) for _ in range(nbnodes)]
    Rnninv = [None] * nbnodes
    
    saveperiod = int(saveperiod)
    updatetoken = 0
    Han = np.outer(hann(L), np.ones(max(nbmicsnode)))
    startupdating = 0
    nbupdates = 0
    Yest = np.zeros(L // 2 + 1, dtype=complex)
    Rnninv_initialized = 0
    count = 0
    count2 = 0
    
    if plotresults == 1:
        pass  # Plotting code can be added here
        
    output = estimation
    
    for iter in range(0, lengthsignal - L, L // 2):
        count += 1
        count2 += L // 2
        
        Yblock = [fft(np.sqrt(Han[:, :nbmicsnode[k]]) * Y[k][iter:iter + L, :]).T[:, :L // 2 + 1] for k in range(nbnodes)]
        Zblock = np.array([Wext[k].T @ Yblock[k] for k in range(nbnodes)])
        
        if onoff[iter] == 1:
            speech_active = True
            Ryysamples += 1
        else:
            speech_active = False
            Rnnsamples += 1
            if Rnninv_initialized == 0 and Rnnsamples > max(nbmicsnode) + nbnodes - 1:
                Rnninv = [np.linalg.inv(Rnn[k][:, :, u]) for k in range(nbnodes) for u in range(L // 2 + 1)]
                Rnninv_initialized = 1
        
        if startupdating == 0 and Ryysamples > max(nbmicsnode) + nbnodes - 1 and Rnnsamples > max(nbmicsnode) + nbnodes - 1:
            startupdating = 1
            Ryysamples = 0
            Rnnsamples = 0
        
        for k in range(nbnodes):
            Zk = np.delete(Zblock, k, axis=0)
            In = np.vstack((Yblock[k], Zk))
            for u in range(L // 2 + 1):
                if speech_active:
                    Ryy[k][:, :, u] = lambda_val * Ryy[k][:, :, u] + (1 - lambda_val) * (In[:, u] @ In[:, u].conj().T)
                else:
                    Rnn[k][:, :, u] = lambda_val * Rnn[k][:, :, u] + (1 - lambda_val) * (In[:, u] @ In[:, u].conj().T)
                    if Rnninv_initialized == 1:
                        Rnninv[k][:, :, u] = (1 / lambda_val) * Rnninv[k][:, :, u] - (
                                Rnninv[k][:, :, u] @ In[:, u]) @ Rnninv[k][:, :, u] @ In[:, u].conj().T / (
                                                        (lambda_val ** 2 / (1 - lambda_val)) +
                                                        lambda_val * In[:, u].conj().T @ Rnninv[k][:, :, u] @ In[:, u])
                
                if startupdating == 1:
                    Rxx = Ryy[k][:, :, u] - Rnn[k][:, :, u]
                    _, D, X = np.linalg.svd(Rxx)
                    Dmax = np.max(D)
                    Rxx = X[:, :1] @ Dmax * X[:, :1].conj().T
                    P = Rnninv[k][:, :, u] @ Rxx
                    Wint[k][:, u] = (1 / (mu + np.trace(P))) * P[:, 0]
                    Wext[k][:, u] = lambda_ext * Wext[k][:, u] + (1 - lambda_ext) * Wext_target[k][:, u]
        
        if Ryysamples >= min_nb_samples and Rnnsamples >= min_nb_samples:
            Ryysamples = 0
            Rnnsamples = 0
            nbupdates += 1
            
            if simultaneous == 0:
                Wext_target[updatetoken][:, :] = (1 - alpha) * Wext_target[updatetoken] + alpha * Wint[updatetoken][0:nbmicsnode[updatetoken], :]
                updatetoken = (updatetoken + 1) % nbnodes
            elif simultaneous == 1:
                for k in range(nbnodes):
                    Wext_target[k][:, :] = (1 - alpha) * Wext_target[k] + alpha * Wint[k][0:nbmicsnode[k], :]
        
        Yest = np.dot(Wint[shownode], In[shownode])
        blockest = np.real(ifft(np.concatenate((Yest, np.flipud(np.conj(Yest[1:L // 2]))))))
        
        for k in range(nbnodes):
            estimation[k][iter:iter + L] += np.sqrt(hann(L)) * blockest
        
        if count2 > saveperiod:
            count2 = 0
            print(str(100 * iter / lengthsignal) + '% processed')
            # Save results
            
    print(str(100 * iter / lengthsignal) + '% processed')
    # Save final results
    
    return output


if __name__ == '__main__':
    # Test code
    sys.exit(main())
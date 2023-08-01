import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from .online_danse import get_window, WOLAparameters

def plot_individual_run(filters: dict, scalings, sigma_sr, sigma_nr, nSamples):
    """Plot a visual comparison of the estimate filters with the FAS-SPF,
    for a single MC run, using the absolulte values."""
    
    # Compute FAS + SPF
    rtf = scalings / scalings[0]
    hs = np.sum(rtf ** 2) * sigma_sr[0] ** 2
    spf = hs / (hs + sigma_nr[0] ** 2)  # spectral post-filter
    fasAndSPF = rtf / (rtf.T @ rtf) * spf  # FAS BF + spectral post-filter

    # Plot comparison in absolute value, for each filter and the first sensor
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 2.5)
    for key in filters.keys():
        axs.plot(np.abs(filters[key][:, 0]), '.-', label=key)
    axs.plot(np.abs(fasAndSPF), '.-',label='FAS + SPF')
    axs.legend()
    axs.grid(which='both')
    axs.set_title(f'{nSamples} samples signals')
    axs.set_xlabel('Filter coefficients (absolute value)')
    fig.tight_layout()
    plt.show(block=False)

    return fig


def plot_final(durations, taus, toPlot: dict):

    # Check if we have data per node, or an average over nodes
    avgAcrossNodesFlag = len(toPlot[list(toPlot.keys())[0]].shape) == 3
    # if `len(toPlot[list(toPlot.keys())[0]].shape) == 4`, the data is available
    # per node and per MC run (i.e., we have a 4D array).

    nMC = toPlot[list(toPlot.keys())[0]].shape[0]
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    allLineStyles = ['-', '--', '-.', ':']
    allMarkers = ['s', 'o', 'x', 'd']
    for idxFilter, filterType in enumerate(toPlot.keys()):
        baseColor = f'C{idxFilter}'
        if idxFilter > len(allLineStyles):
            lineStyle = np.random.choice(allLineStyles)
        else:
            lineStyle = allLineStyles[idxFilter]

        nTaus = toPlot[filterType].shape[2]
        for idxTau in range(nTaus):
            if idxTau > len(allMarkers):
                marker = np.random.choice(allMarkers)
            else:
                marker = allMarkers[idxTau]
            
            if avgAcrossNodesFlag:  # Case where we have an average across nodes
                # Add a patch of color to show the range of values across MC runs
                axes.fill_between(
                    durations,
                    np.amin(toPlot[filterType][:, :, idxTau], axis=0),
                    np.amax(toPlot[filterType][:, :, idxTau], axis=0),
                    color=baseColor,
                    alpha=0.15
                )
                # Case where we have a single tau
                if 'danse' not in filterType and 'online' not in filterType:
                    axes.loglog(
                        durations,
                        np.mean(toPlot[filterType][:, :, idxTau], axis=0),
                        f'{baseColor}{marker}{lineStyle}',
                        label=filterType
                    )
                    break  # no need to loop over tau's
                else:  # Case where we have multiple tau's
                    tauLabel = f'tau{taus[idxTau]}'
                    # Replace dot (".") by "p"
                    tauLabel = tauLabel.replace('.', 'p')
                    axes.loglog(
                        durations,
                        np.mean(toPlot[filterType][:, :, idxTau], axis=0),
                        f'{baseColor}{marker}{lineStyle}',
                        label=f'{filterType}_{tauLabel}'
                    )
            else:  # Case where we have data per node and per MC run
                for k in range(toPlot[filterType].shape[-1]):
                    axes.loglog(
                        durations,
                        np.mean(toPlot[filterType][:, :, idxTau, k], axis=0),
                        f'{baseColor}{marker}{lineStyle}',
                        label=f'{filterType} $k=${k+1}',
                        alpha=(k + 1) / toPlot[filterType].shape[-1]
                    )

    plt.grid(which='both')
    axes.legend(loc='lower left')
    plt.xlabel('Signal duration (s)')
    plt.ylabel('Abs. diff. $\\Delta$ bw. filter and MF$\\cdot$SPF')
    axes.set_title(f'{nMC} MC runs')
    fig.tight_layout()
    plt.show(block=False)

    return fig


def plot_danse_wola_evol(
        K,
        channelToNodeMap,
        dur,
        filterDANSE,
        scalings,
        savefigs=False,
        figLabelRef='',
        wolaParams: WOLAparameters=WOLAparameters(),
        noiseOnlySigs=None,
        refCleanSigs=None,
        exportFolder=''
    ):
    """ Plot DANSE evolution for WOLA processing. """
    # --- Compute baseline filters (matched filter + spectral post-filter)
    # Get WOLA version of latent signal
    win = get_window(wolaParams.winType, wolaParams.nfft)
    kwargs = {
        'fs': wolaParams.fs,
        'window': win,
        'nperseg': wolaParams.nfft,
        'nfft': wolaParams.nfft,
        'noverlap': wolaParams.nfft - wolaParams.hop,
        'return_onesided': True,
        'axis': 0
    }
    refCleanSigs_stft = stft(refCleanSigs, **kwargs)[2]
    refCleanSigs_stft = refCleanSigs_stft.reshape(
        (refCleanSigs_stft.shape[0], -1, refCleanSigs.shape[1])
    )
    # Compute sigma_sr for all STFT bins
    sigma_sr = np.sqrt(np.abs(refCleanSigs_stft) ** 2)
    # Get WOLA version of noise-only signals
    noiseOnlySigs_stft = stft(noiseOnlySigs, **kwargs)[2]
    noiseOnlySigs_stft = noiseOnlySigs_stft.reshape(
        (noiseOnlySigs_stft.shape[0], -1, noiseOnlySigs.shape[1])
    )
    # Compute sigma_nr for all STFT bins
    sigma_nr = np.sqrt(np.abs(noiseOnlySigs_stft) ** 2)

    nSensors = filterDANSE.shape[1]
    nIter = filterDANSE.shape[2]
    for k in range(K):
        # Determine reference sensor index
        nodeChannels = np.where(channelToNodeMap == k)[0]
        idxRef = nodeChannels[0]  # first sensor of node k
        # Compute FAS + SPF
        rtf = scalings / scalings[idxRef]  # relative transfer functions at ref. sensor
        hs = np.sum(rtf ** 2) * np.mean(sigma_sr[:, :, idxRef] ** 2, axis=1)
        spf = hs / (hs + np.mean(sigma_nr[:, :, idxRef] ** 2, axis=1))  # spectral post-filter
        mf = rtf / (rtf.T.conj() @ rtf)  # matched filter

        toPlot_danse = np.zeros((nIter, nSensors))
        toPlot_fas = np.zeros((nIter, nSensors))
        for m in range(nSensors):
            toPlot_danse[:, m] = np.mean(np.abs(filterDANSE[:, m, :, k]), axis=0)
            toPlot_fas[:, m] = np.mean(np.abs(mf[m] * spf))

        # Plot
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8.5, 2.5)
        for m in range(nSensors):
            lab = f'$[\\mathbf{{w}}_k^i]_{m+1}$'
            if m in nodeChannels:
                lab += f' (local)'
            if m == idxRef:
                lab += ' (ref.)'
            ax.semilogy(
                toPlot_danse[:, m],
                f'C{m}.-',
                label=lab
            )
            if m == 0:
                ax.hlines(
                    toPlot_fas[0, m],
                    0,
                    nIter - 1,
                    color=f'C{m}',
                    linestyle='--',
                    label=f'$[\\mathbf{{w}}_{{\\mathrm{{MF}},k}}]_{m+1}\\cdot\\mathrm{{SPS}}_{m+1}$'
                )
            else:
                ax.hlines(
                    toPlot_fas[0, m],
                    0,
                    nIter - 1,
                    color=f'C{m}',
                    linestyle='--'
                )
        ax.set_title(f'Node $k=${k + 1}, channels {nodeChannels + 1}')
        ax.grid(which='both')
        ax.legend(loc='upper right')
        plt.xlabel('Iteration index $i$')
        fig.tight_layout()

        # # Compute delta between DANSE filter and FAS + SPF
        # delta = np.zeros((nIter, nSensors))
        # for m in range(nSensors):
        #     delta[:, m] = np.mean(
        #         np.abs(
        #             filterDANSE[:, m, :, k] - mf[m] * spf
        #         ) ** 2,
        #         axis=0
        #     )
        
        # # Plot
        # fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(8.5, 2.5)
        # ax.semilogy(delta, '.-')
        # ax.set_title(f'Node $k=${k + 1}, channels {nodeChannels + 1}')
        # ax.grid(which='both')
        # plt.xlabel('Iteration index $i$')
        # fig.tight_layout()

        # # Plot DANSE evolution
        # for m in range(nSensors):
        #     lab = f'$[\\mathbf{{w}}_k^i]_{m+1}$'
        #     if m in nodeChannels:
        #         lab += f' (local)'
        #     if m == idxRef:
        #         lab += ' (ref.)'
        #     ax.plot(
        #         np.abs(filterDANSE[m, :, k].T),
        #         f'C{m}.-',
        #         label=lab
        #     )
        #     if m == 0:
        #         ax.hlines(
        #             np.abs(fasAndSPF[m]),
        #             0,
        #             filterDANSE.shape[1] - 1,
        #             color=f'C{m}',
        #             linestyle='--',
        #             label=f'$[\\mathbf{{w}}_{{\\mathrm{{MF}},k}}]_{m+1}\\cdot\\mathrm{{SPS}}_{m+1}$'
        #         )
        #     else:
        #         ax.hlines(
        #             np.abs(fasAndSPF[m]),
        #             0,
        #             filterDANSE.shape[1] - 1,
        #             color=f'C{m}',
        #             linestyle='--'
        #         )

        # ax.set_title(f'Node $k=${k + 1}, channels {nodeChannels + 1}')
        # ax.legend(loc='upper right')
        # ax.grid(which='both')
        # plt.xlabel('Iteration index $i$')
        # fig.tight_layout()
        if savefigs:
            fig.savefig(f'{exportFolder}/danse_evol_n{k+1}_dur{int(dur * 1e3)}ms_{figLabelRef}.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)


def plot_filter(filter, scalings, sigma_sr, sigma_nr):
    fig, axs = plt.subplots(1, 1)
    for n in range(filter.shape[1]):
        # axs.plot(filter[:, n] * np.sum(scalings ** 2), '.-', label=f'MWF weights sensor {n}')
        axs.plot(filter[:, n], f'C{n}.-', label=f'MWF weights sensor {n}')
        # axs.plot(filter[:, n] / np.amax(np.abs(filter[:, n])), '.-', label=f'MWF weights sensor {n}')
        h = scalings / scalings[n]
        hs = np.sum(h ** 2) * sigma_sr[n] ** 2
        spectralPostFilter = hs / (hs + sigma_nr[n] ** 2)
        if n == 0:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--', label='Matched BF and spectral post-filter')
        else:
            axs.plot(h / np.sum(h ** 2) * spectralPostFilter, f'C{n}.--')
    # axs.plot(scalings / np.amax(np.abs(scalings)), '.--', label='Signal scalings')
    # axs.hlines(1, 0, filter.shape[0] - 1, color='k', linestyle='--', linewidth=0.5)
    fig.legend()
    plt.show(block=False)


def plot_results(cleanSigs, noisySignals, filteredSignals, filteredSignal_RTFs):
    # Plot
    nSensors = cleanSigs.shape[1]
    nRows = int(np.floor(np.sqrt(nSensors)))
    nCols = int(np.ceil(nSensors / nRows))
    fig, axs = plt.subplots(nRows, nCols, sharex=True, sharey=True)
    delta = 2 * np.amax(np.abs(noisySignals))
    for n in range(nSensors):
        if nRows == 1:
            currAx = axs[n % nCols]
        else:
            currAx = axs[int(np.floor(n / nCols)), n % nCols]
        # overlay 1
        currAx.plot(noisySignals[:, n], 'k', label=f'Noisy signal')
        currAx.plot(filteredSignals[:, n], 'r', label=f'MWF-filtering')
        currAx.plot(cleanSigs[:, n], 'y', label=f'Clean signal')
        # overlay 2
        currAx.plot(noisySignals[:, n] + delta, 'k', label=f'Noisy signal')
        currAx.plot(filteredSignal_RTFs[:, n] + delta, 'r', label=f'RTF-filtering')
        currAx.plot(cleanSigs[:, n] + delta, 'y', label=f'Clean signal')
        if n == 0:
            currAx.legend()
        currAx.set_title(f'Sensor {n}')
    plt.show(block=False)



def plot_danse_online_evol(
        K,
        channelToNodeMap,
        dur,
        filterDANSE,
        scalings,
        sigma_sr,
        sigma_nr,
        savefigs=False,
        figLabelRef='',
        exportFolder=''
    ):
    """ Plot DANSE evolution for non-WOLA online processing. """

    nSensors = filterDANSE.shape[0]
    for k in range(K):
        # Determine reference sensor index
        nodeChannels = np.where(channelToNodeMap == k)[0]
        idxRef = nodeChannels[0]  # first sensor of node k
        # Compute FAS + SPF
        rtf = scalings / scalings[idxRef]  # relative transfer functions at ref. sensor
        hs = np.sum(rtf ** 2) * sigma_sr[idxRef]
        spf = hs / (hs + sigma_nr[idxRef])  # spectral post-filter
        fasAndSPF = rtf / (rtf.T.conj() @ rtf) * spf  # FAS BF + spectral post-filter

        # Plot
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8.5, 2.5)
        # Plot DANSE evolution
        for m in range(nSensors):
            lab = f'$[\\mathbf{{w}}_k^i]_{m+1}$'
            if m in nodeChannels:
                lab += f' (local)'
            if m == idxRef:
                lab += ' (ref.)'
            ax.plot(
                np.abs(filterDANSE[m, :, k].T),
                f'C{m}.-',
                label=lab
            )
            if m == 0:
                ax.hlines(
                    np.abs(fasAndSPF[m]),
                    0,
                    filterDANSE.shape[1] - 1,
                    color=f'C{m}',
                    linestyle='--',
                    label=f'$[\\mathbf{{w}}_{{\\mathrm{{MF}},k}}]_{m+1}\\cdot\\mathrm{{SPS}}_{m+1}$'
                )
            else:
                ax.hlines(
                    np.abs(fasAndSPF[m]),
                    0,
                    filterDANSE.shape[1] - 1,
                    color=f'C{m}',
                    linestyle='--'
                )

        ax.set_title(f'Node $k=${k + 1}, channels {nodeChannels + 1}')
        ax.legend(loc='upper right')
        ax.grid(which='both')
        plt.xlabel('Iteration index $i$')
        fig.tight_layout()
        if savefigs:
            fig.savefig(f'{exportFolder}/danse_evol_n{k+1}_dur{int(dur * 1e3)}ms_{figLabelRef}.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)



def plot_online_mwf_evol(
        dur,
        filterMWF,
        scalings,
        sigma_sr,
        sigma_nr,
        savefigs=False,
        figLabelRef='',
        exportFolder='',
        beta=None
    ):
    """ Plot MWF evolution for non-WOLA online processing. """

    nSensors = filterMWF.shape[-1]
    for m in range(nSensors):
        # Compute FAS + SPF
        rtf = scalings / scalings[m]  # relative transfer functions at ref. sensor
        hs = np.sum(rtf ** 2) * sigma_sr[m]
        spf = hs / (hs + sigma_nr[m])  # spectral post-filter
        fasAndSPF = rtf / (rtf.T.conj() @ rtf) * spf  # FAS BF + spectral post-filter

        # Plot
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8.5, 2.5)
        # Plot DANSE evolution
        for m2 in range(filterMWF.shape[0]):  # loop over filter taps
            lab = f'$[\\hat{{\\mathbf{{w}}}}_{m+1}^i]_{m2+1}$'
            ax.semilogy(
                np.abs(filterMWF[m2, :, m].T),
                f'C{m2}.-',
                label=lab
            )
            if m2 == 0:
                ax.hlines(
                    np.abs(fasAndSPF[m2]),
                    0,
                    filterMWF.shape[1] - 1,
                    color=f'C{m2}',
                    linestyle='--',
                    label=f'$[\\mathbf{{w}}_{{\\mathrm{{MF}},k}}]_{m+1}\\cdot\\mathrm{{SPS}}_{m+1}$'
                )
            else:
                ax.hlines(
                    np.abs(fasAndSPF[m2]),
                    0,
                    filterMWF.shape[1] - 1,
                    color=f'C{m2}',
                    linestyle='--'
                )

        ti = f'Sensor $k=${m + 1}'
        if beta is not None:
            ti += f', $\\beta={np.round(beta, 3)}$'
        ax.set_title(ti)
        ax.legend(loc='upper right')
        ax.grid(which='both')
        plt.xlabel('Iteration index $i$')
        fig.tight_layout()
        if savefigs:
            fig.savefig(f'{exportFolder}/mwf_evol_n{m+1}_dur{int(dur * 1e3)}ms_{figLabelRef}.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
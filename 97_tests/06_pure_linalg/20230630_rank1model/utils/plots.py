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


def plot_final(
        durations,
        taus,
        toPlot: dict,
        fs=16e3,
        L=1024,
        R=512,
        avgAcrossNodesFlag=False,
        figTitleSuffix=None,
        vad=None
    ):
    """Plot the final results of the MC runs, for each filter type."""

    nMC = toPlot[list(toPlot.keys())[0]].shape[0]
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 4)
    allLineStyles = ['-', '--', '-.', ':']
    for idxFilter, filterType in enumerate(toPlot.keys()):
        baseColor = f'C{idxFilter}'
        if 'online' in filterType or 'wola' in filterType:
            nTaus = toPlot[filterType].shape[2]
            # Plot as function of beta (== as function of tau)
            if 'online' in filterType:
                xAxis = np.arange(0, toPlot[filterType].shape[1]) * L / fs
            elif 'wola' in filterType:
                xAxis = np.arange(0, toPlot[filterType].shape[1]) * R / fs
            for idxTau in range(nTaus):
                ls = allLineStyles[idxTau % len(allLineStyles)]
                tauLeg = f'($\\tau={taus[idxTau]}$ s)'
                if avgAcrossNodesFlag:
                    axes.fill_between(
                        xAxis,
                        np.amin(toPlot[filterType][:, :, idxTau], axis=0),
                        np.amax(toPlot[filterType][:, :, idxTau], axis=0),
                        color=baseColor,
                        alpha=0.15
                    )
                    axes.semilogy(
                        xAxis,
                        np.mean(toPlot[filterType][:, :, idxTau], axis=0),
                        f'{baseColor}{ls}',
                        label=f'{filterType} {tauLeg}',
                    )
                else:
                    for k in range(toPlot[filterType].shape[-1]):
                        axes.semilogy(
                            xAxis,
                            np.mean(toPlot[filterType][:, :, idxTau, k], axis=0),
                            f'{baseColor}{ls}',
                            label=f'{filterType} $k=${k+1} {tauLeg}',
                            alpha=(k + 1) / toPlot[filterType].shape[-1]
                        )
        else:  # <-- batch-mode
            # Plot as function of signal duration
            if avgAcrossNodesFlag:  # Case where we have an average across nodes
                # Add a patch of color to show the range of values across MC runs
                axes.fill_between(
                    durations,
                    np.amin(toPlot[filterType], axis=0),
                    np.amax(toPlot[filterType], axis=0),
                    color=baseColor,
                    alpha=0.15
                )
                axes.semilogy(
                    durations,
                    np.mean(toPlot[filterType], axis=0),
                    f'{baseColor}o-',
                    label=filterType
                )
            else:  # Case where we have data per node and per MC run
                for k in range(toPlot[filterType].shape[-1]):
                    axes.semilogy(
                        durations,
                        np.mean(toPlot[filterType][:, :, k], axis=0),
                        f'{baseColor}o-',
                        label=f'{filterType} $k=${k+1}',
                        alpha=(k + 1) / toPlot[filterType].shape[-1]
                    )
    
    # Add VAD if provided
    if vad is not None:
        axesRight = axes.twinx()
        axesRight.plot(
            np.linspace(0, np.amax(durations), len(vad)),
            vad,
            'k--',
            linewidth=0.5,
            zorder=-1
        )
        axesRight.set_ylabel('VAD (-)')
        axesRight.set_yticks([0, 1])
        axesRight.set_zorder(-1)  # put the VAD behind the other plots
        axes.set_frame_on(False)  # remove the frame of the main axes
        # Set `axes` grid only to horizontal lines
        axes.grid(axis='y', which='both')
        # Add grid to `axesRight` only to vertical lines
        axesRight.set_xticks(axes.get_xticks())
        axesRight.grid(axis='x', which='major')
        # Ensure the grid is behind the other plots
        axesRight.set_axisbelow(True)
    else:
        axes.grid(which='both')

    axes.set_xlabel('Signal duration (s)', loc='left')
    axes.legend(loc='upper right', fontsize='small')
    if nMC == 1:
        ti = '1 MC run'
    else:
        ti = f'{nMC} MC runs'
    if figTitleSuffix is not None:
        ti += f' {figTitleSuffix}'
    axes.set_title(ti)
    axes.set_ylabel('$\\Delta$ bw. estimated filter and baseline')
    flagBatchModeIncluded = any(['batch' in t for t in toPlot.keys()])
    if flagBatchModeIncluded:
        axes.set_xlim([np.amin(durations), np.amax(durations)])
    else:
        axes.set_xlim([0, np.amax(xAxis)])
    
    if 'online' in filterType or 'wola' in filterType:
        # Add secondary x-axis with iterations
        ax2 = axes.secondary_xaxis("top")
        ax2.set_xticks(axes.get_xticks())
        if 'online' in filterType:
            xTicks2 = np.round(axes.get_xticks() * fs / L).astype(int)
        elif 'wola' in filterType:
            xTicks2 = np.round(axes.get_xticks() * fs / R).astype(int)
        ax2.set_xticklabels(xTicks2)
        ax2.set_xlabel('Iteration index (-)', loc='left')
    fig.tight_layout()
    # Adapt y-axis limits to the data
    ymin, ymax = np.inf, -np.inf
    for idxFilter, filterType in enumerate(toPlot.keys()):
        if ('online' in filterType or 'wola' in filterType) and\
            flagBatchModeIncluded:
            idxStart = int(np.amin(durations) * fs // L)
            ymin = min(
                ymin,
                np.amin(toPlot[filterType][:, idxStart:, :])
            )
            ymax = max(
                ymax,
                np.amax(toPlot[filterType][:, idxStart:, :])
            )
        else:
            ymin = min(ymin, np.amin(toPlot[filterType]))
            ymax = max(ymax, np.amax(toPlot[filterType]))
    ymin *= 0.9
    ymax *= 1.1
    axes.set_ylim([ymin, ymax])
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
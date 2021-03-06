def simpler_danse_simultaneous(yin, asc: classes.AcousticScenario, settings: classes.ProgramSettings, oVAD, timeInstants, masterClockNodeIdx, fs):
    """Wrapper for Simultaneous-Node-Updating DANSE (rs-DANSE) [2].

    Parameters
    ----------
    yin : [Nt x Ns] np.ndarray of floats
        The microphone signals in the time domain.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    settings : ProgramSettings object
        The settings for the current run.
    oVAD : [Nt x 1] np.ndarray of booleans or binary ints
        Voice Activity Detector.
    timeInstants : [Nt x Nn] np.ndarray of floats
        Time instants corresponding to the samples of each of the Nn nodes in the network. 
    masterClockNodeIdx : int
        Index of node to be used as "master clock" (0 ppm SRO). 
    fs : list of floats
        Sampling frequency of each node.

    Returns
    -------
    d : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using full-observations vectors (also data coming from neighbors).
    dLocal : [Nt x Nn] np.ndarray of floats
        Time-domain representation of the desired signal at each of the Nn nodes -- using only local observations (not data coming from neighbors).
        -Note: if `settings.computeLocalEstimate == False`, then `dLocal` is output as an all-zeros array.
    """

    # Hard-coded parameters
    alphaExternalFilters = 0.7      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html
    # alphaExternalFilters = 0.5      # external filters (broadcasting) exp. avg. update constant -- cfr. WOLA-DANSE implementation in https://homes.esat.kuleuven.be/~abertran/software.html

    # Initialization (extracting/defining useful quantities)
    _, winWOLAanalysis, winWOLAsynthesis, frameSize, Ns, numIterations, _, neighbourNodes = subs.danse_init(yin, settings, asc)

    # Loop over time instants -- based on a particular reference node
    masterClock = timeInstants[:, masterClockNodeIdx]     # reference clock

    # ??????????????????????????????????????????????????????????????? Arrays initialization ???????????????????????????????????????????????????????????????
    lk = np.zeros(asc.numNodes, dtype=int)          # node-specific broadcast index
    i = np.zeros(asc.numNodes, dtype=int)           # !node-specific! DANSE iteration index
    l = np.zeros(asc.numNodes, dtype=int)           # node-specific frame index
    #
    wTilde = []                                     # filter coefficients - using full-observations vectors (also data coming from neighbors)
    wTildeExternal = []                             # external filter coefficients - used for broadcasting only, updated every `settings.timeBtwExternalFiltUpdates` seconds
    wTildeExternalTarget = []                       # target external filter coefficients -- values towards which `wTildeExternal` slowly converge between each external filter update
    Rnntilde = []                                   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
    Ryytilde = []                                   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
    ryd = []                                        # cross-correlation between observations and estimations
    ytilde = []                                     # local full observation vectors, time-domain
    ytildeHat = []                                  # local full observation vectors, frequency-domain
    z = []                                          # current-iteration compressed signals used in DANSE update
    zBuffer = []                                    # current-iteration "incoming signals from other nodes" buffer
    bufferFlags = []                                # buffer flags (0, -1, or +1) - for when buffers over- or under-flow
    bufferLengths = []                              # node-specific number of samples in each buffer
    phaseShiftFactors = []                          # phase-shift factors for SRO compensation (only used if `settings.compensateSROs == True`)
    SROsEstimates = []                              # SRO estimates per node (for each neighbor)
    residualSROs = []                               # residual SROs, for each node, across DANSE iterations
    dimYTilde = np.zeros(asc.numNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
    oVADframes = np.zeros(numIterations)            # oracle VAD per time frame
    numFreqLines = int(frameSize / 2 + 1)           # number of frequency lines (only positive frequencies)
    if settings.computeLocalEstimate:
        wLocal = []                                     # filter coefficients - using only local observations (not data coming from neighbors)
        Rnnlocal = []                                   # autocorrelation matrix when VAD=0 - using only local observations (not data coming from neighbors)
        Ryylocal = []                                   # autocorrelation matrix when VAD=1 - using only local observations (not data coming from neighbors)
        dimYLocal = np.zeros(asc.numNodes, dtype=int)   # dimension of y_k (== M_k)
        
    for k in range(asc.numNodes):
        dimYTilde[k] = sum(asc.sensorToNodeTags == k + 1) + len(neighbourNodes[k])

        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # dimYTilde[k] = asc.numSensors
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 

        wtmp = np.zeros((numFreqLines, numIterations + 1, dimYTilde[k]), dtype=complex)
        wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
        wTilde.append(wtmp)
        wtmp = np.zeros((numFreqLines, dimYTilde[k]), dtype=complex)
        wtmp[:, 0] = 1      # initialize filter as a selector of the unaltered first sensor signal
        wTildeExternal.append(wtmp)
        wtmp = np.zeros((numFreqLines, dimYTilde[k]), dtype=complex)
        wtmp[:, 0] = 1      # initialize filter as a selector of the unaltered first sensor signal
        wTildeExternalTarget.append(wtmp)
        #
        ytilde.append(np.zeros((frameSize, numIterations, dimYTilde[k]), dtype=complex))
        ytildeHat.append(np.zeros((numFreqLines, numIterations, dimYTilde[k]), dtype=complex))
        #
        sliceTilde = np.finfo(float).eps * np.eye(dimYTilde[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
        Rnntilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # noise only
        Ryytilde.append(np.tile(sliceTilde, (numFreqLines, 1, 1)))                    # speech + noise
        ryd.append(np.zeros((numFreqLines, dimYTilde[k]), dtype=complex))   # noisy-vs-desired signals covariance vectors
        #
        bufferFlags.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))    # init all buffer flags at 0 (assuming no over- or under-flow)
        bufferLengths.append(np.zeros((len(masterClock), len(neighbourNodes[k]))))
        #
        if settings.broadcastDomain == 't':
            z.append(np.empty((frameSize, 0), dtype=float))
        elif settings.broadcastDomain == 'f':
            z.append(np.empty((numFreqLines, 0), dtype=complex))
        zBuffer.append([np.array([]) for _ in range(len(neighbourNodes[k]))])
        #
        phaseShiftFactors.append(np.zeros(dimYTilde[k]))   # initiate phase shift factors as 0's (no phase shift)
        SROsEstimates.append(np.zeros(len(neighbourNodes[k])))
        residualSROs.append(np.zeros((dimYTilde[k], numIterations)))
        #
        if settings.computeLocalEstimate:
            dimYLocal[k] = sum(asc.sensorToNodeTags == k + 1)
            wtmp = np.zeros((numFreqLines, numIterations + 1, dimYLocal[k]), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wLocal.append(wtmp)
            sliceLocal = np.finfo(float).eps * np.eye(dimYLocal[k], dtype=complex)   # single autocorrelation matrix init (identities -- ensures positive-definiteness)
            Rnnlocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # noise only
            Ryylocal.append(np.tile(sliceLocal, (numFreqLines, 1, 1)))                    # speech + noise
    # Desired signal estimate [frames x frequencies x nodes]
    dhat = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)        # using full-observations vectors (also data coming from neighbors)
    d = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhat`
    dhatLocal = np.zeros((numFreqLines, numIterations, asc.numNodes), dtype=complex)   # using only local observations (not data coming from neighbors)
    dLocal = np.zeros((len(masterClock), asc.numNodes))  # time-domain version of `dhatLocal`

    # Autocorrelation matrices update counters
    numUpdatesRyy = np.zeros(asc.numNodes)
    numUpdatesRnn = np.zeros(asc.numNodes)
    minNumAutocorrUpdates = np.amax(dimYTilde)  # minimum number of Ryy and Rnn updates before starting updating filter coefficients
    # Booleans
    startUpdates = np.full(shape=(asc.numNodes,), fill_value=False)         # when True, perform DANSE updates every `nExpectedNewSamplesPerFrame` samples
    # ??????????????????????????????????????????????????????????????? Arrays initialization ???????????????????????????????????????????????????????????????

    # Prepare events to be considered in main `for`-loop
    eventsMatrix = subs.get_events_matrix(timeInstants,
                                            frameSize,
                                            Ns,
                                            settings.broadcastLength,
                                            asc.nodeLinks,
                                            fs=fs
                                            )
    
    zFull = []
    for k in range(asc.numNodes):
        zFull.append(np.empty((0,)))

    # External filter updates (for broadcasting)
    lastExternalFiltUpdateInstant = np.zeros(asc.numNodes)   # [s]

    t0 = time.perf_counter()        # loop timing
    # Loop over time instants when one or more event(s) occur(s)
    for idxGlobal, events in enumerate(eventsMatrix):

        # Parse event matrix (+ inform user)
        t, eventTypes, nodesConcerned = subs.events_parser(events, startUpdates, printouts=settings.printouts.events_parser)
        # Loop over events
        for idxEvent in range(len(eventTypes)):

            # Node index
            k = int(nodesConcerned[idxEvent])
            event = eventTypes[idxEvent]

            if event == 'broadcast':        # <-- data compression + broadcast to neighbour nodes
                
                # Extract current local data chunk
                idxEndChunk = int(np.floor(t * fs[k]))
                if settings.broadcastDomain == 't':
                    idxBegChunk = np.amax([idxEndChunk - 2 * frameSize, 0])   # don't go into negative sample indices!
                    yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]
                    # Pad zeros at beginning if needed
                    if idxEndChunk - idxBegChunk < 2 * frameSize:   # Using 2*N to keep power of 2 for FFT
                        yLocalCurr = np.concatenate((np.zeros((2 * frameSize - yLocalCurr.shape[0], yLocalCurr.shape[1])), yLocalCurr))
                elif settings.broadcastDomain == 'f':
                    idxBegChunk = np.amax([idxEndChunk - frameSize, 0])   # don't go into negative sample indices!
                    yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]
                    # Pad zeros at beginning if needed
                    if idxEndChunk - idxBegChunk < frameSize:
                        yLocalCurr = np.concatenate((np.zeros((frameSize - yLocalCurr.shape[0], yLocalCurr.shape[1])), yLocalCurr))
                    
                # Perform broadcast -- update all buffers in network
                zBuffer = subs.broadcast(
                            t,
                            k,
                            fs[k],
                            settings.broadcastLength,
                            yLocalCurr,
                            wTildeExternal[k],
                            # wTilde[k][:, i[k], :],
                            frameSize,
                            neighbourNodes,
                            lk,
                            zBuffer,
                            settings.broadcastDomain,
                        )
                    
            elif event == 'update':         # <-- DANSE filter update

                skipUpdate = False  # flag to skip update if needed

                # Extract current local data chunk
                idxEndChunk = int(np.floor(t * fs[k]))
                idxBegChunk = idxEndChunk - frameSize     # `N` samples long chunk
                yLocalCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == k+1]

                # Compute VAD
                VADinFrame = oVAD[idxBegChunk:idxEndChunk]
                oVADframes[i[k]] = sum(VADinFrame == 0) <= frameSize / 2   # if there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1
                if oVADframes[i[k]]:    # Count number of spatial covariance matrices updates
                    numUpdatesRyy[k] += 1
                else:
                    numUpdatesRnn[k] += 1
                
                # ??????????????????????????????????????????????????????????????? Build local observations vector ???????????????????????????????????????????????????????????????
                # Process buffers
                if settings.broadcastDomain == 't':
                    z[k], bufferFlags = subs.process_incoming_signals_buffers(
                                        zBuffer[k],
                                        z[k],
                                        neighbourNodes[k],
                                        i[k],
                                        frameSize,
                                        Ns=Ns,
                                        L=settings.broadcastLength,
                                        lastExpectedIter=numIterations - 1,
                                        broadcastDomain=settings.broadcastDomain)

                    zFull[k] = np.concatenate((zFull[k], z[k][Ns:, 0]))     # selecting the first neighbor (index 0)

                
                # Wipe local buffers
                zBuffer[k] = [np.array([]) for _ in range(len(neighbourNodes[k]))]


                if settings.broadcastDomain == 't':
                    # Build full available observation vector
                    yTildeCurr = np.concatenate((yLocalCurr, z[k]), axis=1)
                    ytilde[k][:, i[k], :] = yTildeCurr
                    # Go to frequency domain
                    ytildeHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(ytilde[k][:, i[k], :] * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                    ytildeHat[k][:, i[k], :] = ytildeHatCurr[:numFreqLines, :]      # Keep only positive frequencies

                elif settings.broadcastDomain == 'f':
                    # Broadcasting done in frequency-domain
                    yLocalHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yLocalCurr * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                    # Build z from scratch -- ignore broadcasted data -- [19/04/2022 decision]
                    z = np.empty((numFreqLines, 0), dtype=complex)
                    for q in neighbourNodes[k]:
                        yq = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags == q+1]
                        yqHat = 1 / winWOLAanalysis.sum() * np.fft.fft(yq * winWOLAanalysis[:, np.newaxis], frameSize, axis=0)
                        yqHat = yqHat[:numFreqLines, :]
                        zq = np.einsum('ij,ij->i', wTildeExternal[q][:, :asc.numSensorPerNode[q]].conj(), yqHat)     # zq = wqq^H * yq
                        # zq = np.einsum('ij,ij->i', wTildeExternal[q][:, :asc.numSensorPerNode[q]], yqHat)     # zq = wqq^T * yq
                        z = np.concatenate((z, zq[:, np.newaxis]), axis=1)
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                        # z = np.concatenate((z, yqHat), axis=1)
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                        # # TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP TMP 
                    ytildeHat[k][:, i[k], :] = np.concatenate((yLocalHatCurr[:numFreqLines, :], z), axis=1)
                # ???????????????????????????????????????????????????????????? Build local observations vector ???????????????????????????????????????????????????????????????

                # ??????????????????????????????????????????????????????????????? Spatial covariance matrices updates ???????????????????????????????????????????????????????????????
                Ryytilde[k], Rnntilde[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :],
                                                Ryytilde[k], Rnntilde[k], settings.expAvgBeta, oVADframes[i[k]])
                if settings.computeLocalEstimate:
                    # Local observations only
                    Ryylocal[k], Rnnlocal[k] = subs.spatial_covariance_matrix_update(ytildeHat[k][:, i[k], :dimYLocal[k]],
                                                    Ryylocal[k], Rnnlocal[k], settings.expAvgBeta, oVADframes[i[k]])
                # ??????????????????????????????????????????????????????????????? Spatial covariance matrices updates ???????????????????????????????????????????????????????????????
                
                # Check quality of autocorrelations estimates -- once we start updating, do not check anymore
                if not startUpdates[k] and numUpdatesRyy[k] >= minNumAutocorrUpdates and numUpdatesRnn[k] >= minNumAutocorrUpdates:
                    startUpdates[k] = True

                if startUpdates[k] and not settings.bypassFilterUpdates and not skipUpdate:
                    # No `for`-loop versions
                    if settings.performGEVD:    # GEVD update
                        wTilde[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryytilde[k], Rnntilde[k], settings.GEVDrank, settings.referenceSensor)
                        if settings.computeLocalEstimate:
                            wLocal[k][:, i[k] + 1, :], _ = subs.perform_gevd_noforloop(Ryylocal[k], Rnnlocal[k], settings.GEVDrank, settings.referenceSensor)

                    else:                       # regular update (no GEVD)
                        wTilde[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryytilde[k], Rnntilde[k], settings.referenceSensor)
                        if settings.computeLocalEstimate:
                            wLocal[k][:, i[k] + 1, :] = subs.perform_update_noforloop(Ryylocal[k], Rnnlocal[k], settings.referenceSensor)
                else:
                    # Do not update the filter coefficients
                    wTilde[k][:, i[k] + 1, :] = wTilde[k][:, i[k], :]
                    if settings.computeLocalEstimate:
                        wLocal[k][:, i[k] + 1, :] = wLocal[k][:, i[k], :]
                    if skipUpdate:
                        print(f'Node {k+1}: {i[k] + 1}^th update skipped.')
                if settings.bypassFilterUpdates:
                    print('!! User-forced bypass of filter coefficients updates !!')



                # ???????????????????????????????????????????????????????????????  Update external filters (for broadcasting)  ???????????????????????????????????????????????????????????????

                # Systematically update the external filter coefficients, slowly towards `wTildeExternalTarget`
                # betaExternal = settings.expAvgBeta ** 10    # corresponding to a 10 time faster time constant
                # betaExternal = 0    # no smoothing
                betaExternal = settings.expAvgBeta ** 2
                wTildeExternal[k] = betaExternal * wTildeExternal[k] + (1 - betaExternal) * wTildeExternalTarget[k]

                # Update targets
                if t - lastExternalFiltUpdateInstant[k] >= settings.timeBtwExternalFiltUpdates:
                    # Inverted definition w.r.t. the `beta` of exponential averaging -- see szurley2013a, eq.(19)
                    wTildeExternalTarget[k] = (1 - alphaExternalFilters) * wTildeExternalTarget[k] + alphaExternalFilters * wTilde[k][:, i[k] + 1, :]
                    # Update last external filter update instant [s]
                    lastExternalFiltUpdateInstant[k] = t
                    if settings.printouts.externalFilterUpdates:    # inform user
                        print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {settings.timeBtwExternalFiltUpdates}s)')
                
                # ???????????????????????????????????????????????????????????????  Update external filters (for broadcasting)  ???????????????????????????????????????????????????????????????


                # ----- Compute desired signal chunk estimate -----
                dhatCurr = np.einsum('ij,ij->i', wTilde[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                dhat[:, i[k], k] = dhatCurr
                # Transform back to time domain (WOLA processing)
                dChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatCurr, frameSize)
                d[idxBegChunk:idxEndChunk, k] += np.real_if_close(dChunk)   # overlap and add construction of output time-domain signal
                
                if settings.computeLocalEstimate:
                    # Local observations only
                    dhatLocalCurr = np.einsum('ij,ij->i', wLocal[k][:, i[k] + 1, :].conj(), ytildeHat[k][:, i[k], :dimYLocal[k]])   # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850
                    dhatLocal[:, i[k], k] = dhatLocalCurr
                    # Transform back to time domain (WOLA processing)
                    dLocalChunk = winWOLAsynthesis.sum() * winWOLAsynthesis * subs.back_to_time_domain(dhatLocalCurr, frameSize)
                    dLocal[idxBegChunk:idxEndChunk, k] += np.real_if_close(dLocalChunk)   # overlap and add construction of output time-domain signal
                # -----------------------------------------------------------------------
                
                # Increment DANSE iteration index
                i[k] += 1
                
                
                
                # if k == 0:
                #     # Dynamic plotting (updating at every DANSE update)
                #     # of yk, yq, zq, and dk
                #     if i[k] == 1:
                #         fig = plt.figure(figsize=(8,4))
                #         ax1 = fig.add_subplot(211)
                #         ax2 = fig.add_subplot(212)
                #     #
                #     if settings.broadcastDomain == 't':
                #         ax1.plot(masterClock[:len(zFull[k])], zFull[k])
                #         ax1.set_ylim([-1,1])  
                #         ax1.set_title('z')
                #     elif settings.broadcastDomain == 'f':
                #         ax1.plot(20 * np.log10(np.abs(yLocalHatCurr[:numFreqLines, :])), label='Local sensor data yk')
                #         ax1.plot(20 * np.log10(np.abs(dhatCurr)), label='Enhanced signal dk')
                #         ax1.set_ylim([-100,0])  
                #         ax1.set_title('Local')
                #         ax1.legend()
                #     ax1.grid()
                #     #
                #     if settings.broadcastDomain == 't':
                #         ax2.plot(masterClock[:len(zFull[k])], yin[Ns:idxEndChunk, asc.sensorToNodeTags == k+1])
                #         ax2.set_ylim([-1,1])
                #         ax2.set_title('y')
                #     elif settings.broadcastDomain == 'f':
                #         yNeighborCurr = yin[idxBegChunk:idxEndChunk, asc.sensorToNodeTags != k+1]
                #         yNeighborHatCurr = 1 / winWOLAanalysis.sum() * np.fft.fft(yNeighborCurr, frameSize, axis=0)
                #         ax2.plot(20 * np.log10(np.abs(yNeighborHatCurr[:numFreqLines, :])), label='Neighbor sensor data yq')
                #         ax2.plot(20 * np.log10(np.abs(z)), label='Compressed neighbor data: zq')
                #         ax2.set_ylim([-100,0])
                #         ax2.set_title('Neighbor')
                #         ax2.legend()
                #     ax2.grid()
                #     # draw the plot
                #     plt.draw() 
                #     plt.pause(0.01)
                #     ax1.clear()
                #     ax2.clear()

    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.round(masterClock[-1], 2)}s of signal processed in {str(datetime.timedelta(seconds=dur))}s.')
    print(f'(Real-time processing factor: {np.round(masterClock[-1] / dur, 4)})')

    # Export empty array if local desired signal estimate was not computed
    if (dLocal == 0).all():
        dLocal = np.array([])

    stop = 1


    return d, dLocal
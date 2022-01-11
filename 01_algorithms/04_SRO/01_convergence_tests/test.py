import numpy as np
import matplotlib.pyplot as plt

initialPhaseShift = 0           # $\Delta\varphi_0$
nFrames = 2**5              	# number of time frames
trueIncrementalPhaseShift = 1024/2 * 30e-6 / (1 + 30e-6)   # N/2 * \eps^\star / (1 + \eps^\star)

frameNumbers = np.arange(1,nFrames+1)
truePhaseShifts = frameNumbers * trueIncrementalPhaseShift  # k * N/2 * \eps^\star / (1 + \eps^\star)

forgettingFactor = 1

# Choose update method
updateMethod = 'sum'
# updateMethod = 'nosum'   

phaseShiftEstimates = np.zeros(nFrames)
residualPhaseShifts = np.zeros(nFrames)
for k in np.arange(nFrames):

    if k > 0:
        residualPhaseShifts[k] = truePhaseShifts[k] - phaseShiftEstimates[k - 1]
    else:
        residualPhaseShifts[k] = truePhaseShifts[k] - 0

    # if k > 0:
    #     error = 0.01 * np.nanmean(phaseShiftEstimates[phaseShiftEstimates != 0]) * np.random.uniform(-1, 1)
    # else:
        error = 0

    if updateMethod == 'sum':
        phaseShiftEstimates[k] = error + forgettingFactor * phaseShiftEstimates[k - 1] + np.sum(residualPhaseShifts[:k+1])
    elif updateMethod == 'nosum':
        phaseShiftEstimates[k] = error + forgettingFactor * phaseShiftEstimates[k - 1] + residualPhaseShifts[k]

    # if k > 0:
    #     print('Frame k=%i:\nTrue POD (k): %.1f; Estimate (k-1): %.1f; Residual (k): %.1f' \
    #         % (k, truePhaseShifts[k], phaseShiftEstimates[k - 1], residualPhaseShifts[k]))

phaseShiftEstimates = np.insert(phaseShiftEstimates,0,initialPhaseShift)
phaseShiftEstimates = phaseShiftEstimates[:-1]

fig = plt.figure(figsize=(8,5), constrained_layout=True)
ax = fig.add_subplot(211)
ax.plot(frameNumbers, truePhaseShifts ,'.-')
ax.plot(frameNumbers, phaseShiftEstimates,'.-')
ax.grid()
plt.legend(['True phase shift $\\varphi_k^\\ast$', 'Estimated phase shift $\\hat{\\varphi}_k$'])
ax.set(xlabel='Frame index $k$')
plt.xlim((0, nFrames))
#
ax = fig.add_subplot(212)
ax.plot(frameNumbers, truePhaseShifts / frameNumbers,'.-')
ax.plot(frameNumbers, phaseShiftEstimates / frameNumbers,'.-')
ax.grid()
plt.legend(['True incr. phase shift $\\varphi_k^\\ast / k$', 'Estimated incr. phase shift $\\hat{\\varphi}_k / k$'])
ax.set(xlabel='Frame index $k$')
plt.xlim((0, nFrames))
#
plt.suptitle('Forgetting factor $\lambda$ = %.2f' % forgettingFactor)
plt.show()

stop =1


    
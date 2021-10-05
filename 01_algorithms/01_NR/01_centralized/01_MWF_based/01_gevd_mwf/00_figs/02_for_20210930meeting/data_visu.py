import numpy as np
import os
import matplotlib.pyplot as plt

# GEVD comparison
folders = ['01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\twosources_GEVDrank1',
 '01_algorithms\\01_NR\\01_centralized\\01_MWF_based\\01_GEVD_MWF\\00_figs\\02_for_20210930meeting\\twosources_GEVDrank2']

nSensors = 5

SNRimps = np.zeros((nSensors,3,len(folders)))
for ii, folder in enumerate(folders):
    onlycsv = [f for f in os.listdir(folder) if os.path.splitext(f)[1] == '.csv']
    for jj, file in enumerate(onlycsv):
        my_data = np.genfromtxt(folder + '\\' + file, delimiter=',')
        my_data = my_data[1:,1:]
        SNRimps[:,jj,ii] = my_data[-1,:]

# PLOT
wd = 0.25
fig, ax = plt.subplots()
for ii in range(SNRimps.shape[1]):
    ph1 = ax.bar(ii-wd/2, np.mean(np.squeeze(SNRimps[:,ii,0])), width=wd, yerr=np.std(np.squeeze(SNRimps[:,ii,0])), color='red', alpha=0.5, ecolor='black', capsize=10)
    ph2 = ax.bar(ii+wd/2, np.mean(np.squeeze(SNRimps[:,ii,1])), width=wd, yerr=np.std(np.squeeze(SNRimps[:,ii,1])), color='blue', alpha=0.5, ecolor='black', capsize=10)
    # ax.errorbar(np.mean(np.squeeze(SNRimps[:,ii,:]), axis=0), np.std(np.squeeze(SNRimps[:,ii,:]), axis=0))
plt.legend([ph1,ph2],['GEVD rank 1','GEVD rank 2'])
plt.xticks(ticks=np.arange(SNRimps.shape[1]), labels=['None', '1s every 1s', '1s every 3s'])
ax.set(xlabel='Forced speech pauses', ylabel='[dB]', title='SNR improvements across %i indiv. nodes' % nSensors)
plt.show()

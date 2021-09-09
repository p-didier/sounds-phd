import numpy as np
import random
# Acoustic Scenario (AS) generation script.

nAS = 10		# Number of AS to generate
Fs = 16e3		# Sampling frequency [samples/s]
RIR_l = 2**12	# RIR length [samples]
minRd = 3		# Smallest room dimension possible [m]
maxRd = 7		# Largest room dimension possible [m]

T60max = 1.5*RIR_l/Fs	# Largest possible T60
T60min = 0.4*RIR_l/Fs	# Smallest possible T60

for ii in range(nAS):

	rd = np.random.uniform(low=minRd, high=maxRd, size=(3,))	# Generate random room dimensions
	T60 = np.random.uniform(low=T60min, high=T60max, size=(1,))	# Generate random reverberation time
	V = np.prod(rd)
	print(V)

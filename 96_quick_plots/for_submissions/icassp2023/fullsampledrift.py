import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

params_epsSmall = [7.5, 2]
params_epsMedium = [75, 20]
params_epsLarge = [275, 50]

# Matplotlib parameters
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,1)
fig.set_size_inches(8.5, 3.5)
# Small SROs
x = np.linspace(params_epsSmall[0] - 3*params_epsSmall[1], params_epsSmall[0] + 3*params_epsSmall[1], 100)
pdf = stats.norm.pdf(x, params_epsSmall[0], params_epsSmall[1])
axes.plot(x, pdf / np.amax(pdf))
axes.grid()
# Medium SROs
x = np.linspace(params_epsMedium[0] - 3*params_epsMedium[1], params_epsMedium[0] + 3*params_epsMedium[1], 100)
pdf = stats.norm.pdf(x, params_epsMedium[0], params_epsMedium[1])
axes.plot(x, pdf / np.amax(pdf))
axes.grid()
# Large SROs
x = np.linspace(params_epsLarge[0] - 3*params_epsLarge[1], params_epsLarge[0] + 3*params_epsLarge[1], 100)
pdf = stats.norm.pdf(x, params_epsLarge[0], params_epsLarge[1])
axes.plot(x, pdf / np.amax(pdf))
axes.grid()
#
axes.legend(['Small SROs', 'Medium SROs', 'Large SROs'])
plt.tight_layout()	
plt.show()

#%%
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import plot_side_room

# rd2D = [3, 6]
# nNodes = 3
# rs = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(3,))
# rn = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(3,))
# rs = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(10, nNodes))


# fig = plt.figure(figsize=(8,4))
# ax = fig.add_subplot(111)
# plot_side_room(ax, rd2D, rs, rn, r, sensorToNodeTags, scatsize=20)


f, (a0, a1) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]})
a0[0].plot(np.random.random(10), 'b')
a0[1].plot(np.random.random(10), 'r')
from pathlib import Path, PurePath
import sys
import matplotlib.pyplot as plt
import numpy as np
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/_general_fcts')
from plotting.twodim import plot_side_room

rd2D = [3, 6]
nNodes = 3
rs = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(3,))
rn = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(3,))
rs = np.random.uniform(low=rd2D[0], high=rd2D[1], size=(10, nNodes))


fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
plot_side_room(ax, rd2D, rs, rn, r, sensorToNodeTags, scatsize=20)


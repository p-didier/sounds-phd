
import numpy as np
import matplotlib as mpl
from fake_package.fcns import outter_plot_fcn

# Testing runtime configuration ('rc') parameters
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True

data = np.random.random(100)

outter_plot_fcn(data)

stop = 1
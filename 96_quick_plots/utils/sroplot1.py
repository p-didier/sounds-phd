#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.style.use('default')  # <-- for Jupyter: white figures background


def plotsro(exportpath):
    n = 30
    x = np.arange(n)
    y = np.linspace(start=0, stop=n-1, num=n+1)

    f = 0.75/5
    t = np.linspace(start=0, stop=n-1, num=1000)
    u = np.sin(2 * np.pi * f * t)

    idx_x = np.zeros(len(x), dtype=int)
    for ii in range(len(x)):
        idx_x[ii] = np.argmin(np.abs(t - x[ii]))
    idx_y = np.zeros(len(y), dtype=int)
    for ii in range(len(y)):
        idx_y[ii] = np.argmin(np.abs(t - y[ii]))

    fig = plt.figure(figsize=(8,1))
    ax = fig.add_subplot(111)
    plt.plot(x, np.full_like(x, fill_value=1), 'b.:')
    plt.plot(y, np.full_like(y, -1), 'r.:')
    plt.plot(t, u + 3, 'tab:gray')
    plt.plot(x, u[idx_x] + 3, 'b.')
    plt.plot(y, u[idx_y] + 3, 'r.')
    plt.grid(axis='both')
    plt.yticks([-1, 1, 3], labels=['Node $q$', 'Node $k$', 'True signal'])
    plt.ylim((-2, 5))
    plt.xticks(x, labels=[])
    plt.xlim((-0.5, int(n/2)))
    plt.xlabel('Time')
    fig.savefig(exportpath)
    plt.show()


def plotsto(exportpath):
    n = 30
    d = 0.2
    x = np.arange(n)
    y = np.linspace(start=0, stop=n-1, num=n) + d

    f = 0.75/5
    t = np.linspace(start=0, stop=n-1, num=1000)
    u = np.sin(2 * np.pi * f * t)

    idx_x = np.zeros(len(x), dtype=int)
    for ii in range(len(x)):
        idx_x[ii] = np.argmin(np.abs(t - x[ii]))
    idx_y = np.zeros(len(y), dtype=int)
    for ii in range(len(y)):
        idx_y[ii] = np.argmin(np.abs(t - y[ii]))

    fig = plt.figure(figsize=(8,1))
    ax = fig.add_subplot(111)
    plt.plot(x, np.full_like(x, fill_value=1), 'b.:')
    plt.plot(y, np.full_like(y, -1), 'r.:')
    plt.plot(t, u + 3, 'tab:gray')
    plt.plot(x, u[idx_x] + 3, 'b.')
    plt.plot(y, u[idx_y] + 3, 'r.')
    plt.grid(axis='both')
    plt.yticks([-1, 1, 3], labels=['Node $q$', 'Node $k$', 'True signal'])
    plt.ylim((-2, 5))
    plt.xticks(x, labels=[])
    plt.xlim((-0.5, int(n/2)))
    plt.xlabel('Time')
    fig.savefig(exportpath)
    plt.show()

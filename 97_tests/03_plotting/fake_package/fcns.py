
import matplotlib.pyplot as plt

def outter_plot_fcn(data):

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.plot(data)
    plt.show()

    return None
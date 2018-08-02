import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

def show_temperature_distribution(z_coords, temperatures):
    z = np.unique(z_coords)
    print(z)
    temp_ = []

    for z_ in z:
        temp = []
        for index, t in enumerate(temperatures):
            if z_coords[index] == z_:
                temp.append(t)
        temp_.append(np.average(temp))

    plt.plot(z, temp_)

def show_surface_plot(x, y, value):
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x, y, value)
    pylab.show()

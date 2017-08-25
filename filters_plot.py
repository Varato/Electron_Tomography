import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *

f = fftshift(fftfreq(64))
omega = 2*np.pi*f

ramp = 2*np.abs(f)
hamming = ramp*(0.54 + 0.46 * np.cos(omega / 2))
hann = ramp *(1 + np.cos(omega / 2)) / 2
cosine = np.cos(omega)
none = np.ones(len(f))

sl = ifftshift(ramp)
sl[1:] = sl[1:] * np.sin(ifftshift(omega)[1:]) / ifftshift(omega)[1:]
sl = fftshift(sl)

filters = [none, ramp, hamming, hann, sl, cosine]
fil_names = ["none", "ramp", "hamming", "hann", "Shepp-Logan", "cosine"]

fig, axes = plt.subplots(ncols = 3, nrows = 2, figsize = [9,7])

k = 0
for i in range(2):
    for j in range(3):
        axes[i,j].plot(f, filters[k], color="red")
        axes[i,j].set_title(fil_names[k])
        k += 1

plt.savefig("imgs/filters_plot.png")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
import skimage.transform as st
from PIL import Image

# cirle test
# Nx = Ny = 120
# x, y = np.mgrid[-5:5:Nx*1j, -5:5:Ny*1j]
# r2 = x*x + y*y
# obj = np.where(x*x/9 + y*y/4 <= 1, 1, 0)

# theta = range(0, 180, 1)
# sinogram = st.radon(obj, theta)
# rc_obj = st.iradon(sinogram, theta)


# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
# ax1.imshow(obj, cmap=plt.cm.Greys_r)
# ax2.imshow(sinogram, cmap = plt.cm.Greys_r)
# ax3.imshow(rc_obj, cmap = plt.cm.Greys_r)



# phantom test
phantom = imread(data_dir + "/phantom.png", as_grey=True)
phantom = st.rescale(phantom, scale=0.4, mode='reflect')

# def higt_pass_filter(in_arr, radius = 3):
#     (xl,yl) = in_arr.shape
#     (xx,yy) = np.mgrid[-1:1:xl*1j, -1:1:yl*1j]
#     fil = np.where(xx*xx + yy*yy <= radius**2, 0, 1)
#     out_arr = fil * in_arr
#     return out_arr.copy()

def recon(dtheta, ax = None, axf1 = None, axf2 = None, filter = "ramp"):
    dtheta = int(dtheta)
    theta = range(0,180,dtheta)
    sinogram = st.radon(phantom, theta, circle = False)
    rc_phantom = st.iradon(sinogram, theta, circle = False, filter = filter)

    error = rc_phantom - phantom
    rms = np.sqrt(np.mean(error**2))

    ft = np.fft.fft2(rc_phantom, norm = "ortho")
    ft[0,0] = 0
    # ft = higt_pass_filter(ft, radius=0)
    spec = abs(np.fft.fftshift(ft))**2
    # maximum = np.max(spec)
    # spec /= maximum

    if ax:
        ax.imshow(rc_phantom, cmap = plt.cm.Greys_r)
        ax.set_title("{}˚, {}".format(dtheta, filter if filter else "None"))
        ax.set_xlabel("rms = {:.4f}".format(rms))
        ax.set_xticks([])
        ax.set_yticks([])
    if axf1:
        axf1.imshow(spec, cmap = plt.cm.Greys_r)
        axf1.axis("off")
    if axf2:
        slicef = spec[spec.shape[0]//2]
        axf2.plot(slicef)
        axf2.axis([0, 160, 0, 1])
        axf2.set_xticks([])
        axf2.set_xlabel("frequency")
    return rms

if __name__ == "__main__":
    dtheta = 2
    filters = [None, "ramp", "shepp-logan", "cosine"]
    fig, axes = plt.subplots(ncols = 4, nrows = 2, figsize = [11,7])

    for i in range(len(filters)):
        rms = recon (dtheta, ax = axes[0,i], axf1 = None, axf2 = axes[1,i], filter = filters[i])

    plt.savefig("imgs/filters_compare.png")
    plt.show()































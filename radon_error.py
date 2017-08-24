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
def error_at_dtheta(dtheta, ax=None, filter = "ramp"):
	dtheta = int(dtheta)
	theta = range(0,180,dtheta)
	sinogram = st.radon(phantom, theta, circle = True)
	rc_phantom = st.iradon(sinogram, theta, circle = True, filter = filter)

	error = rc_phantom - phantom
	rms = np.std(error)

	if ax:
		ax.imshow(rc_phantom, cmap = plt.cm.Greys_r)
		ax.set_xlabel("{}˚".format(dtheta))
		ax.set_xticks([])
		ax.set_yticks([])
	return rms

if __name__ == "__main__":
	dthetas = [1, 5, 10, 18, 50]
	rmses = []
	filter = "hann"

	# fig0, ax0 = plt.subplots()
	fig, axes = plt.subplots(ncols = len(dthetas)+1, nrows = 2)
	for i in range(len(dthetas)):
		rmses.append( error_at_dtheta(dthetas[i], ax=axes[1,i+1], filter=filter) )

	axes[1,0].imshow(phantom, cmap = plt.cm.Greys_r)
	axes[1,0].set_xticks([])
	axes[1,0].set_yticks([])
	axes[1,0].set_xlabel("origin")

	axes[0,0] = plt.subplot(2,1,1)
	axes[0,0].plot(dthetas, rmses, "r-o")
	axes[0,0].set_xlabel("angle interval")
	axes[0,0].set_ylabel("rms error")
	axes[0,0].set_title("error trend with filter = {}".format(filter))

	plt.savefig("imgs/{}_error".format(filter if filter else "None"))

	plt.show()






























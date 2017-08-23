import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
import skimage.transform as st

Nx = Ny = Nz = 120

np.random.seed(2017)

def construct_obj():
    x, y, z = np.mgrid[-100:100:Nx*1j, -100:100:Ny*1j, -100:100:Nz*1j]
    gwnoise = abs(np.random.normal(loc = 0, scale = .01,  size=[Nx, Ny, Nz]))
    #  obj = np.where(x*x/80**2 + y*y/50**2 + z*z/10**2 <= 1, gwnoise, 0)
    obj = np.where( np.logical_or( (x+30)**2/2025 + y**2/2025 + z**2/2025 <=1,\
        (x-50)**2/100 + y**2/8100 + z**2/900 <=1 ), gwnoise, 0 )
    return obj

def low_pass_filter_density_map(in_arr, damping=-1., thr=1.E-3, num_cycles=2):
    (xl,yl,zl) = in_arr.shape
    (xx,yy,zz) = np.mgrid[-1:1:xl*1j, -1:1:yl*1j, -1:1:zl*1j]
    fil = np.fft.ifftshift(np.exp(damping*(xx*xx + yy*yy + zz*zz)))
    out_arr = in_arr.copy()
    for i in range(num_cycles):
        ft = fil*np.fft.fftn(out_arr)
        out_arr = np.real(np.fft.ifftn(ft))
        out_arr *= (out_arr > thr)
    return out_arr.copy()

def slice_wise_radon(obj, theta=range(80)):
    sinograms = np.zeros( (min(Ny, Nz), len(theta), Nx) )
    for s in range(Nx):
        obj_slice = obj[s]
        sinograms[:,:,s] = st.radon(obj_slice, theta = theta, circle = True)
    return sinograms, theta

def slice_wise_iradon(sinograms, theta):
    recon_obj = np.zeros( (Nx,) + (min(Ny, Nz),)*2 )
    for s in range(Nx):
        recon_obj[s,:,:] = st.iradon(sinograms[:,:,s], theta=theta, circle=True)
    return recon_obj

def constract_projections(sinograms, theta):
    projections = np.zeros((min(Nx, Ny), Nx, len(theta)))
    for t in range(len(theta)):
        projections[:,:,t] = sinograms[:,t,:]
    return projections

def get_obj_slice(obj, index, axis=0):
    if axis == 0:
        return obj[index,:,:]
    elif axis == 1:
        return obj[:,index,:]
    elif axis == 2:
        return obj[:,:,index]
    else:
        print("wrong axis input.")


if __name__ == "__main__":
    obj = construct_obj()
    # obj = low_pass_filter_density_map(obj)


    angle_interval = 5
    theta = range(0,180,angle_interval)
    sinograms, theta = slice_wise_radon(obj, theta=theta)
    projections = constract_projections(sinograms, theta)

    recon_obj = slice_wise_iradon(sinograms, theta)

    error = recon_obj - obj
    rms = np.std(error)
    print("rms error = {}".format(np.std(error)))

    # visulization 1: comparasion
    fig1, ((ax1,ax2), (ax3,ax4)) = plt.subplots(ncols = 2, nrows = 2)
    index = int(Ny/2)
    axis = 1
    im1 = ax1.imshow(get_obj_slice(obj, index, axis), cmap=plt.cm.Greys_r)
    im2 = ax2.imshow(get_obj_slice(recon_obj, index, axis), cmap=plt.cm.Greys_r)
    index = int(Nz/2)
    axis = 2
    im3 = ax3.imshow(get_obj_slice(obj, index, axis), cmap=plt.cm.Greys_r)
    im4 = ax4.imshow(get_obj_slice(recon_obj, index, axis), cmap=plt.cm.Greys_r)

    ax1.text(30,-10,"original obj")
    ax2.text(20,-10,"reconstructed obj")
    ax1.text(-60, 60, "mid slice \n along y")
    ax3.text(-60, 60, "mid slice \n along z")
    ax4.text(-100, 140, "angle interval = {} ,rms error = {:.5f}".format(angle_interval, rms))

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    fig1.colorbar(im1, ax=ax1)
    fig1.colorbar(im2, ax=ax2)
    fig1.colorbar(im3, ax=ax3)
    fig1.colorbar(im4, ax=ax4)


    plt.savefig("mid_slices.png")

    # visulization 2: sinograms
    index = 90
    fig2, (ax1, ax2) = plt.subplots(ncols = 2)
    print(sinograms[:,:,3].shape)
    im1 = ax1.imshow(get_obj_slice(obj, index, axis=0), cmap = plt.cm.Greys_r)
    im2 = ax2.imshow(sinograms[:,:,index], cmap = plt.cm.Greys_r)
    ax1.set_title("slice along x")
    ax2.set_title("corresponding sinogram")
    fig2.colorbar(im1, ax = ax1)
    fig2.colorbar(im2, ax = ax2)
    plt.savefig("sinograms.png")


    # visulization 3: projections
    indexes = np.linspace(0, len(theta)-1, 5, dtype=int)
    fig3, axes = plt.subplots(ncols = len(indexes))
    for i in range(len(indexes)):
        im = axes[i].imshow(projections[:,:,indexes[i]], cmap = plt.cm.Greys_r)
        axes[i].axis("off")
        axes[i].set_title("proj @ {}".format(theta[indexes[i]]))
        # fig3.colorbar(im, ax = axes[i])
    plt.savefig("projections.png")




    plt.show()






import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage as ndi

fname = sys.argv[1]

with h5py.File(fname, 'r') as f:
    t = f['t'][:]
    x = f['x'][:]
    y = f['y'][:]
    Z = f['z'][:]
    E = f['e'][:]
    N = f['n'][:]

xx, yy = np.meshgrid(x, y)

for i, k in enumerate(range(t.shape[0])):

    zz = Z[:,:,k]

    if 1: # smooth and interpolate before regridding
        #gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
        #zz = convolve(zz, gauss_kernel, boundary='wrap', normalize_kernel=True)
        zz = ndi.median_filter(zz, 3)

    zz[np.abs(zz)>np.nanstd(zz)*3] = np.nan

    #zz -= np.nanmean(zz)

    std = np.nanstd(zz) * 3

    plt.figure()
    plt.pcolormesh(xx, yy, zz, vmin=-std, vmax=std, cmap=plt.cm.RdBu)
    plt.title(str(t[i]))

plt.show()

import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]
xvar = sys.argv[2]
yvar = sys.argv[3]
zvar = sys.argv[4]

try:
    xy = sys.argv[5]
except:
    xy = None


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


with h5py.File(fname, 'r') as f:
    x = f[xvar][:]
    y = f[yvar][:]
    z = f[zvar][:]

    if xy:
        x, y = transform_coord(4326, 3031, x, y)


# Plot map
cmap = plt.cm.PiYG
#cmap = plt.cm.RdBu
#cmap = plt.cm.RdYlBu
vmin = - .1 #np.nanstd(z) * 3
vmax =   .1 #np.nanstd(z) * 3
plt.scatter(x.ravel(), y.ravel(), c=z.ravel(), s=3,
        vmin=vmin, vmax=vmax,
        cmap=cmap, rasterized=True)
'''
plt.contourf(x, y, z, 60,
        vmin=vmin, vmax=vmax,
        cmap=cmap, rasterized=True)
'''

plt.colorbar()

plt.show()


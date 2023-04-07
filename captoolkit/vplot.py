import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# --- Read in velocity field --------------------------------

# This can be passed as command-line arg.
FNAME = ""

# Default variable names in the NetCDF file
XVAR = "x"
YVAR = "y"
ZVAR = "v"
UVAR = "vx"
VVAR = "vy"

step = 1

# Limits to subset data (Ross)
# x1, x2, y1, y2 = -600000, 400000, -1400000, -800000  # Ross
x1, x2, y1, y2 = -3333000, 3333000, -3333000, 3333000  # Bedmap2


def load_velocity(
    fname, vname=["x", "y", "vx", "vy"], step=1, xlim=None, ylim=None, flipud=False
):
    """ Load velocity grid from NetCDF4. """

    # Default variable names in the NetCDF file
    xvar, yvar, uvar, vvar = vname

    if ".h5" in fname:
        d = h5py.File(fname, "r")
    else:
        ds = Dataset(fname, "r")
        d = ds.variables

    x = d[xvar][:]  # 1d
    y = d[yvar][:]  # 1d
    U = d[uvar][:]
    V = d[vvar][:]

    # Limits to subset data
    if xlim is not None:
        x1, x2 = xlim
        y1, y2 = ylim
        (j,) = np.where((x > x1) & (x < x2))
        (i,) = np.where((y > y1) & (y < y2))
        x = x[j.min() : j.max()]
        y = y[i.min() : i.max()]
        U = U[i.min() : i.max(), j.min() : j.max()]  # 2d
        V = V[i.min() : i.max(), j.min() : j.max()]  # 2d

    if ".h5" in fname:
        d.close()
    else:
        ds.close()

    if flipud:
        U, V = np.flipud(U), np.flipud(V)

    return x[::step], y[::step], U[::step, ::step], V[::step, ::step]


ifile = sys.argv[1] if sys.argv[1:] else FNAME

vnames = [XVAR, YVAR, UVAR, VVAR]
x, y, U, V = load_velocity(ifile, vname=vnames, step=step)

S = np.sqrt(U * U + V * V)

plt.figure()
plt.pcolormesh(x, y, S, rasterized=True, vmin=0, vmax=1000)
plt.colorbar()

plt.show()

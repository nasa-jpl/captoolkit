"""
Regrid a 2D velocity field (u, v) onto a 2D height field.

"""

import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.convolution import (Gaussian2DKernel, convolve,
                                 interpolate_replace_nans)
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

# === EDIT =================================================

# NOTE: Process one field at a time (u, v)

# NetCDF4 2D velocity data
path1 = "/Users/paolofer/data/velocity/summary/ANT_G0240_0000_merge_w_phase.nc"
xvar1 = "x"
yvar1 = "y"

# >>> OPTION: vx | vy
zvar1 = "vx"

# File with coordinates for regridding
path2 = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"
xvar2 = "x"
yvar2 = "y"

# Save regridded velocity as
# >>> OPTION: u_ref10 | v_ref10
zvar2 = "u_ref10"

# Where to save the regridded field
FILE_OUT = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

# ==========================================================


def h5read(fname, vnames):
    with h5py.File(fname, "r") as f:
        variables = [f[v][()] for v in vnames]

        return variables if len(vnames) > 1 else variables[0]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, _v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(_v)
                print("updated", k)
            else:
                f[k] = np.squeeze(_v)
                print("created", k)


def ncread(ifile, vnames):
    ds = Dataset(ifile, "r")  # NetCDF4
    d = ds.variables

    return [d[v][:] for v in vnames]


print("Reading netcdf4 velocty ...")

x1, y1, z1 = ncread(path1, [xvar1, yvar1, zvar1])
da = xr.DataArray(z1, dims=("y", "x"), coords={"y": y1, "x": x1})

print(x1[1] - x1[0], y1[1] - y1[0])

print("Reading hdf5 x/y coordinates...")

x2, y2 = h5read(path2, [xvar2, yvar2])

print("Smoothing ...")

veloc = da.values[:]

plt.matshow(da.values[:], vmin=-500, vmax=500, cmap='RdBu')

da.values[:] = convolve(
    veloc, Gaussian2DKernel(5, x_size=25, y_size=25), boundary="extend"
)

plt.matshow(da.values[:], vmin=-500, vmax=500, cmap='RdBu')

print("Regridding ...")
print("old shape:", da.values.shape)

da = da.interp(x=x2, y=y2)

print("new shape:", da.values.shape)

plt.matshow(da.values[:], vmin=-500, vmax=500, cmap='RdBu')

plt.show()

h5save(FILE_OUT, {zvar2: da.values}, "a")
print("saved ->", FILE_OUT)

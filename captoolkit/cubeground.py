# -*- coding: utf-8 -*-
"""
Resample grounded cube (t,y,x) onto floating cube (y,x,t).

Notes:
    Grounded cube is provided monthly, floating is quarterly
    Grounded cube is not corrected for FAC

"""
import sys
import warnings

import h5py
import numpy as np
import xarray as xr
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from netCDF4 import Dataset

warnings.filterwarnings("ignore")


# --- EDIT HERE -------------------------------------------------

fgrd = "/Users/paolofer/data/grounded/synthesis_19852019_r3840m.nc"  # Johan
# fgrd = "/Users/paolofer/data/grounded/sec_mrg.nc"  # Ludwig
tvar = "time"
xvar = "x"
yvar = "y"
zvar = "elev"  # Johan
# zvar = "sec"  # Ludwig

fcube = "/Users/paolofer/work/melt/data/FULL_CUBE_v3_REDUCED.h5"
tcube = "t"
xcube = "x"
ycube = "y"

mask_grounded = "mask_grounded"

# saveas = "dh_ground_ludwig"
saveas = "dh_ground_johan2"

# Averaging window (same as melt product -> 5 months)
window = 5  # TODO: Check this for xarray.rolling()

# --- END EDIT --------------------------------------------------


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def ncread(ifile, vnames):
    ds = Dataset(ifile, "r")  # NetCDF4
    d = ds.variables

    return [d[v][:] for v in vnames]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            f[k] = np.squeeze(v)


def running_mean_cube(cube, window=3, axis=0):
    half = int(window / 2.0)

    for k in range(half, cube.shape[axis] - half + 1):
        if axis == 0:
            cube[k] = np.nanmean(cube[k - half : k + half + 1], axis=axis)
        elif axis == 2:
            cube[:, :, k] = np.nanmean(cube[:, :, k - half : k + half + 1], axis=axis)
        else:
            print("averaging axis must be 0 or 2")

    return cube


def find_nearest(arr, val):
    return (np.abs(arr - val)).argmin()


ifile = sys.argv[1] if sys.argv[1:] else fgrd

print("loading CUBE FLOAT x/y/t ...")
t_cube, x_cube, y_cube = h5read(fcube, [tcube, xcube, ycube])

print("loading CUBE GROUND x/y/t/z ...")
t_grd, x_grd, y_grd, grd = ncread(fgrd, [tvar, xvar, yvar, zvar])

da_grd = xr.DataArray(grd, [("time", t_grd), ("y", y_grd), ("x", x_grd)])

da_grd = da_grd.where(da_grd != -9999, 0)  # replace -9999 -> 0

print("averaging in time ...")
# da_grd = da_grd.rolling(time=window, center=True).mean()  #FIXME: Not working
da_grd.values[:] = running_mean_cube(da_grd.values, window, axis=2)

print("regridding in time ...")
da_grd = da_grd.interp(time=t_cube)

print("extending spatial boundaries ...")

for k in range(len(t_cube)):
    da_grd.values[k] = interpolate_replace_nans(
        da_grd.values[k], Gaussian2DKernel(1), boundary="extend"
    )

print("regridding in space ...")
da_grd = da_grd.interp(x=x_cube, y=y_cube)

print("masking grounded ...")
(mask,) = h5read(fcube, [mask_grounded])
da_grd.coords["mask"] = (("y", "x"), mask)
da_grd = da_grd.where(da_grd.mask == 1)

if 1:
    # Save data to original cube file
    da_grd = da_grd.transpose("y", "x", "time")  # (t,y,x) -> (y,x,t)

    # h5save(fcube, {saveas: da_grd.values}, "a")
    # print("saved ->", fcube)

    h5save("JUNK.h5", {saveas: da_grd.values}, "w")

# --- Test ----------------------------------------------

if 1:

    # Load
    (grd,) = h5read("JUNK.h5", [saveas])
    (t, H) = h5read(fcube, ["t", "H8"])

    # Plot
    import matplotlib.pyplot as plt

    i, j, name = 770, 365, "PIG Grounded"
    i_, j_, name = 831, 364, "PIG Floating"

    a = grd[i, j, :]
    b = H[i_, j_, :]

    k = find_nearest(t, 2005)
    a -= a[k]
    b -= b[k]

    a = np.gradient(a, 0.25)
    b = np.gradient(b, 0.25)

    plt.plot(t, a, t, b, label=("grounded", "floating"))
    plt.ylabel("m")
    plt.show()

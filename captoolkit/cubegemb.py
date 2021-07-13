# -*- coding: utf-8 -*-
"""
Resample GEMB cube(s) (y,x,t) onto floating cube (y,x,t).

Notes:
    Run code twice: once for SMB and once for Firn.
    Edit params for each run and according data resolution.

"""
import warnings

import sys
import h5py
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from glob import glob
from numba import jit
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

warnings.filterwarnings("ignore")

# === EDIT HERE ================================================== #

# fgemb = "/Users/paolofer/data/smb/gemb/GEMB_SMB_*.nc"
fgemb = "/Users/paolofer/data/firn/gemb/GEMB_FAC_*.nc"
tgemb = "t"
xgemb = "x"
ygemb = "y"
# zgemb = "SMB"
zgemb = "FAC"

fcube = "data/FULL_CUBE_v3.h5"
tcube = "t"
xcube = "x"
ycube = "y"

# saveas = "smb_gemb8"
saveas = "fac_gemb8"

# Averaging window (time steps = months)
window = 5  # new GEMB dt = 1 month

SAVE = True
PLOT = True

# TODO: With the new GEMB data:
# - Stack multiple file grids
# - No need to subset in time
# - Change the averaging window (t_step == 1 month)
# - Perform all operations directily with xarray

# === END EDIT =================================================== #


def h5read(fname, vnames):
    with h5py.File(fname, "r") as f:
        variables = [f[v][()] for v in vnames]
        return variables if len(vnames) > 1 else variables[0]


def ncread(fname, vnames):
    with nc.Dataset(fname, "r") as ds:
        d = ds.variables
        variables = [d[v][:].filled(fill_value=np.nan) for v in vnames]
        return variables if len(vnames) > 1 else variables[0]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(v)
                print(k, "updated ->", fname)
            else:
                f[k] = np.squeeze(v)
                print(k, "created ->", fname)


def stack_grids(fnames, gridname):
    """Read grids from several files and return stacked grids."""
    return np.dstack([ncread(f, gridname) for f in fnames])


def stack_arrays(fnames, arrname):
    """Read arrays from several files and return concat array."""
    return np.concatenate([ncread(f, arrname) for f in fnames])


@jit(nopython=True)
def running_mean_axis2(cube, window, out):
    """Fast moving average for 3D array along last dimension.

    Pass out = cube.copy() to keep original values at the ends (half window).
    """

    assert window % 2 > 0, "Window must be odd!"
    half = int(window / 2.0)

    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            series = cube[i, j, :]
            if np.isnan(series).all():
                continue

            for k in range(half, cube.shape[2] - half):
                start, stop = k - half, k + half + 1
                series_window = series[start:stop]

                asum = 0.0
                count = 0
                for n in range(window):
                    asum += series_window[n]
                    count += 1

                out[i, j, k] = asum / count


def regrid_nearest_axis2(cube, t_cube, t_new):
    """Find nearest values along last dimension."""
    i_new = find_nearest(t_cube, t_new)
    return cube[:, :, i_new]


def find_nearest(arr, val):
    """Find index(es) of "nearest" value(s).

    Parameters
    ----------
    arr : array_like (ND)
        The array to search in (nd). No need to be sorted.
    val : scalar or array_like
        Value(s) to find.

    Returns
    -------
    out : tuple
        The index (or tuple if nd array) of nearest entry found. If `val` is a
        list of values then a tuple of ndarray with the indices of each value
        is return.

    See also
    --------
    find_nearest2

    """
    idx = []
    if np.ndim(val) == 0:
        val = np.array([val])
    for v in val:
        idx.append((np.abs(arr - v)).argmin())
    idx = np.unravel_index(idx, arr.shape)
    return idx if len(idx) > 1 else idx[0]


print("loading CUBE FLOAT x/y/t ...")
t_cube, x_cube, y_cube = h5read(fcube, [tcube, xcube, ycube])

print("loading multiple grids GEMB x/y/t/z ...")
fgembs = glob(fgemb)
fgembs.sort()

grids = []
for f in fgembs:
    gemb, x_gemb, y_gemb, t_gemb = ncread(f, [zgemb, xgemb, ygemb, tgemb])
    grids.append(
        xr.DataArray(
            gemb,
            dims=("y", "x", "t"),
            coords={"t": t_gemb, "y": y_gemb, "x": x_gemb},
        )
    )

da_gemb = xr.concat(grids, dim="t")

print("averaging in time ...")
da_gemb = da_gemb.rolling(t=window, center=True).mean()

print("regridding in time ...")
da_gemb = da_gemb.interp(t=t_cube)

print(da_gemb)

print("extending spatial boundaries before ...")
for k in range(t_cube.size):
    da_gemb.values[:, :, k] = interpolate_replace_nans(
        da_gemb.values[:, :, k], Gaussian2DKernel(2), boundary="extend"
    )

print("regridding in space ...")
da_gemb = da_gemb.interp(x=x_cube, y=y_cube)

print("extending spatial boundaries after ...")
for k in range(t_cube.size):
    da_gemb.values[:, :, k] = interpolate_replace_nans(
        da_gemb.values[:, :, k], Gaussian2DKernel(5), boundary="extend"
    )
    da_gemb.values[:, :, k] = interpolate_replace_nans(
        da_gemb.values[:, :, k], Gaussian2DKernel(3), boundary="extend"
    )

if 0:
    print('masking grounded ...')
    mask = h5read(fcube, ['mask_floating'])
    da_gemb.coords['mask'] = (('y', 'x'), mask)
    da_gemb = da_gemb.where(da_gemb.mask == 1)

# Save data to original cube file
da_gemb = da_gemb.transpose("y", "x", "t")  # (t,y,x) -> (y,x,t)

if SAVE:
    h5save(fcube, {saveas: da_gemb.values}, "a")
    print('data saved.')

    # h5save(
    #     "FAC_GEMB.h5",
    #     {
    #         saveas: da_gemb.values,
    #         "x": da_gemb.x.values,
    #         "y": da_gemb.y.values,
    #         "t": da_gemb.t.values,
    #     },
    #     "w",
    # )

# --- Plot ----------------------------------------------

if PLOT:

    from utils import test_ij_3km

    _, B, C, x, y, t = h5read(
        fcube, ["fac_gemb", "fac_imau", "fac_gsfc", "x", "y", "t"]
    )

    A = da_gemb.values

    i, j = test_ij_3km["STEP_2"]
    # i, j, name = 770, 365, 'PIG Grounded'
    # i, j, name = 831, 364, "PIG Floating"

    a = A[i, j, :]
    b = B[i, j, :]
    c = C[i, j, :]

    a -= a[0]
    b -= b[0]
    c -= c[0]

    plt.plot(t, a, label="FAC GEMB")
    plt.plot(t, b, label="FAC IMAU")
    plt.plot(t, c, label="FAC GSFC")
    plt.ylabel("FAC (m)")
    plt.legend()
    plt.show()

    plt.matshow(A[:, :, 10], cmap="RdBu")
    plt.plot([j], [i], 'or')
    plt.matshow(B[:, :, 10], cmap="RdBu")
    plt.plot([j], [i], 'ok')
    plt.show()


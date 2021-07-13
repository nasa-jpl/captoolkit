# -*- coding: utf-8 -*-
"""
Convert FAC cube to h_fac (m) -> height cube.

Notes:
    FAC should be applied to height (h), not to thickness (H).
    FAC is not hydrostatically compensated so: altim dh = full dFAC + change.
    Before applying FAC to h all time series must be referenced to the same t.

"""
import sys
import warnings
import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numba import jit

# from tqdm import trange

warnings.filterwarnings("ignore")

# === EDIT HERE ==================================================

# fgemb = '/Users/paolofer/data/smb/gemb/GEMB_FAC_1979-2018_5k.nc'
fgemb = "/Users/paolofer/data/smb/gemb/GEMB_SMB_1979-2018_5k.nc"
tgemb = "t"
xgemb = "x"
ygemb = "y"
# zgemb = 'FAC'
zgemb = "SMB"

fcube = "cube_full/FULL_CUBE_v2.h5"
tcube = "t"
xcube = "x"
ycube = "y"

# saveas = 'fac_gemb'
saveas = "smb_gemb"

# Averaging window (tme steps = months)
window = 31  # GEMB dt = 5 days!!!

# Time interval for subsetting
t1, t2 = 1991.0, 2019.0

# === END EDIT ===================================================


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def ncread(ifile, vnames):
    ds = nc.Dataset(ifile, "r")
    d = ds.variables
    return [d[v][:].filled(fill_value=np.nan) for v in vnames]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            f[k] = np.squeeze(v)


@jit(nopython=True)
def running_mean_along_axis0(cube, window, out):
    """Fast moving average for 3D array along first dimension."""

    assert window % 2 != 0, "Window must be odd!"
    half = int(window / 2.0)

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            series = cube[:, i, j]
            if np.isnan(series).all():
                continue

            for k in range(half, cube.shape[0] - half):
                start, stop = k - half, k + half + 1
                series_window = series[start:stop]

                asum = 0.0
                count = 0
                for n in range(window):
                    asum += series_window[n]
                    count += 1

                out[k, i, j] = asum / count


def regrid_time_nearest(cube, t_cube, t_new):
    i_new = find_nearest(t_cube, t_new)
    return cube[:, :, i_new]


# find_nearest = lambda arr, val: (np.abs(arr-val)).argmin()


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


ifile = sys.argv[1] if sys.argv[1:] else fgemb

print("loading CUBE FLOAT x/y/t ...")
t_cube, x_cube, y_cube = h5read(fcube, [tcube, xcube, ycube])

print("loading CUBE GEMB x/y/t/z ...")
t_gemb, x_gemb, y_gemb, gemb = ncread(fgemb, [tgemb, xgemb, ygemb, zgemb])
print(gemb.shape, type(gemb))

print("subsetting in time ...")
(k,) = np.where((t_gemb > t1) & (t_gemb < t2))
gemb = gemb[:, :, k[0] : k[-1]]
t_gemb = t_gemb[k[0] : k[-1]]

# print('creating xarray ...')
# da_gemb = xr.DataArray(gemb, dims=('y','x','time'),
#                       coords={'time': t_gemb, 'y': y_gemb, 'x': x_gemb})

# print(da_gemb)

# NOTE: Excessively slow!
# Subset cube in time
# da_gemb = da_gemb.where((da_gemb.time > t1) & (da_gemb.time < t2), drop=True)

print("averaging in time ...")
# da_gemb = da_gemb.rolling(time=window, center=True).mean()
out = np.zeros_like(gemb)
running_mean_along_axis2(gemb, window, out)
gemb = out

print("regridding in time ...")
# da_gemb = da_gemb.interp(time=t_cube)
gemb = regrid_time_nearest(gemb, t_gemb, t_cube)
t_gemb = t_cube

print("creating xarray ...")
da_gemb = xr.DataArray(
    gemb,
    dims=("y", "x", "time"),
    coords={"time": t_gemb, "y": y_gemb, "x": x_gemb},
)

print("extending spatial boundaries ...")
for k in range(t_cube.size):
    da_gemb.values[:, :, k] = interpolate_replace_nans(
        da_gemb.values[:, :, k], Gaussian2DKernel(1), boundary="extend"
    )

print("regridding in space ...")
da_gemb = da_gemb.interp(x=x_cube, y=y_cube)

# print('masking grounded ...')
# mask, = h5read(fcube, ['grounded_mask'])
# da_gemb.coords['mask'] = (('y', 'x'), mask)
# da_gemb = da_gemb.where(da_gemb.mask == 1)

# Save data to original cube file
da_gemb = da_gemb.transpose("y", "x", "time")  # (t,y,x) -> (y,x,t)
h5save(fcube, {saveas: da_gemb.values}, "a")
h5save("SMB_GEMB.h5", {saveas: da_gemb.values}, "w")
print("saved ->", fcube)

# --- Plot ----------------------------------------------

if 0:

    (A,) = h5read("SMB_GEMB.h5", ["smb_gemb"])
    (B,) = h5read(fcube, ["smb_racmo"])
    (t,) = h5read(fcube, ["t"])

    import matplotlib.pyplot as plt

    # i, j, name = 770, 365, 'PIG Grounded'
    i, j, name = 831, 364, "PIG Floating"

    a = A[i, j, :]
    b = B[i, j, :]

    a -= np.nanmean(a)
    b -= np.nanmean(b)

    a = np.cumsum(a)
    b = np.cumsum(b)

    a -= np.nanmean(a)
    b -= np.nanmean(b)

    plt.plot(t, a, label="SMB GEMB")
    plt.plot(t, b, label="SMB RACMO")
    plt.ylabel("SMB (m)")
    plt.legend()
    plt.show()

"""
Stack, Regrid, and Combine multiple velocity fields (GeoTIFFs).

1. Stack 2D geotiffs into a 3D xarray.DataArray
2. Regrid (to cube) and extrapolate in time
3. Combine 3D regrided field with 2D full (Alex + Eric) field

Notes:
    Regrid/Merge one component of velocity (vx, vy) at a time (edit below).

"""

import glob
import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.convolution import (Gaussian2DKernel, convolve,
                                 interpolate_replace_nans)
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

# === EDIT =================================================

# NOTE: Process one field at a time (u, v)

# GeoTIFF files to stack and regrid (for one component of velocity)
# >>> OPTION: *_vx.tif | *_vy.tif
path1 = "ASE/ASE_sim_v4/*_vx.tif"

# File with coordinates for regridding
path2 = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

# Component of 2D velocity to merge from CUBE
# >>> OPTION: u_ref | v_ref
vname = "u_ref10"

# Save new 3D velocity as
# >>> OPTION: u10 | v10
vsave = "u10"

# where to save
FILE_OUT = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

# ==========================================================


def get_time_from_filenames(filenames):
    """Helper function to create time array from file name."""

    return [float(os.path.split(f)[1].split("_")[2]) for f in filenames]


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


print("Reading and stacking geotiffs (regional 3D veloc) ...")

fnames = sorted(glob.glob(path1))

print(fnames)

time = xr.Variable("t", get_time_from_filenames(fnames))

chunks = {"x": 2000, "y": 2000, "band": 1}  # faster reads

da = xr.concat([xr.open_rasterio(f, chunks=None) for f in fnames], dim=time).squeeze(
    drop=True  # remove 'band' dim
)

print(da.x[1] - da.x[0], da.y[1] - da.y[0])

print("Reading hdf5 (continental 2D veloc) ...")

x, y, t, v = h5read(path2, ["x", "y", "t", vname])
da2 = xr.DataArray(v, dims=("y", "x"), coords={"y": y, "x": x})

print(da.values.shape)


plt.matshow(da.values[10, :, :], vmin=-1000, vmax=1000, cmap='RdBu')

if 1:

    # NOTE: Use 25 kernel size (10000 m) with 5 std (1200 m)

    veloc = da.values[:]

    mask_k = np.isnan(veloc)

    for k in range(veloc.shape[0]):

        print("smoothing slide #", k)

        v_k = veloc[k, :, :]

        v_k = convolve(
            v_k, Gaussian2DKernel(5, x_size=25, y_size=25), boundary="extend"
        )

        veloc[k, :, :] = v_k

    veloc[mask_k] = np.nan

    da.values = veloc

plt.matshow(da.values[10, :, :], vmin=-1000, vmax=1000, cmap='RdBu')

print("old shape:", da.values.shape)

# Regrid one field at a time due to lack of memory

interp_xy = []

for k in range(da.shape[0]):

    print("regridding slice #", k)

    da_k = da.isel(t=k)
    da_k = da_k.interp(x=x, y=y)
    interp_xy.append(da_k)

da_interp_xy = xr.concat(interp_xy, dim="t")

da = da_interp_xy.interp(t=t)

print("new shape:", da.values.shape)

print("Extrapolating values in time ...")

da = da.ffill(dim="t").bfill(dim="t")

print("Combining fields: regional 3D + full 2D -> full 3D ...")

da3 = da.combine_first(da2)

da3 = da3.transpose("y", "x", "t")
h5save(FILE_OUT, {vsave: da3.values}, "a")
print("saved ->", FILE_OUT)


# Plot
plt.figure()
da3.isel(t=10).plot(vmin=-1000, vmax=1000)

plt.figure()
ts1 = np.abs(da.isel(y=845, x=360))
ts2 = np.abs(da.isel(y=885, x=370))
ts3 = np.abs(da.isel(y=895, x=375))
ts4 = np.abs(da.isel(y=930, x=380))
ts5 = np.abs(da.isel(y=960, x=365))

ts1 -= ts1[0]
ts2 -= ts2[0]
ts3 -= ts3[0]
ts4 -= ts4[0]
ts5 -= ts5[0]

ts1.plot(label="PIG")
ts2.plot(label="Thwaites")
ts3.plot(label="Thwaites calv")
ts4.plot(label="Crosson")
ts5.plot(label="Dotson")
plt.legend()

plt.figure()
ts1 = np.abs(da.isel(y=825, x=365))
ts2 = np.abs(da.isel(y=870, x=380))
ts3 = np.abs(da.isel(y=885, x=390))
ts4 = np.abs(da.isel(y=940, x=390))
ts5 = np.abs(da.isel(y=950, x=385))

ts1 -= ts1[0]
ts2 -= ts2[0]
ts3 -= ts3[0]
ts4 -= ts4[0]
ts5 -= ts5[0]

ts1.plot(label="PIG G")
ts2.plot(label="Thwaites G")
ts3.plot(label="Thwaites calv G")
ts4.plot(label="Crosson G")
ts5.plot(label="Dotson G")
plt.legend()

plt.show()

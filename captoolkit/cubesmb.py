# -*- coding: utf-8 -*-
# TODO: Separate code into RACMO and ERA5
"""
1. Convert RACMO SMB to dH/dt_smb, same shape as full_cube.
2. Convert ERA5 precip+evap+runoff to dH/dt_smb, same shape as full_cube.

Units:
    RACMO:
        SMB [kg/m2/month]  # ignore the Meatadata!!!
        time [days since 1950-01-01 00:00:00.0], daily averaged values
    ERA5:
        Precip/Evap/Runoff [m/m2/day]  # ignore the Meatadata!!!
        time [hours since 1900-01-01 00:00:00.0], monthly averaged values

Recipe:
    - Convert time: hours -> days -> JD -> years
    - Convert units: 1/month * 12 -> 1/yr / 1/day * 365.2425 -> 1/yr
    - Convert mass: mass -> thickness [m of ice eq.] (assume solid ice)
    - Smooth w/5-month running mean
    - Regrid height-cube time series

Notes:
    Conversion from Mass to equivalent Height if no Firn Air has been removed
    (i.e. SMB has the volume of snow/firn):

    rho_w = 1028.  # density of ocean water (kg/m3)
    rho_i = 450.   # density of snow/firn (400-550 kg/m3)
    Z1 /= rho_i  # mass rate dM/dt (kg/yr) => thickness of snow/firn dh/dt (m/yr)
    Z1 *= (rho_w - rho_i) / rho_w  # buoyancy compensation (for floating ice)

"""
import sys
import warnings

import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


# import pyresample as pr

# === EDIT HERE =============================================

fsmb = "/Users/paolofer/data/smb/racmo/SMB_ANT27_monthly_RACMO2.3p2_197901_201612.nc"
fera5 = "/Users/paolofer/data/era5/adaptor.mars.internal-1559246138.871853-28867-12-aa84fcda-6a01-4161-9e59-ea415d26c27b.nc"
fcube = "cube_full/FULL_CUBE.h5"

# Default variable names in the SMB  NetCDF file
tvar = "time"
xvar = "lon"
yvar = "lat"
zvar = "smb"

tcube = "t"
xcube = "x"
ycube = "y"

saveas = "smb_era5"

# Averaging window (months)
window = 5

# Density of solid ice for converting mass -> thickness
rho_i = 917.0

# Time interval for subsetting
t1, t2 = 1991.5, 2019.0

# === END EDIT ==============================================


def day2dyr(time, since="1950-01-01 00:00:00"):
    """ Convert days since epoch to decimal years. """
    t_ref = Time(since, scale="utc").jd  # convert days to jd

    return Time(t_ref + time, format="jd", scale="utc").decimalyear


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Polar Stereo AnIS (x/y): 3031
        Polar Stereo GrIS (x/y): 3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))
    # proj2 = pyproj.Proj("+proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0")  # rotated lon/lat

    return pyproj.transform(proj1, proj2, x, y)


def sgolay1d(h, window=3, order=1, deriv=0, dt=1.0, mode="nearest", time=None):
    """Savitztky-Golay filter with support for NaNs

    If time is given, interpolate NaNs otherwise pad w/zeros.

    dt is spacing between samples.
    """
    h2 = h.copy()
    (ii,) = np.where(np.isnan(h2))
    (jj,) = np.where(np.isfinite(h2))

    if len(ii) > 0 and time is not None:
        h2[ii] = np.interp(time[ii], time[jj], h2[jj])
    elif len(ii) > 0:
        h2[ii] = 0
    else:
        pass
    h2 = savgol_filter(h2, window, order, deriv, delta=dt, mode=mode)

    return h2


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


def regrid_cube(x1, y1, z1, x2, y2):
    """
    Regrid Z1(t,y,x) onto Z2(y,x,t), keeping the original time.

    x1/y1/x2/y2 are 2D arrays.
    z1 is a 3D array.
    """
    z2 = np.full((x2.shape[0], x2.shape[1], z1.shape[0]), np.nan)  # -> (y,x,t)

    for k in range(z1.shape[0]):
        print("regridding:", k)
        z2[:, :, k] = griddata(
            (x1.ravel(), y1.ravel()),
            z1[k, :, :].ravel(),
            (x2, y2),
            fill_value=np.nan,
            method="linear",
        )

    return z2


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][:] for v in vnames]


def ncread(ifile, vnames):
    ds = Dataset(ifile, "r")  # NetCDF4
    d = ds.variables

    return [d[v][:] for v in vnames]


ifile = sys.argv[1] if sys.argv[1:] else fsmb

t_cube, x_cube, y_cube, cube, adv, div = h5read(
    fcube, [tcube, xcube, ycube, "H_thick", "advHv", "divHv"]
)

if 0:
    print("loading RACMO file ...")
    t_smb, lon_smb, lat_smb, smb = ncread(fsmb, [tvar, xvar, yvar, zvar])

    # Reduce SMB dimensions: 4 -> 3
    smb = smb[:, 0, :, :]

    # Convert "days since 1950-01-01 00:00:00" -> year
    t_smb = day2dyr(t_smb, since="1950-01-01 00:00:00")

    # Convert M/mo -> H/yr
    smb *= 12.0  # [kg/m2/mo] -> [kg/m2/yr]
    smb /= rho_i  # dM/dt [kg/yr] -> dH/dt [m/yr]

else:
    print("loading ERA5 file ...")
    t_smb, lon_smb, lat_smb, precip, evap, runoff = ncread(
        fera5, ["time", "longitude", "latitude", "tp", "e", "ro"]
    )

    # Subset
    lat_smb, precip, evap, runoff = (
        lat_smb[600:],
        precip[:, 600:, :],
        evap[:, 600:, :],
        runoff[:, 600:, :],
    )

    # Get SMB
    smb = precip - evap - runoff  # [m.w.eq./mo]

    # Convert "hours since 1900-01-01 00:00:00.0" -> years
    t_smb = day2dyr(t_smb / 24.0, since="1900-01-01 00:00:00")

    # Convert m.water.eq/day -> m.ice.eq/yr
    smb *= 365.2425  # [m/m2/day] -> [m/m2/yr]
    smb *= 1000 / rho_i  # (rho_w/rho_i) * H_w = H_i [m.w.e/yr] -> [m.i.e/yr]


# Subset SMB in time
(kk,) = np.where((t_smb > t1) & (t_smb < t2))
t_smb, smb = t_smb[kk], smb[kk, :, :]

if np.ndim(lon_smb) == 1:
    lon_smb, lat_smb = np.meshgrid(lon_smb, lat_smb)  # 1d -> 2d

# Transform geodetic -> polar stereo
# NOTE: Interp on x/y comes better than on lon/lat!
X_smb, Y_smb = transform_coord(4326, 3031, lon_smb, lat_smb)  # 2d
X_cube, Y_cube = np.meshgrid(x_cube, y_cube)  # 2d

if 0:
    print("Time averaging ...")
    smb = running_mean_cube(smb, window, axis=0)

    # Regrid in time
    import xarray as xr

    da_smb = xr.DataArray(
        smb, [("t", t_smb), ("y", range(smb.shape[1])), ("x", range(smb.shape[2]))]
    )

    print("regridding in time ...")
    smb = da_smb.interp(t=t_cube).values

    print("regridding in space ...")
    smb = regrid_cube(X_smb, Y_smb, smb, X_cube, Y_cube)


"""
print(smb.shape)
plt.pcolormesh(X_cube, Y_cube, smb[:,:,2])
#plt.matshow(smb[:,:,2])
plt.show()
"""

# --- Test ----------------------------------------------

# Subset region for testing (inclusive)
# Do not load full data into memory!

if 1:

    # Load
    with h5py.File("SMB_RACMO.h5", "r") as f:
        smb_racmo = f["smb"][:]

    with h5py.File("SMB_ERA5.h5", "r") as f:
        smb_era5 = f["smb"][:]

    # Plot
    import pandas as pd
    import matplotlib.pyplot as plt

    t = t_cube

    # i, j = 836, 368 # PIG
    # i, j = 366, 147 # Larsen C
    i, j = 510, 1600  # Amery

    mask = np.isnan(cube)
    smb_racmo[mask] = np.nan
    smb_era5[mask] = np.nan

    p = smb_racmo[i, j, :]
    p2 = smb_era5[i, j, :]
    H = cube[i, j, :]
    advec = adv[i, j, :]
    diver = div[i, j, :]

    """
    plt.matshow(smb_racmo[:,:,10])
    plt.matshow(smb_era5[:,:,10])
    plt.matshow(cube[:,:,10])
    plt.show()
    sys.exit()
    """

    p -= np.nanmean(p)
    p2 -= np.nanmean(p2)

    dHdt = sgolay1d(H, window=5, order=1, deriv=1, dt=t[1] - t[0], time=None)

    # dHdt += advec + diver
    dHdt -= np.nanmean(dHdt)

    print("SMB mean rate (m/yr):", np.nanmean(p))

    plt.figure(figsize=(14, 5))
    plt.subplot(211)
    plt.plot(t, p, linewidth=2, label="RACMO")
    plt.plot(t, p2, linewidth=2, label="ERA5")
    plt.plot(t, dHdt, linewidth=2, label="dH/dt")
    plt.legend()
    plt.title("SMB - Amery")
    plt.ylabel("meters of ice eq / yr")
    plt.subplot(212)
    plt.plot(
        t,
        dHdt - p,
        linewidth=2,
        label="dH/dt-RACMO (std=%.2f)" % np.nanstd(dHdt[:-5] - p[:-5]),
    )
    plt.plot(
        t,
        dHdt - p2,
        linewidth=2,
        label="dH/dt-ERA5 (std=%.2f)" % np.nanstd(dHdt[:-5] - p2[:-5]),
    )
    plt.legend()
    plt.ylabel("meters / yr")

    plt.show()
    sys.exit()


# NOTE: All cubes should be saved in the original h file

if 0:
    # Save
    with h5py.File("SMB_ERA5.h5", "w") as f:
        f["smb"] = smb
        f["x"] = x_cube
        f["y"] = y_cube
        f["t"] = t_cube

if 0:
    # Save data to orignal cube file
    with h5py.File(fcube, "a") as f:
        f[saveas] = smb

    print("data saved ->", fcube)

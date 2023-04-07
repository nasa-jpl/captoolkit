"""Estimate cube uncertainties.

- detrend full cube -> resid
- estimate std/sqrt(6) -> 2D field
- extend, filter, and mask 2D field
- calc combined error -> err_h, err_firn, err_veloc, err_smb

Variables:

    height -> h (dh_xcal)
    thickness -> H
    thickness change -> dHdt
    divergence -> div
    basal melt rate -> melt

Error propagation:

        err_h(x,y) = std(h - h_trend) / sqrt(6)  # 4 missions + 2 modes

        err_H(x,y) = sqrt(err_h^2 + err_fac**2) * buoyancy

    where `buoyancy = rho_ocean / (rho_ocean - rho_ice)`

    Assuming `err_t = 0` and `err_H = const in t`:

        err_dHdt(x,y) = sqrt(2) * err_H / dt

    where dt = 1 year.

    Assuming no significant error in `H` and `x/y`,
    and `err_u = err_v = const (and independent):

        err_div(x,y,t) = 2 * H * err_u / dx

    where dx = 3 km (the grid spacing).

        err_melt(x,y,t) = sqrt(err_dHdt^2 + err_div^2 + err_smb^2)

Timespan of each mission:

    ERS1: 1992-1996
    ERS2: 1995-2003
    ENVI: 2002-2011
    CRY2: 2010-2018
    ICES: 2003-2010

"""
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import median_filter

from utils import read_h5, save_h5, sgolay1d

warnings.filterwarnings("ignore")


# NOTE: Use as raw heights as possible
FILE_CUBE = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

tspans = [(1992, 1995.5), (1995.5, 2002.5), (2002.5, 2010.5), (2010.5, 2018)]


def _detrend(y, window=29):

    if np.isnan(y).all():
        y_resid = y
    else:
        y_trend = sgolay1d(y, window=window)
        y_resid = y - y_trend

    return y_resid


def std_residuals(y_):

    y = y_.copy()

    if np.isnan(y).all():
        y_std = np.nan
    else:
        y[y == 0] = np.nan
        y_std = np.nanstd(_detrend(y))

    return y_std


def nanstd(y):

    if np.isnan(y).all():
        return np.nan
    else:
        return np.nanstd(y)


# TODO: Edit this function variable read


def load_data(fname):

    vnames = ["x", "y", "t", "dh_xcal", "fac_gemb_err8", "smb_gemb_err8"]

    (x, y, t, dh, err_fac, err_smb) = read_h5(fname, vnames)

    # Create xarray
    ds = xr.Dataset(
        {
            "dh": (("y", "x", "t"), dh),
            "err_fac": (("y", "x", "t"), err_fac),
            "err_smb": (("y", "x", "t"), err_smb),
        },
        coords={"y": y, "x": x, "t": t},
    )

    print(ds)

    return ds


ds = load_data(FILE_CUBE)

# Get indices for each mission
k_tspans = [np.where((ds.t > t1) & (ds.t <= t2))[0] for t1, t2 in tspans]

# --- Error height (h) --- #

print("detrending and calc std ...")

std = np.apply_along_axis(std_residuals, 2, ds.dh)

mask_nan = np.isnan(std)

print("filtering std field ...")

std[std > 1] = 1.0  # Limit max std to 1m (to avoid garbage)

std = interpolate_replace_nans(std, Gaussian2DKernel(2), boundary="extend")

std = median_filter(std, size=3)

std[mask_nan] = np.nan

err_std = std / np.sqrt(6)  # four missions and two modes (ice/oce)

# NOTE: Do not use this!

if 0:
    # Using each mission error (3D)
    err_ra = np.array([0.23, 0.21, 0.13, 0.11]) / np.sqrt(2)  # One per mission

    err_full = [np.sqrt(err_std ** 2 + e_ra ** 2) for e_ra in err_ra]  # Four 2D fields

    # 2Ds -> 3D
    err_dh = np.dstack(
        [
            np.repeat(e[:, :, np.newaxis], len(k_tspan), axis=2)

            for e, k_tspan in zip(err_full, k_tspans)
        ]
    )

else:
    # Using only std (2D)
    err_dh = err_std

# --- Error thickness (H) --- #

rho_ocean, rho_ice = 1028.0, 917.0
buoyancy = rho_ocean / (rho_ocean - rho_ice)

# Load Firn error

err_H = np.sqrt(err_dh[:, :, None] ** 2 + ds.err_fac ** 2) * buoyancy

# --- Error thickness rate (dH/dt)--- #

dt = 1.0  # years

err_H2 = np.roll(err_H, 3, axis=2)
err_dHdt = np.sqrt(err_H ** 2 + err_H2 ** 2) / dt

# --- Error melt rate (b) --- #

H = read_h5(FILE_CUBE, ["H_filt10"])

# Assuming u and v are independent and have the same error

err_u = 5.0  # m/yr
dx = 3000.0  # m

err_div = 2 * np.abs(H) * err_u / dx

err_melt = np.sqrt(err_dHdt ** 2 + err_div ** 2 + ds.err_smb ** 2)

# Save

if 0:

    FILE_SAVE = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

    save_h5(
        FILE_SAVE,
        {
            "dh_err10": err_dh,
            "H_err10": err_H,
            "dHdt_net_err10": err_dHdt,
            "dHdt_div_err10": err_div,
            "dHdt_melt_err10": err_melt,
        },
    )  # not corrected for firn

    print("saved ->", FILE_SAVE)


# Plot to double check


def plot_error(t, cube, error, indices, title="", detrend=False):

    for k, (i, j) in enumerate(indices, start=1):

        y = cube[i, j, :]  # 3D

        if detrend:
            y = _detrend(y)

        if np.ndim(error) > 0:
            e = error[i, j, ...]  # 2D or 3D
        else:
            e = error

        plt.subplot(len(indices), 1, k)
        plt.errorbar(t, y, yerr=e, capsize=3)
        plt.title(title)


if 1:

    FILE_CUBE = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

    H, dHdt, div, melt = read_h5(
        FILE_CUBE, ["H_filt10", "dHdt_net10", "dHdt_div_filt10", "dHdt_melt10"]
    )

    indices = [
        (1100, 900),
        (845, 365),
        (1020, 400),
        (505, 1485),
        (250, 700),
        (360, 160),
        (1090, 1655),
    ]

    plt.figure()
    plot_error(ds.t, ds.dh.values, err_dh, indices, "Error in height", detrend=True)

    plt.figure()
    plot_error(ds.t, H, err_H, indices, "Error in thickness", detrend=True)

    plt.figure()
    plot_error(ds.t, dHdt, err_dHdt, indices, "Error in thickness rate")

    plt.figure()
    plot_error(ds.t, div, err_div, indices, "Error in divergence")

    plt.figure()
    plot_error(ds.t, melt, err_melt, indices, "Error in melt rate")

    plt.show()

"""
Regrid velocity field onto height/DEM grid.

"""
import sys
import h5py
import pyproj
import warnings
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from netCDF4 import Dataset

warnings.filterwarnings("ignore")


# === Edit ===========================================================

# Apply median filter to u and v before regridding
MEDIAN_FILT = True

# DEM/Cube file
HFILE = (
    "/Users/paolofer/code/captoolkit/captoolkit/work/cube_full/FULL_CUBE_v2.h5"
)

HVARS = ["x", "y", "h_mean_cs2"]

# HDF5 velocity file
VFILE = "/Users/paolofer/data/velocity/merged/ANT_G0240_0000_PLUS_450m_v2.h5"
# VFILE = '/Users/paolofer/data/velocity/gardner/ANT_G0240_0000.nc'  # Alex
# VFILE = '/Users/paolofer/data/velocity/rignot/antarctica_ice_velocity_450m_v2.nc'  # Eric

VVARS = ["x", "y", "vx", "vy"]  # Alex
# VVARS = ['x', 'y', 'VX', 'VY']  # Eric

# ====================================================================


def h5read(ifile, VVARS):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in VVARS]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            f[k] = np.squeeze(v)


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Stereo AnIS (x/y):  3031
        Stereo GrIS (x/y):  3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))
    return pyproj.transform(proj1, proj2, x, y)


def xregrid2d(x1, y1, z1, x2, y2):
    import xarray as xr

    da = xr.DataArray(z1, [("y", y1), ("x", x1)])
    return da.interp(x=x2, y=y2).values


def main():

    print("loading ...")

    x, y, h = h5read(HFILE, HVARS)
    x_v, y_v, u, v = h5read(VFILE, VVARS)

    # Replace NaNs with Zeros
    h_mask = np.isnan(h)
    u_mask = np.isnan(u)
    v_mask = np.isnan(v)
    h[h_mask] = 0
    u[u_mask] = 0
    v[v_mask] = 0

    if MEDIAN_FILT:
        u = ndi.median_filter(u, 3)
        v = ndi.median_filter(v, 3)

    print("re-gridding ...")

    u = xregrid2d(x_v, y_v, u, x, y)
    v = xregrid2d(x_v, y_v, v, x, y)

    if 0:
        h5save(HFILE, {"u": u, "v": v}, "a")
        print("saved.")


if __name__ == "__main__":
    main()

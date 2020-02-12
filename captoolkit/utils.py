"""
High-level functions used across the CAP-Toolkit package.

"""
import h5py
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal


# --- Utilitiy functions --- #


def print_args(args):
    """Print arguments passed to argparse."""
    print("Input arguments:")
    for arg in list(vars(args).items()):
        print(arg)


def read_h5(fname, vnames):
    """Generic HDF5 reader.

    vnames : ['var1', 'var2', 'var3']
    """
    with h5py.File(fname, "r") as f:
        variables = [f[v][()] for v in vnames]

        return variables if len(vnames) > 1 else variables[0]


def save_h5(fname, vardict, mode="a"):
    """Generic HDF5 writer.

    vardict : {'name1': var1, 'name2': va2, 'name3': var3}
    """
    with h5py.File(fname, mode) as f:
        for k, v in list(vardict.items()):
            if k in f:
                f[k][:] = np.squeeze(v)
            else:
                f[k] = np.squeeze(v)


def is_empty(ifile):
    """Test if file is corruted or empty"""
    try:
        with h5py.File(ifile, "r") as f:
            if bool(list(f.keys())):
                return False
            else:
                return True
    except IOError:
        return True


def find_nearest(arr, val):
    """Find index of 'nearest' value(s).

    Args:
        arr (nd array) : The array to search in (nd). No need to be sorted.
        val (scalar or array) : Value(s) to find.

    Returns:
        out (tuple or scalar) : The index (or tuple if nd array) of nearest
            entry found. If `val` is a list of values then a tuple of ndarray
            with the indices of each value is return.

    See also:
        find_nearest2

    """
    idx = []

    if np.ndim(val) == 0:
        val = np.array([val])

    for v in val:
        idx.append((np.abs(arr - v)).argmin())
    idx = np.unravel_index(idx, arr.shape)

    return idx if val.ndim > 1 else idx[0]


def mad_std(x, axis=None):
    """Robust standard deviation (using MAD)."""

    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


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
    # Convert coordinates

    return pyproj.transform(proj1, proj2, x, y)


# --- Processing functions --- #


def sgolay1d(h, window=3, order=1, deriv=0, dt=1.0, mode="nearest", time=None):
    """Savitztky-Golay filter with support for NaNs.

    If time is given, interpolate NaNs otherwise pad w/zeros.
    If time is given, calculate dt as t[1]-t[0].

    Args:
        dt (int): spacing between samples (for correct units).

    Notes:
        Works with numpy, pandas and xarray objects.

    """
    if isinstance(h, (pd.Series, xr.DataArray)):
        h = h.values
    if isinstance(time, (pd.Series, xr.DataArray)):
        time = time.values

    _h = h.copy()
    (i_nan,) = np.where(np.isnan(_h))
    (i_valid,) = np.where(np.isfinite(_h))

    if i_valid.size < 5:
        return _h
    elif time is not None:
        _h[i_nan] = np.interp(time[i_nan], time[i_valid], _h[i_valid])
        dt = np.abs(time[1] - time[0])
    else:
        _h[i_nan] = 0

    return signal.savgol_filter(_h, window, order, deriv, delta=dt, mode=mode)


# TODO: Think if dx, dy should be applied here !!!
def sgolay2d(z, window_size, order, derivative=None):
    """Two dimensional data smoothing and least-square gradient estimate.

    Code from:
        http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Reference:
        A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.

    """
    # number of terms in the polynomial expression
    # TODO: Double check this (changed for Py3)
    n_terms = (order + 1) * (order + 2) // 2

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size ** 2 < n_terms:
        raise ValueError("order is too high for the window size")

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2,)

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))

    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(
        np.flipud(z[1 : half_size + 1, :]) - band
    )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(
        np.flipud(z[-half_size - 1 : -1, :]) - band
    )
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(
        np.fliplr(z[:, 1 : half_size + 1]) - band
    )
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(
        np.fliplr(z[:, -half_size - 1 : -1]) - band
    )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(
        np.flipud(np.fliplr(z[1 : half_size + 1, 1 : half_size + 1])) - band
    )
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(z[-half_size - 1 : -1, -half_size - 1 : -1]))
        - band
    )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(
        np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band
    )
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(
        np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band
    )

    # solve system and convolve

    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))

        return signal.fftconvolve(Z, m, mode="valid")
    elif derivative == "col":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))

        return signal.fftconvolve(Z, -c, mode="valid")
    elif derivative == "row":
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))

        return signal.fftconvolve(Z, -r, mode="valid")
    elif derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))

        return (
            signal.fftconvolve(Z, -r, mode="valid"),
            signal.fftconvolve(Z, -c, mode="valid"),
        )


def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=False):
    """Construct output grid-coordinates."""
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1
    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)

    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi

# --- Test functions --- #


# Some edge test cases (for the 3-km grid)
test_ij_3km = [
    (845, 365),  # 0 PIG Floating 1
    (831, 364),  # 1 PIG Floating 2
    (1022, 840),  # 2 CS-2 only 1
    (970, 880),  # 3 CS-2 only 2
    (100, 1170),  # 4 fig1  large peaks at mission overlaps
    (100, 766),  # 5 fig2  peak at mission overlap
    (7, 893),  # 6 step change at beguining
    (8, 892),  # 7 with hole
    (9, 889),  # 8 with large hole
    (11, 893),  # 9 step in divergence
]

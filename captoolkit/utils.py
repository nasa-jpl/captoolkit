"""
High-level functions used across the CAP-Toolkit package.

"""
import h5py
import numpy as np
import pyproj
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.ndimage import map_coordinates
try:
    from gdalconst import *
except ImportError as e:
    from osgeo.gdalconst import *
from osgeo import gdal, osr
from scipy import signal

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



def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=False):
    """
    Construct 2D-grid given input boundaries

    :param xmin: x-coord. min
    :param xmax: x-coord. max
    :param ymin: y-coors. min
    :param ymax: y-coord. max
    :param dx: x-resolution
    :param dy: y-resolution
    :param return_2d: if true return grid otherwise vector
    :return: 2D grid or 1D vector
    """

    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)

    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi

def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2
    usgin EPSG number

    :param proj1: current projection (4326)
    :param proj2: target projection (3031)
    :param x: x-coord in current proj1
    :param y: y-coord in current proj1
    :return: x and y now in proj2
    """

    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))

    return pyproj.transform(proj1, proj2, x, y)


def mad_std(x, axis=None):
    """
    Robust std.dev using median absolute deviation

    :param x: data values
    :param axis: target axis for computation
    :return: std.dev (MAD)
    """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def interpmed(x, y, z, Xi, Yi, n, d):
    """
    2D median interpolation of scattered data

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param Xi: x-coord. grid (2D)
    :param Yi: y-coord. grid (2D)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :return: 1D array of interpolated values
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]

        zi[i] = np.median(zc)

    return zi


def interpgaus(x, y, z, s, Xi, Yi, n, d, a):
    """
    2D interpolation using a gaussian kernel
    weighted by distance and error

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. errors
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :return: 1D vec. of prediction, sigma and nobs
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    if np.all(np.isnan(s)): s = np.ones(s.shape)

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]
        sc = s[idx]

        if len(zc[~np.isnan(zc)]) == 0: continue

        # Weights
        wc = (1./sc**2) * np.exp(-(dxy**2)/(2*a**2))

        # Avoid singularity
        wc += 1e-6

        # Predicted value
        zi[i] = np.nansum(wc * zc) / np.nansum(wc)

        # Weighted rmse
        sigma_r = np.nansum(wc * (zc - zi[i])**2) / np.nansum(wc)

        # Obs. error
        sigma_s = 0 if np.all(s == 1) else np.nanmean(sc)

        # Prediction error
        ei[i] = np.sqrt(sigma_r ** 2 + sigma_s ** 2)

        # Number of points in prediction
        ni[i] = 1 if n == 1 else len(zc)

    return zi, ei, ni


def interpkrig(x, y, z, s, Xi, Yi, d, a, n):
    """
    2D interpolation using ordinary kriging/collocation
    with second-order markov covariance model.

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. error added to diagonal
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :param n: number of nearest neighbours
    :return: 1D vec. of prediction, sigma and nobs
    """

    n = int(n)

    # Check
    if n == 1:
        print('n > 1 needed!')
        return

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    # Convert to meters
    a *= 0.595 * 1e3
    d *= 1e3

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if dxy.min() > d:
            continue

        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]

        if len(zc) < 2: continue

        m0 = np.median(zc)
        c0 = np.var(zc)

        # Covariance function for Dxy
        Cxy = c0 * (1 + (dxy / a)) * np.exp(-dxy / a)

        # Compute pair-wise distance
        dxx = cdist(np.c_[xc, yc], np.c_[xc, yc], "euclidean")

        # Covariance function Dxx
        Cxx = c0 * (1 + (dxx / a)) * np.exp(-dxx / a)

        # Measurement noise matrix
        N = np.eye(len(Cxx)) * sc * sc

        # Solve for the inverse
        CxyCxxi = np.linalg.solve((Cxx + N).T, Cxy.T)

        # Predicted value
        zi[i] = np.dot(CxyCxxi, zc) + (1 - np.sum(CxyCxxi)) * m0

        # Predicted error
        ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))

        # Number of points in prediction
        ni[i] = len(zc)

    return zi, ei, ni


def spatial_filter(x, y, z, dx, dy, n_sigma=3.0):
    """
    Spatial outlier editing filter

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param dx: filter res. in x (m)
    :param dy: filter res. in y (m)
    :param n_sigma: cutt-off value
    :param thres: max absolute value of data
    :return: filtered array containing nan-values
    """

    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    f_bin = stats.binned_statistic_2d(x, y, z, bins=(Ne, Nn))

    index = f_bin.binnumber

    ind = np.unique(index)

    zo = z.copy()

    for i in range(len(ind)):

        # index for each bin
        idx, = np.where(index == ind[i])

        zb = z[idx]

        if len(zb[~np.isnan(zb)]) == 0:
            continue

        dh = zb - np.nanmedian(zb)

        foo = np.abs(dh) > n_sigma * np.nanstd(dh)

        zb[foo] = np.nan

        zo[idx] = zb

    return zo


def interp2d(x, y, z, xi, yi, **kwargs):
    """
    Raster to point interpolation based on
    scipy.ndimage import map_coordinates

    :param x: x-coord. in 2D (m)
    :param y: x-coord. in 2D (m)
    :param z: values in 2D
    :param xi: interp. point in x (m)
    :param yi: interp. point in y (m)
    :param kwargs: see map_coordinates
    :return: array of interp. values
    """

    x = np.flipud(x)
    y = np.flipud(y)
    z = np.flipud(z)

    x = x[0,:]
    y = y[:,0]

    nx, ny = x.size, y.size

    x_s, y_s = x[1] - x[0], y[1] - y[0]

    if np.size(xi) == 1 and np.size(yi) > 1:
        xi = xi * ones(yi.size)
    elif np.size(yi) == 1 and np.size(xi) > 1:
        yi = yi * ones(xi.size)

    xp = (xi - x[0]) * (nx - 1) / (x[-1] - x[0])
    yp = (yi - y[0]) * (ny - 1) / (y[-1] - y[0])

    coord = np.vstack([yp, xp])

    zi = map_coordinates(z, coord, **kwargs)

    return zi


def tiffread(ifile):
    """
    Reading tif-file to memory

    :param ifile: path+name of tif file
    :return: X, Y, Z, dx, dy and proj
    """

    file = gdal.Open(ifile, GA_ReadOnly)
    metaData = file.GetMetadata()
    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    proj = src.ExportToWkt()

    Nx = file.RasterXSize
    Ny = file.RasterYSize

    trans = file.GetGeoTransform()

    dx = trans[1]
    dy = trans[5]

    Xp = np.arange(Nx)
    Yp = np.arange(Ny)

    (Xp, Yp) = np.meshgrid(Xp, Yp)

    X = trans[0] + (Xp + 0.5) * trans[1] + (Yp + 0.5) * trans[2]
    Y = trans[3] + (Xp + 0.5) * trans[4] + (Yp + 0.5) * trans[5]

    band = file.GetRasterBand(1)

    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def tiffwrite(ofile, X, Y, Z, dx, dy, proj, otype='float'):
    """
    Writing raster to a tif-file

    :param ofile: name of ofile
    :param X: x-coord of raster (2D)
    :param Y: y-coord of raster (2D)
    :param Z: values (2D)
    :param dx: grid-spacing x
    :param dy: grid-spacing y
    :param proj: projection (epsg number)
    :param dtype: save as 'int' or 'float'
    :return: written file to memory
    """

    proj = int(proj)

    N, M = Z.shape

    driver = gdal.GetDriverByName("GTiff")

    if otype == 'int':
        datatype = gdal.GDT_Int32

    if otype == 'float':
        datatype = gdal.GDT_Float32

    ds = driver.Create(ofile, M, N, 1, datatype)

    src = osr.SpatialReference()

    src.ImportFromEPSG(proj)

    ulx = np.min(np.min(X)) - 0.5 * dx

    uly = np.max(np.max(Y)) + 0.5 * dy

    geotransform = [ulx, dx, 0, uly, 0, -dy]

    ds.SetGeoTransform(geotransform)

    ds.SetProjection(src.ExportToWkt())

    ds.GetRasterBand(1).SetNoDataValue(np.nan)

    ds.GetRasterBand(1).WriteArray(Z)

    ds = None


def binning(x, y, xmin=None, xmax=None, dx=1 / 12.,
             window=3 / 12., interp=False, median=False):
    """Time-series binning (w/overlapping windows).

        Args:
        x,y: time and value of time series.
        xmin,xmax: time span of returned binned series.
        dx: time step of binning.
        window: size of binning window.
        interp: interpolate binned values to original x points.
    """
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax, dx)  # time steps
    bins = [(ti, ti + window) for ti in steps]  # bin limits

    N = len(bins)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in range(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            xb[i] = 0.5 * (t1 + t2)
            continue

        ybv = y[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)
        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        try:
            yb = np.interp(x, xb, yb)
            eb = np.interp(x, xb, eb)
            sb = np.interp(x, xb, sb)
            xb = x
        except:
            pass

    return xb, yb, eb, nb, sb


def hampel_filter1d(x, k, t0=3):
    """
    Hampel-filter for outlier editing

    :param x: values
    :param k: window size (int)
    :param t0: sigma threshold value
    :return: filtered array with nan's
    """

    x = np.pad(x, k, 'constant', constant_values=9999)
    x[x == 9999] = np.nan
    n = len(x)
    y = x.copy()
    L = 1.4826

    for i in range((k + 1),(n - k)):
        if np.isnan(x[(i - k):(i + k+1)]).all():
            continue
        x0 = np.nanmedian(x[(i - k):(i + k+1)])
        S0 = L * np.nanmedian(np.abs(x[(i - k):(i + k+1)] - x0))

        if np.abs(x[i] - x0) > t0 * S0:
            y[i] = np.nan

    y = y[k:-k]

    return y


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
"""

        Helper functions for altimetry algorithms for captoolkit and pyAltim

"""
# Core Python & typing
from typing import Optional, Tuple
import os

# Numerical & data handling
import numpy as np
import pandas as pd
from numpy.linalg import pinv, lstsq

# xarray & raster handling
import xarray as xr
import rasterio
import rioxarray as rxr
from rasterio.transform import from_origin
from rasterio.crs import CRS
from osgeo import gdal, osr
from affine import Affine

# SciPy
from scipy import stats, signal
from scipy.ndimage import map_coordinates
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree, distance
from scipy.stats import median_abs_deviation as mad_std

# HDF5
import h5py

# Numba
from numba import njit, jit, prange

# Projection
import pyproj
from pyproj import Transformer


##########################################################################
# Function for iterative weighted least squares					           #
##########################################################################


def lstsq(
    A: np.ndarray,
    y: np.ndarray,
    w: np.ndarray = None,
    n_iter: int = None,
    n_sigma: float = None,
    ylim: float = None,
    cov: bool = False,
    weight: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative (optionally weighted) least-squares solver with outlier rejection.

    :param A: Design matrix (N x M)
    :param y: Observations (N)
    :param w: Weights for each observation (N), if weight=True
    :param n_iter: Maximum number of iterations for outlier rejection
    :param n_sigma: Sigma threshold for outlier rejection (e.g., 3 for 3-sigma)
    :param ylim: Optional absolute residual threshold for outlier rejection
    :param cov: Whether to return coefficient uncertainties
    :param weight: Whether to apply the weights in fitting
    :return: Tuple (x, e, bad)
        - x: Fitted model coefficients
        - e: Standard error estimates (NaN if cov=False)
        - bad: Boolean array marking outlier positions in original y
    """

    y = y.astype(float)  # Ensure float for NaN handling
    A = A.copy()

    if n_sigma is None:
        n_iter = 1
    elif n_iter is None:
        n_iter = 5

    if weight and w is not None:
        W = np.diag(w)
        A = W @ A
        y = W @ y

    x = np.full(A.shape[1], np.nan)
    e = np.full(A.shape[1], np.nan)

    bad = np.zeros_like(y, dtype=bool)

    for i in range(n_iter):
        good = np.isfinite(y)

        if good.sum() < A.shape[1]:
            break  # Not enough data to solve

        try:
            x = np.linalg.lstsq(A[good], y[good], rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        if n_sigma is not None:
            residuals = y - A @ x
            std = mad_std(residuals[good])

            outlier_mask = np.abs(residuals) > n_sigma * std
            if ylim is not None:
                outlier_mask |= np.abs(residuals) > ylim

            y[outlier_mask] = np.nan

            # If no new outliers are found, break
            if not outlier_mask.any():
                break

    # Final residuals and error estimates
    good = np.isfinite(y)
    bad = ~good

    if cov:
        try:
            residuals = y[good] - A[good] @ x
            s2 = np.var(residuals)
            cov_matrix = s2 * pinv(A[good].T @ A[good])
            e = np.sqrt(np.diag(cov_matrix))
        except Exception:
            pass  # Leave e as NaNs

    return x, e, bad


##########################################################################
# Function for reading tif files without GDAL							       #
##########################################################################
def tiffread(ifile: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float,
                                  float, Optional[str]]:
    """
    Read a TIFF file and extract coordinates, raster data, resolution, and projection.

    Returns x/y grids, z-values, dx/dy resolution, and projection (WKT or None).
    Handles descending coordinates and single-pixel rasters.
    """
    try:
        ds = rxr.open_rasterio(ifile, chunks=None)
    except Exception as e:
        raise IOError(f"Could not open file {ifile}: {e}")

    # Extract 1D coordinate arrays
    x = ds.x.values
    y = ds.y.values
    z = np.squeeze(ds.values)

    # Ensure x ascending
    if x[0] > x[-1]:
        x = x[::-1]
        z = z[:, ::-1]

    # Ensure y ascending (south → north)
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    # Compute resolution (handle single-pixel)
    dx = np.abs(x[1] - x[0]) if x.size > 1 else 1.0
    dy = np.abs(y[1] - y[0]) if y.size > 1 else 1.0

    # Create 2D coordinate grids
    x2d, y2d = np.meshgrid(x, y)

    # Get projection if available
    proj = None
    try:
        if ds.rio.crs:
            proj = ds.rio.crs.to_wkt()
    except Exception:
        proj = None

    return x2d, y2d, z, dx, dy, proj

##########################################################################
# Function for reprojectind coordinates 									   #
##########################################################################
def transform_coord(proj1_epsg, proj2_epsg, x: float, y: float) -> \
        tuple[float, float]:
    """
    Transform coordinates from one projection to another using EPSG codes.

    :param proj1_epsg: EPSG of the source proj (int or str), 4326 or "4326"
    :param proj2_epsg: EPSG of the target proj (int or str), 3031 or "3031"
    :param x: X coordinate in the source projection
    :param y: Y coordinate in the source projection
    :return: Tuple of (x, y) coordinates in the target projection
    """
    src = f"EPSG:{int(proj1_epsg)}"
    tgt = f"EPSG:{int(proj2_epsg)}"

    transformer = Transformer.from_crs(src, tgt, always_xy=True)
    return transformer.transform(x, y)

##########################################################################
# Function for raster interpolation 							   			   #
##########################################################################

def interp2d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
             xi: np.ndarray, yi: np.ndarray, **kwargs) -> np.ndarray:
    """
    Interpolate a 2D raster (z) at points (xi, yi) using map_coordinates.
    Assumes x and y are 2D coordinate grids with ascending coordinates.

    :param x: 2D array of x-coordinates (meshgrid)
    :param y: 2D array of y-coordinates (meshgrid)
    :param z: 2D array of raster values
    :param xi: Interpolation x-points (1D or 2D)
    :param yi: Interpolation y-points (1D or 2D)
    :param kwargs: Passed to map_coordinates (e.g., order=1)
    :return: Interpolated values (same shape as xi/yi)
    """

    xi = np.asarray(xi)
    yi = np.asarray(yi)

    # Extract 1D coordinate vectors
    x1d = x[0, :]
    y1d = y[:, 0]

    nx = x1d.size
    ny = y1d.size

    # Safe denominators (prevent divide by zero)
    x_range = x1d[-1] - x1d[0] if nx > 1 else 1.0
    y_range = y1d[-1] - y1d[0] if ny > 1 else 1.0

    # Convert to pixel coordinates
    xp = (xi - x1d[0]) * (nx - 1) / x_range
    yp = (yi - y1d[0]) * (ny - 1) / y_range

    # Stack coordinates for map_coordinates
    coords = np.vstack([yp.ravel(), xp.ravel()])

    # Perform interpolation
    zi = map_coordinates(z, coords, mode='nearest', **kwargs)

    # Reshape to match input points
    return zi.reshape(xi.shape)

##########################################################################
# Function for 2D binning of data 							   			   #
##########################################################################


@njit
def binning(
        x,
        y,
        xmin=None,
        xmax=None,
        dx=1 / 12.,
        window=3 / 12.,
        median=False):
    """
    Time-series binning with overlapping windows using mean or median.
    Performs 3-sigma clipping for mean if median=False. Handles NaNs.

    Returns:
        xb: bin centers
        yb: binned values
        eb: standard deviation
        nb: number of points
        sb: sum of values
    """

    # Ensure scalar xmin/xmax
    if xmin is None:
        xmin = float(np.nanmin(x))
    if xmax is None:
        xmax = float(np.nanmax(x))

    steps = np.arange(xmin, xmax, dx)
    N = len(steps)

    xb = np.empty(N)
    yb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.zeros(N)
    sb = np.zeros(N)

    for i in range(N):
        t1 = steps[i]
        t2 = t1 + window
        xb[i] = 0.5 * (t1 + t2)

        # Find indices within the window
        idx = np.where((x >= t1) & (x <= t2))[0]
        if idx.size == 0:
            continue

        yv = y[idx]
        # Remove NaNs
        yv = yv[~np.isnan(yv)]
        if yv.size == 0:
            continue

        if median:
            yb[i] = np.nanmedian(yv)
            eb[i] = np.nanstd(yv)
            nb[i] = yv.size
            sb[i] = np.nansum(yv)
        else:
            mu = np.nanmean(yv)
            sigma = np.nanstd(yv)
            if sigma > 0:
                clipped = yv[(yv >= mu - 3 * sigma) & (yv <= mu + 3 * sigma)]
            else:
                clipped = yv

            if clipped.size > 0:
                yb[i] = np.nanmean(clipped)
                eb[i] = np.nanstd(clipped)
                nb[i] = clipped.size
                sb[i] = np.nansum(clipped)

    return xb, yb, eb, nb, sb

##########################################################################
# Function for wrapping longitude to 0-360 degress 						   #
##########################################################################


def wrapTo360(arr):
    """
    Wrapping array of values in degrees to 0-360 degrees

    :param arr: value in degress
    :return warr: wrapped value
    """
    warr = arr.copy()
    positiveInput = (warr > 0)
    warr = np.mod(warr, 360)
    warr[(warr == 0) & positiveInput] = 360
    return warr

##########################################################################
# Function for wrapping longitude to -180 to 180 degress 					   #
##########################################################################


def wrapTo180(arr):
    """
    Wrapping array of values in degrees to -180 to 180 degrees

    :param arr: value in degress
    :return warr: wrapped value
    """
    warr = arr.copy()
    idx = (warr < -180.) | (180. < warr)
    warr[idx] = wrapTo360(warr[idx] + 180.) - 180.
    return warr

##########################################################################
# Function for estimaging std.dev based on Absolute Median Deviation		   #
##########################################################################


def mad_std(x, axis=None):
    """
    Robust std.dev using median absolute deviation. Handles NaN's.

    :param x: data values
    :param axis: target axis for computation
    :return: std.dev (MAD)
    """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

##########################################################################
# Function for estimating standard error based on MAD	   					   #
##########################################################################


def mad_se(x, axis=None):
    """
    Robust Robust standard error (using MAD). Handles NaN's.
    :param x: data values
    :param axis: target axis for computation
    :return: standard error based on MAD
    """
    return mad_std(x, axis=axis) / np.sqrt(np.sum(~np.isnan(x, axis=axis)))

##########################################################################
# Function for filtering data based on MAD	   							   #
##########################################################################


def median_filter(x, n=3):
    """
    Remove values greater than n * MAD (set to NaN)
    :param x: data values
    :param n: integer for editing (3*MAD)
    :return: edited x-values (contains NaN's)
    """
    x[np.abs(x - np.nanmedian(x)) > n_median * mad_std(x)] = np.nan
    return x

##########################################################################
# Function for constructing 2D or 1D grids for e.g interpolation			   #
##########################################################################


def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=True):
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

##########################################################################
# Function for computing bin indicies - like binned_statistics_2d	           #
##########################################################################


@njit
def compute_bin_indices(x, y, xmin, ymin, dx, dy, Ne, Nn):
    """
Compute 1D bin indices (like flattened 2D bins) for each (x,y) point.

Args:
    x, y: arrays of coordinates
    xmin, ymin: min values for reference (usually min of x,y)
    dx, dy: bin sizes
    Ne, Nn: number of bins in x and y directions

Returns:
    indices: array of bin indices (1-based), 0 means out of range
"""

    ix = ((x - xmin) / dx).astype(np.int64)
    iy = ((y - ymin) / dy).astype(np.int64)

    indices = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        if 0 <= ix[i] < Ne and 0 <= iy[i] < Nn:
            indices[i] = iy[i] * Ne + ix[i] + 1  # 1-based bin index
        else:
            indices[i] = 0  # out of range
    return indices

##########################################################################
# Function for removing outliers inside a defined spatial boundiing box	   #
##########################################################################


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

    index = compute_bin_indices(x, y, x.min(), y.min(), dx, dy, Ne, Nn)

    ind = np.unique(index)

    zo = z.copy()

    for i in range(len(ind)):

        idx, = np.where(index == ind[i])

        zb = z[idx]

        i_good = ~np.isnan(zb)

        if len(zb[i_good]) == 0:
            continue

        dh = zb - np.nanmedian(zb)

        foo = np.abs(dh) > n_sigma * np.nanstd(dh)

        zb[foo] = np.nan

        zo[idx] = zb

    return zo
##########################################################################
# Function for peak detection in data (based on MATLAB version)	  	       #
##########################################################################


def findpeaks(
        x,
        mph=None,
        mpd=1,
        threshold=0,
        edge='rising',
        kpsh=False,
        valley=False):
    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        index of the peaks in `x`.

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(
            np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(
            np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak

        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

##########################################################################
# Function for filling non-defined data points	  	                       #
##########################################################################


def fillnans(x, method='nearest'):
    """
    Interpolates and fills NaN data using interpolation

    :param x: vector contaning NaN's
    :param method: choice of interpolation (neareast,linear,cubic etc.)
    :return: vector with filled NaN's
    """
    idx = np.arange(x.shape[0])

    good = np.where(np.isfinite(x))

    f = interp1d(idx[good], x[good],
                 kind=method, fill_value='extrapolate')

    return np.where(np.isfinite(x), x, f(idx))

##########################################################################
# Function for filtering/interpolation time series 	  	                       #
##########################################################################


@jit(nopython=True)
def window_filter(x: np.ndarray, y: np.ndarray, dx: float) -> np.ndarray:
    """
    Apply a moving window median filter to smooth `y` values
    based on the proximity of `x` values.

    For each element in `x`, this function finds all points within
    a distance `dx` and computes the median of the corresponding `y`
    values in that window.

    Parameters:
    ----------
    x : np.ndarray
        1D array of x-coordinates (must be same length as y)
    y : np.ndarray
        1D array of values to filter
    dx : float
        Window half-width: includes all points within ±dx of each x[i]

    Returns:
    -------
    yf : np.ndarray
        Filtered y values (same shape as input)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    yf = y.copy()

    for i in range(len(x)):
        # Compute distance to current x[i]
        distance = np.abs(x - x[i])

        # Find indices within the window
        in_window = np.argwhere(distance < dx).flatten()

        # Compute the median of y values in the window
        yf[i] = np.nanmedian(y[in_window])

    return yf

##########################################################################
# Function for smoothing time series 	  	                                   #
##########################################################################


@jit(nopython=True)
def box_filter1d(x: np.ndarray, k: int) -> np.ndarray:
    """
    Apply a 1D box (mean) filter to smooth input data.

    Each element is replaced with the mean of its surrounding
    values within a window of size 2k+1, centered at the current element.
    NaN values are ignored in the averaging.

    Parameters:
    ----------
    x : np.ndarray
        1D input array to filter. Can contain NaNs.
    k : int
        Half-width of the filter window. Total window size is 2k + 1.

    Returns:
    -------
    y : np.ndarray
        Smoothed array of same shape as input.
    """
    n = len(x)
    y = x.copy()

    for i in range(n):
        # Compute window boundaries (clipped to array bounds)
        start = max(0, i - k)
        end = min(n, i + k + 1)

        # Compute mean ignoring NaNs
        window = x[start:end]
        count = 0
        total = 0.0

        for j in range(len(window)):
            if not np.isnan(window[j]):
                total += window[j]
                count += 1

        if count > 0:
            y[i] = total / count
        else:
            y[i] = np.nan

    return y

##########################################################################
#    Function for spatial filtering using surface model                        #
##########################################################################


def spatial_filter_param(x, y, z, dx, dy, niter=None, sigma=None, thres=None):
    """
    spatial parametric filter using bi-quadratic surface model
    to and edits residuals. should accept lat/lon as coords.
    :param x : x-coord
    :param y : y-coord
    :param z : vector of data to filter
    :param dx: size of box x-direction
    :param dy: size of box y-direction
    :param niter: number of least-squares iterations
    :param sigma: outlier threshold for residuals
    :param thres: absolut threshold for residuals
    :return: vector of filtered values
    """

    # Grid dimensions
    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    # Bin data
    f_bin = stats.binned_statistic_2d(x, y, x, bins=(Ne, Nn))

    # Get bin numbers for the data
    index = f_bin.binnumber

    # Unique indexes
    ind = np.unique(index)

    # Create output
    zo = z.copy()

    # Number of unique index
    for i in range(len(ind)):

        # index for each bin
        idx, = np.where(index == ind[i])

        # Get data
        xb = x[idx]
        yb = y[idx]
        zb = z[idx]

        # Centering of coordinates
        dxb, dyb = xb - xb.mean(), yb - yb.mean()

        # Design matrix
        Ab = np.vstack((np.ones(xb.shape), dxb, dyb,
                        dxb * dyb, dxb**2, dyb**2,
                        dyb * dxb**2, dxb * dyb**2,
                        (dxb**2) * (dyb**2))).T

        # Iterative least-squares fit of data
        ibad = lstsq(Ab.copy(), zb.copy(), n_iter=niter,
                     n_sigma=sigma, ylim=thres)[2]

        # Set to NaN again
        zb[ibad] = np.nan

        # Replace data
        zo[idx] = zb

    return zo

##########################################################################
#    Function for spatial interpoaltion using median                           #
##########################################################################


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


##########################################################################
#    Function for spatial interpoaltion using gaussian kernel                  #
##########################################################################


def interpgaus(x, y, z, s, Xi, Yi, n=10, d=np.inf, a=100.0):
    """
    2D Gaussian kernel interpolation using spatial coordinates and errors.

    Each interpolation point is estimated using the `n` nearest neighbors
    within distance `d`, weighted by observational error and a Gaussian
    distance kernel with correlation length `a`.

    Parameters:
    ----------
    x : array_like
        1D array of x-coordinates (e.g., meters)
    y : array_like
        1D array of y-coordinates (same length as x)
    z : array_like
        1D array of observed values at (x, y)
    s : array_like
        1D array of observational errors (same length as z)
    Xi : array_like
        2D or 1D array of x-coordinates for interpolation
    Yi : array_like
        2D or 1D array of y-coordinates for interpolation
    n : int
        Number of nearest neighbors to consider for each interpolation
    d : float
        Maximum distance allowed (meters); skip if all neighbors are farther
    a : float
        Correlation length (Gaussian kernel parameter, in meters)

    Returns:
    -------
    zi : np.ndarray
        Interpolated values at (Xi, Yi)
    ei : np.ndarray
        Estimated interpolation errors
    ni : np.ndarray
        Number of neighbors used at each interpolation point
    """
    # Flatten interpolation target coordinates
    xi = np.ravel(Xi)
    yi = np.ravel(Yi)

    num_points = len(xi)

    # Initialize outputs
    zi = np.full(num_points, np.nan)
    ei = np.full(num_points, np.nan)
    ni = np.zeros(num_points)

    # Create spatial index tree for neighbor search
    tree = cKDTree(np.column_stack((x, y)))

    # Replace NaNs in errors with 1 if all are NaN
    if np.all(np.isnan(s)):
        s = np.ones_like(z)

    for i in range(num_points):

        # Make into a lsit for kd-tree
        interp_point = (xi[i], yi[i])

        # Find n nearest neighbors
        dxy, idx = tree.query(interp_point, k=n)

        # If n=1, ensure idx and dxy are iterable
        if n == 1:
            idx = [idx]
            dxy = [dxy]

            # Ensure they are arrays
        dxy = np.asarray(dxy)
        idx = np.asarray(idx)

        # Skip if all neighbors are too far
        if np.all(dxy > d):
            continue

        # Extract values and errors at neighbor points
        z_neighbors = z[idx]
        s_neighbors = s[idx]

        # Skip if all z values are NaN
        if np.all(np.isnan(z_neighbors)):
            continue

        # Compute Gaussian weights with error adjustment
        weights = (1.0 / s_neighbors**2) * np.exp(-0.5 * (dxy / a)**2)

        # Stabilize small weights to avoid division by zero
        weights += 1e-6

        # Weighted prediction (ignore NaNs)
        valid = ~np.isnan(z_neighbors)
        z_valid = z_neighbors[valid]
        w_valid = weights[valid]

        # Compute value at prediction point
        zi[i] = np.sum(w_valid * z_valid) / np.sum(w_valid)

        # Weighted residual variance (model error)
        residuals = z_valid - zi[i]
        sigma_r = np.sum(w_valid * residuals**2) / np.sum(w_valid)

        # Mean of sensor errors (measurement error)
        sigma_s = 0.0 if np.all(s == 1) else np.nanmean(s_neighbors[valid])

        # Total estimated error
        ei[i] = np.sqrt(sigma_r**2 + sigma_s**2)

        # Count of neighbors used
        ni[i] = len(z_valid)

    return zi, ei, ni

##########################################################################
#    Function for spatial interpoaltion using collocation/ordinary kriging     #
##########################################################################


def interpkrig(x, y, z, s, Xi, Yi, d, a, n):
    """
    2D Ordinary Kriging interpolation using a second-order Markov model.

    Each interpolation is done using the `n` nearest neighbors within a maximum
    distance `d`, applying a stationary covariance model with correlation
    length `a`. Observational errors are incorporated via a diagonal noise matrix.

    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (in meters)
    y : array_like
        1D array of y-coordinates (in meters)
    z : array_like
        1D array of data values at coordinates (x, y)
    s : array_like
        1D array of observational errors (same length as z)
    Xi : array_like
        Interpolation point x-coordinates (1D or 2D)
    Yi : array_like
        Interpolation point y-coordinates (1D or 2D)
    d : float
        Maximum distance allowed (in meters)
    a : float
        Correlation length scale (in meters)
    n : int
        Number of nearest neighbors to use (minimum = 2)

    Returns
    -------
    zi : np.ndarray
        Interpolated values (1D array)
    ei : np.ndarray
        Estimated interpolation errors
    ni : np.ndarray
        Number of observations used at each interpolation point
    """

    if n < 2:
        raise ValueError(
            "At least 2 neighbors (n > 1) are required for kriging.")

    # Convert all inputs to float32 for performance
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    Xi = np.asarray(Xi, dtype=np.float32)
    Yi = np.asarray(Yi, dtype=np.float32)

    # Flatten target grid
    xi = Xi.ravel()
    yi = Yi.ravel()
    num_points = len(xi)

    # Prepare output arrays
    zi = np.full(num_points, np.nan, dtype=np.float32)
    ei = np.full(num_points, np.nan, dtype=np.float32)
    ni = np.full(num_points, 0, dtype=np.float32)

    # Build spatial index
    tree = cKDTree(np.column_stack((x, y)))

    # Convert distance units from km to m (Markov model uses meters)
    a_m = a * 0.595  # 0.595 factor specific to 2nd-order Markov
    d_m = d

    for i in range(num_points):

        # Query k nearest neighbors
        dxy, idx = tree.query((xi[i], yi[i]), k=n)

        # Ensure output is iterable for k=1
        if n == 1:
            dxy = np.array([dxy])
            idx = np.array([idx])

        # Skip if all neighbors are too far
        if np.min(dxy) > d_m:
            continue

        # Extract local data
        xc, yc = x[idx], y[idx]
        zc, sc = z[idx], s[idx]

        # Check if we have enough data or bad data
        if len(zc) < 2 or np.any(np.isnan(zc)):
            continue

        # Empirical mean and variance
        mean_z = np.median(zc)
        var_z = np.var(zc)

        # Covariance between interpolation point and data (Cxy)
        Cxy = var_z * (1 + dxy / a_m) * np.exp(-dxy / a_m)

        # Compute point-to-point distance (P2P)
        dxx = cdist(np.column_stack((xc, yc)),
                    np.column_stack((xc, yc)), 'euclidean')

        # Compute covariance for P2P distance
        Cxx = var_z * (1 + dxx / a_m) * np.exp(-dxx / a_m)

        # Add observational errors (diagonal noise matrix)
        N = np.eye(len(zc), dtype=np.float32) * (sc ** 2)

        # Solve kriging system: (Cxx + N)^-1 * Cxy using Cholesky decomposition
        try:
            cho = cho_factor(Cxx + N, lower=True, check_finite=False)
            weights = cho_solve(cho, Cxy)
        except np.linalg.LinAlgError:
            continue  # Skip ill-conditioned neighborhoods

        # Ordinary kriging estimate (with bias correction)
        zi[i] = np.dot(weights, zc) + (1 - np.sum(weights)) * mean_z

        # Kriging variance (interpolation error)
        var_error = var_z - np.dot(weights, Cxy)
        ei[i] = np.sqrt(np.abs(var_error)) if var_error >= 0 else 0.0

        # Save number of neighbors used
        ni[i] = len(zc)

    # Convert to float64 for consistency
    return zi.astype(np.float64), ei.astype(np.float64), ni.astype(np.float64)

################################################################################
#    Function for writing tif files with GDAL                                  #
################################################################################

def tiffwrite_gdal(ofile, X, Y, Z, proj, otype='float'):
    """
    Write a raster to a GeoTIFF file using GDAL, handling 2D X/Y grids correctly.

    :param ofile: output file name
    :param X: 2D array of x-coordinates (meshgrid)
    :param Y: 2D array of y-coordinates (meshgrid)
    :param Z: 2D array of values
    :param proj: EPSG projection code (integer)
    :param otype: data type to save ('int' or 'float')
    """

    N, M = Z.shape
    proj = int(proj)

    # Flip Z if Y increases downward (top row must be first)
    if np.all(np.diff(Y[:,0]) > 0):
        Z = np.flipud(Z)

    # Compute geotransform from bounds
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    dx = (xmax - xmin) / (M - 1)
    dy = (ymax - ymin) / (N - 1)
    geotransform = [xmin, dx, 0, ymax, 0, -dy]

    # Select GDAL data type
    if otype == 'int':
        datatype = gdal.GDT_Int32
    elif otype == 'float':
        datatype = gdal.GDT_Float32
    else:
        raise ValueError("otype must be 'int' or 'float'")

    # Create GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(ofile, M, N, 1, datatype)

    # Set projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(proj)
    ds.SetProjection(srs.ExportToWkt())

    # Set geotransform
    ds.SetGeoTransform(geotransform)

    # Handle NoData
    nodata_value = np.nan
    Z_clean = np.where(np.isnan(Z), nodata_value, Z)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.WriteArray(Z_clean)

    ds = None  # Close and write file

################################################################################
#    Function for writing tif files using rasterio                             #
################################################################################


def tiffwrite_rasterio(ofile, X, Y, Z, proj, otype=None, compress="deflate", tiled=True, blocksize=256):
    """
    Write a 2D array to a compressed, tiled GeoTIFF using rasterio.
    Handles 2D X/Y grids and ensures coordinates are correct.

    :param ofile: Output file path
    :param X: 2D x-coordinate array (pixel centers)
    :param Y: 2D y-coordinate array (pixel centers)
    :param Z: 2D data array to write
    :param proj: EPSG code (int) or PROJ string (str)
    :param otype: Output dtype (e.g., 'float32', 'int16'), auto-inferred if None
    :param compress: Compression algorithm ('deflate', 'lzw', 'zstd', or None)
    :param tiled: Enable tiled writing (bool)
    :param blocksize: Tile/block size (typically 128, 256, or 512)
    """

    if X.shape != Y.shape or Z.shape != X.shape:
        raise ValueError("X, Y, and Z must have the same shape.")

    # Flip Z if Y increases downward (rasterio expects top row first)
    if np.all(np.diff(Y[:,0]) > 0):
        Z = np.flipud(Z)

    # Compute GeoTIFF transform from full bounds
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    transform = from_bounds(xmin, ymin, xmax, ymax, Z.shape[1], Z.shape[0])

    # Set CRS
    if isinstance(proj, int):
        crs = CRS.from_epsg(proj)
    elif isinstance(proj, str):
        crs = CRS.from_string(proj)
    else:
        raise TypeError("`proj` must be an integer EPSG code or a PROJ string.")

    # Determine dtype
    dtype = np.dtype(otype).name if otype else Z.dtype.name
    if not rasterio.dtypes.check_dtype(dtype):
        dtype = 'float32'

    # Write GeoTIFF
    with rasterio.open(
        ofile,
        "w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress=compress,
        tiled=tiled,
        blockxsize=blocksize if tiled else None,
        blockysize=blocksize if tiled else None
    ) as dst:
        dst.write(Z.astype(dtype), 1)

################################################################################
#    Function for computing hillshade from a 2D raster                         #
################################################################################

def hillshade(elevation, azimuth=315, altitude=45, cell_size=1, z_factor=1.0):
    """
    Compute hillshade from a 2D elevation array with optional vertical exaggeration.

    Parameters:
    ----------
    elevation : 2D numpy array
        DEM or elevation grid.
    azimuth : float
        Sun azimuth angle in degrees (default: 315° = NW).
    altitude : float
        Sun altitude angle in degrees above horizon (default: 45°).
    cell_size : float
        Spatial resolution of each pixel.
    z_factor : float
        Vertical exaggeration factor (default: 1.0 = no exaggeration).

    Returns:
    -------
    hillshade : 2D numpy array
        Hillshade values scaled from 0–255.
    """
    azimuth_rad = np.deg2rad(360 - azimuth + 90)
    altitude_rad = np.deg2rad(altitude)

    # Compute gradients with vertical exaggeration
    x, y = np.gradient(elevation * z_factor, cell_size, cell_size)

    slope = np.pi/2 - np.arctan(np.sqrt(x**2 + y**2))
    aspect = np.arctan2(-x, y)
    aspect = np.where(aspect < 0, 2*np.pi + aspect, aspect)

    shaded = (
        np.sin(altitude_rad) * np.sin(slope) +
        np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    )

    hillshade = 255 * (shaded - shaded.min()) / (shaded.max() - shaded.min())
    return hillshade.astype(np.uint8)

################################################################################
#    Function for computing spatial covariance                                 #
################################################################################


@njit(parallel=True, fastmath=True)
def spatial_covariance_points(x, y, z, max_dist, lag_step):
    """
    Compute isotropic spatial covariance for irregular point data.

    Parameters
    ----------
    x, y : 1D numpy arrays
        Coordinates of points.
    z : 1D numpy array
        Values at points (same length as x and y).
    max_dist : float
        Maximum distance to consider.
    lag_step : float
        Bin width (distance interval).

    Returns
    -------
    lags : 1D numpy array
        Lag bin centers.
    cov : 1D numpy array
        Covariance for each lag distance.
    counts : 1D numpy array
        Number of pairs contributing to each lag.
    """
    n = len(x)
    mean_z = np.mean(z)
    zc = z - mean_z

    n_lags = int(max_dist // lag_step)
    cov = np.zeros(n_lags)
    counts = np.zeros(n_lags)

    for i in prange(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > max_dist:
                continue
            lag_idx = int(dist // lag_step)
            if lag_idx < n_lags:
                cov[lag_idx] += zc[i] * zc[j]
                counts[lag_idx] += 1

    cov = np.where(counts > 0, cov / counts, np.nan)
    lags = (np.arange(n_lags) + 0.5) * lag_step
    return lags, cov, counts


################################################################################
#    Function for reading GRAVSOFT grids                                       #
################################################################################


def read_gravsoft_grid(filepath):
    """
    Reads a GRAVSOFT 2D grid file and returns the grid as a 2D numpy array,
    along with metadata such as coordinate extents and spacing.

    Parameters:
        filepath (str): Path to the GRAVSOFT grid file.

    Returns:
        grid (np.ndarray): 2D array of grid values [lat x lon]
        latitudes (np.ndarray): 1D array of latitude coordinates (north to south)
        longitudes (np.ndarray): 1D array of longitude coordinates (west to east)
        metadata (dict): Dictionary with lat/lon limits and spacing
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # --- Step 1: Parse header line ---
    header = lines[0].strip().split()
    if len(header) != 6:
        raise ValueError("Header line must contain 6 values.")

    lat1, lat2, lon1, lon2, dlat, dlon = map(float, header)

    # --- Step 2: Calculate grid size ---
    n_lat = int(round((lat2 - lat1) / dlat)) + 1
    n_lon = int(round((lon2 - lon1) / dlon)) + 1

    # --- Step 3: Read all remaining values ---
    data_values = []
    for line in lines[1:]:
        values = line.strip().split()
        data_values.extend(map(float, values))

    if len(data_values) != n_lat * n_lon:
        raise ValueError(f"Expected {n_lat * n_lon} values, got {len(data_values)}")

    # --- Step 4: Reshape into 2D grid ---
    grid = np.array(data_values).reshape((n_lat, n_lon))

    # --- Step 5: Create coordinate arrays ---
    latitudes = np.linspace(lat1, lat2, n_lat)
    longitudes = np.linspace(lon1, lon2, n_lon)

    # --- Step 6: Metadata ---
    metadata = {
        "lat_min": min(lat1, lat2),
        "lat_max": max(lat1, lat2),
        "lon_min": min(lon1, lon2),
        "lon_max": max(lon1, lon2),
        "dlat": dlat,
        "dlon": dlon,
        "n_lat": n_lat,
        "n_lon": n_lon
    }

    return grid, latitudes, longitudes, metadata


    metadata = {
        "lat_north": lat_north,
        "lat_south": lat_south,
        "lon_west": lon_west,
        "lon_east": lon_east,
        "dphi": dphi,
        "dlambda": dlambda,
        "n_lat": n_lat,
        "n_lon": n_lon
    }

    return grid, latitudes, longitudes, metadata


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

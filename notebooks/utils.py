"""

captoolkit - JPL Cryosphere Altimetry Processing Toolkit

Set of tools for processing and integrating satellite and airborne altimetry data.

Credits
Fernando Paolo (paolofer@jpl.nasa.gov) - Core developer
Johan Nilsson (johan.nilsson@jpl.nasa.gov) - Core developer
Alex Gardner (alex.s.gardner@jpl.nasa.gov) - Project PI

Jet Propulsion Laboratory, California Institute of Technology

"""

import numpy as np
import pyproj
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.ndimage import map_coordinates

def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Construct output grid-coordinates."""

    # Setup grid dimensions
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    # Initiate x/y vectors for grid
    x_i = np.linspace(xmin, xmax, num=Ne)
    y_i = np.linspace(ymin, ymax, num=Nn)

    return np.meshgrid(x_i, y_i)


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def medip(x, y, z, Xi, Yi, n, d):
    """2D interpolation using median."""

    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan

    # Create kdtree
    tree = cKDTree(np.c_[x, y])

    # Loop through observations
    for i in range(len(xi)):

        # Find closest number of observations
        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        # Check max distance
        if dxy.min() > d: continue

        # Get parameters
        zc = z[idx]

        # Check for empty solutions
        if len(zc[~np.isnan(zc)]) == 0: continue

        # Predicted value
        zi[i] = np.nanmedian(zc)

    # Return interpolated points
    return zi


def gaussip(x, y, z, s, Xi, Yi, n, d, a):
    """2D interpolation using gaussian weight."""

    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan

    # Create kdtree
    tree = cKDTree(np.c_[x, y])

    # Test sigma vector
    if np.all(np.isnan(s)): s = np.ones(s.shape)

    # Loop through observations
    for i in range(len(xi)):

        # Find closest number of observations
        (dr, idx) = tree.query((xi[i], yi[i]), k=n)

        # Check max distance
        if dr.min() > d: continue

        # Get parameters
        zc = z[idx]
        sc = s[idx]

        # Check for empty solutions
        if len(zc[~np.isnan(zc)]) == 0: continue

        # Compute the weighting factor
        wc = (1./sc**2) * np.exp(-(dr**2)/(2*a**2))

        # Add somethibng small to avoid singularity
        wc += 1e-6

        # Predicted value
        zi[i] = np.nansum(wc*zc) / np.nansum(wc)

    # Return interpolated points
    return zi


def lscip(x, y, z, s, Xi, Yi, d, a, n):
    """2D interpolation using ordinary kriging"""

    # Cast as int!
    n = int(n)

    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    # Create KDTree
    tree = cKDTree(np.c_[x, y])

    # Convert to meters
    a *= 0.595 * 1e3
    d *= 1e3

    # Loop through observations
    for i in range(len(xi)):

        # Find closest number of observations
        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        # Check minimum distance
        if dxy.min() > d: continue

        # Get parameters
        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]

        # Need minimum of two observations
        if len(zc) < 2: continue

        # Estimate local median (robust) and local variance of data
        m0 = np.nanmedian(zc)
        c0 = np.nanvar(zc)

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

        # Number of data used for prediction
        ni[i] = len(zc)

    # Return interpolated values
    return zi, ei, ni


def spatial_filter(x, y, z, dx, dy, sigma=5.0):
    """ Cleaning of spatial data """

    # Grid dimensions
    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    # Bin data
    f_bin = stats.binned_statistic_2d(x, y, z, bins=(Ne,Nn))

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
        zb = z[idx]

        # Make sure we have enough
        if len(zb[~np.isnan(zb)]) == 0:
            continue

        # Set to median of values
        dh = zb - np.nanmedian(zb)

        # Identify outliers
        foo = np.abs(dh) > sigma * np.nanstd(dh)

        # Set to nan-value
        zb[foo] = np.nan

        # Replace data
        zo[idx] = zb

    # Return filtered array
    return zo


def interp2d(x, y, z, xi, yi, **kwargs):
    """Raster to point interpolation."""

    x = np.flipud(x)
    y = np.flipud(y)
    z = np.flipud(z)

    x = x[0,:]
    y = y[:,0]

    nx, ny = x.size, y.size

    x_s, y_s = x[1] - x[0], y[1] - y[0]

    if np.size(xi) == 1 and np.size(yi) > 1:
        xi = xi * np.ones(yi.size)
    elif np.size(yi) == 1 and np.size(xi) > 1:
        yi = yi * np.ones(xi.size)

    xp = (xi - x[0]) * (nx - 1) / (x[-1] - x[0])
    yp = (yi - y[0]) * (ny - 1) / (y[-1] - y[0])

    coord = np.vstack([yp, xp])

    zi = map_coordinates(z, coord, **kwargs)

    return zi

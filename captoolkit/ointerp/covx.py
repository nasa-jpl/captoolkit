#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate empirical spatial covariance from data.

Example:
    python covx.py ~/data/ers1/floating/filt_scat_det/joined_pts_a.h5_ross \
            -v t_year lon lat h_res -t 1995.0 1995.25 -x -160 0 -l 0 15 -d 1

"""

# TODO: Test wich one is faster: covx or covxx!
# NOTE: covxx is MUCH faster!
# TODO: Calculate in parallel for several tiles!
# TODO: Select N tiles at random!

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
from .time import time as tim
import matplotlib.pyplot as plt
from numba import jit, int32, float64
from scipy.spatial.distance import cdist, pdist, squareform


def get_args():
    """ Get command-line arguments. """
    des = "Optimal Interpolation of spatial data"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
        "ifile",
        metavar="ifile",
        type=str,
        nargs="+",
        help="name of input file (in HDF5)",
    )
    parser.add_argument(
        "-o",
        metavar="ofile",
        dest="ofile",
        type=str,
        nargs=1,
        help="name of output file (in ASCII)",
        default=[None],
    )
    parser.add_argument(
        "-v",
        metavar=("t", "x", "y", "z"),
        dest="vnames",
        type=str,
        nargs=4,
        help=("name of time/x/y/obs variables in the HDF5"),
        default=[None],
        required=True,
    )
    parser.add_argument(
        "-t",
        metavar=("t1", "t2"),
        dest="tlim",
        type=float,
        nargs=2,
        help=("time span to subest for covariance calc"),
        default=[None],
    )
    parser.add_argument(
        "-x",
        metavar=("x1", "x2"),
        dest="xlim",
        type=float,
        nargs=2,
        help=("x/lon span to subest for covariance calc"),
        default=[None],
    )
    parser.add_argument(
        "-y",
        metavar=("y1", "y2"),
        dest="ylim",
        type=float,
        nargs=2,
        help=("y/lat span to subest for covariance calc"),
        default=[None],
    )
    parser.add_argument(
        "-l",
        metavar=("dmin", "dmax"),
        dest="dlim",
        type=float,
        nargs=2,
        help=("distance span to calculate sample covariances"),
        default=[0, 10],
    )
    parser.add_argument(
        "-d",
        metavar=("dx"),
        dest="dx",
        type=float,
        nargs=1,
        help=("interval to define a distance class"),
        default=[1],
    )
    parser.add_argument(
        "-p",
        metavar=("epsg_num"),
        dest="proj",
        type=str,
        nargs=1,
        help=("EPSG proj number (AnIS=3031, GrIS=3413)"),
        default=["3031"],
    )
    return parser.parse_args()


def print_args(args):
    print("Input arguments:")
    for arg in vars(args).items():
        print(arg)


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

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


# -----------------------------------------------------------

""" Compiled functions. """


@jit(nopython=True)
def where(dist, lag, tol):
    """ Where 'dist' is between 'lag' +/- 'tol'. """
    l1 = lag - tol
    l2 = lag + tol
    M, N = dist.shape
    ii = np.empty(N * M, int32)
    jj = np.empty(N * M, int32)
    k = 0
    for i in range(M):
        for j in range(N):
            d = dist[i, j]
            if d >= l1 and d < l2:
                ii[k] = i
                jj[k] = j
                k = k + 1
    return (ii[:k], jj[:k])


@jit(nopython=True)
def covxy(x, y):
    """ Covariance => average of pair products. """
    N = x.shape[0]
    psum = 0
    for i in range(N):
        psum = psum + x[i] * y[i]
    return psum / N


@jit(nopython=True)
def distcov(x, y, dists, lags, tol):
    """
    Calculate sample distance convariance.

    x, y: pair of values to compute covariance.
    dists: distances between x and y (condensed mat).
    lags: discrete lag values to estimate sample cov.
    tol: half-width of the lag interval (lag +/- tol).
    """
    # n_pairs = x.shape[0] * y.shape[0]
    n_lags = lags.shape[0]
    cov = np.empty(n_lags, float64)

    for i_lag in range(n_lags):
        lag = lags[i_lag]
        ii, jj = where(dist, lag, tol)

        if len(ii) > 0:
            cov[i_lag] = covxy(x[ii], y[jj])
        else:
            cov[i_lag] = np.nan
    return cov


# ----------------------------------------------------------

""" Helper functions. """


def subset_data(t, x, y, z, tlim=(1995.25, 1995.5), xlim=(-1, 1), ylim=(-1, 1)):
    """ Subset data domain (add NaNs). """
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    (ii,) = np.where(tt & xx & yy)
    return t[ii], x[ii], y[ii], z[ii]


def remove_nans(t, x, y, z):
    """ Mask NaNs and Zeros. """
    (ii,) = np.where((z != 0) & ~np.isnan(z))
    return t[ii], x[ii], y[ii], z[ii]


# Parser argument to variable
args = get_args()

# Read input from terminal
ifile = args.ifile[0]
ofile = args.ofile[0]
tvar = args.vnames[0]
xvar = args.vnames[1]
yvar = args.vnames[2]
zvar = args.vnames[3]
tlim = args.tlim[:]
xlim = args.xlim[:]
ylim = args.ylim[:]
dlim = args.dlim[:]
dx = args.dx[0] * 1e3  # km -> m
proj = args.proj[0]

# Print parameters to screen
print_args(args)


print("reading input file ...")

with h5py.File(ifile, "r") as f:

    step = 1
    time = f[tvar][::step]
    lon = f[xvar][::step]
    lat = f[yvar][::step]
    obs = f[zvar][::step]

    if 0:
        # Remove uncorrected data (this should be done before applying this code?)
        b = f["h_bs"][::step]
        obs[b == 0] = np.nan
        obs[np.isnan(b)] = np.nan

if None in tlim:
    tlim = [np.nanmin(time), np.nanmax(time)]

if None in xlim:
    xlim = [np.nanmin(lon), np.nanmax(lon)]

if None in ylim:
    ylim = [np.nanmin(lat), np.nanmax(lat)]

if ofile is None:
    path, ext = os.path.splitext(ifile)
    ofile = path + ".covx"

dmin, dmax = [d * 1e3 for d in dlim]

# Subset data in space and time
time, lon, lat, obs = subset_data(time, lon, lat, obs, tlim=tlim, xlim=xlim, ylim=ylim)
time, lon, lat, obs = remove_nans(time, lon, lat, obs)

if len(obs) < 10:
    print("not sufficient data points!")
    sys.exit()

# TODO: Think whether we want this before of after subsetting?!
# Convert to stereo coordinates
x, y = transform_coord(4326, proj, lon, lat)

# Plot (for testing)
if 0:
    plt.scatter(
        x, y, c=obs, s=1, rasterized=True, vmin=-np.nanstd(obs), vmax=np.nanstd(obs)
    )
    plt.show()
    sys.exit()

""" Calculate sample covariance. """

print("calculating pdist ...")
t0 = tim()

# Compute all distances between data points
X = np.column_stack((x, y))
dist = cdist(X, X, metric="euclidean")  # -> distance mat

print("time:", tim() - t0)

# Distance lags
lag = np.arange(dmin, dmax + dx, dx)

# Half width of distance interval
tol = dx / 2.0

# Center data
# obs -= np.nanmean(obs)  ##FIXME: This might be a bad idea!

print("calculating discov ...")
t0 = tim()

cov = distcov(obs, obs, dist, lag, tol)

print("time:", tim() - t0)

""" Save sample covariances. """

if 1:
    np.savetxt(ofile, np.column_stack((lag, cov)), fmt="%.6f")
    print("file out ->", ofile)

# Plot (for testing)
if 1:
    plt.plot(lag, cov, "o")
    plt.show()

print("done.")

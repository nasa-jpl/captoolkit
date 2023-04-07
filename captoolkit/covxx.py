#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate data covariance as a function of spatial distance.

The calculation of pair-wise distances on large datasets is a computing
intensive task. This process is sped up by using index-mapping tricks
from condensed-matrix to distance-matrix and by compiling (w/JIT)
computing-intensive functions.

Note:
    This code (covxx.py) is much faster than the original (covx.py).

Example:
    covxx.py ~/data/ers1/floating/filt_scat_det/joined_pts_a.h5_ross \
        -v t_year lon lat h_res -t 1995.0 1995.25 -x -160 0 -l 0 15 -d 1

"""
import argparse
import os
import sys
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from numba import float64, int32, jit
from scipy.spatial.distance import cdist, pdist

from .time import time as tim


def get_args():
    """Get command-line arguments."""
    des = "Estimate spatial covariance."
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
        "ifile",
        metavar="ifile",
        type=str,
        nargs="+",
        help="name of input file(s) in HDF5",
    )
    parser.add_argument(
        "-o",
        metavar="ofile",
        dest="ofile",
        type=str,
        nargs=1,
        help="name of output ASCII file",
        default=[None],
    )
    parser.add_argument(
        "-e",
        metavar="ext",
        dest="ext",
        type=str,
        nargs=1,
        help="extension of output ASCII file (.covx)",
        default=[".covx"],
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
        help=("x-coord span to subest for covariance calc"),
        default=[None],
    )
    parser.add_argument(
        "-y",
        metavar=("y1", "y2"),
        dest="ylim",
        type=float,
        nargs=2,
        help=("y-coord span to subest for covariance calc"),
        default=[None],
    )
    parser.add_argument(
        "-l",
        metavar=("dmin", "dmax"),
        dest="dlim",
        type=float,
        nargs=2,
        help=("distance span to calculate sample covariances (km)"),
        default=[0, 5],
    )
    parser.add_argument(
        "-d",
        metavar=("dx"),
        dest="dx",
        type=float,
        nargs=1,
        help=("interval to define a distance class (km)"),
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
    parser.add_argument(
        "-m",
        metavar=("n_rand"),
        dest="nrand",
        type=int,
        nargs=1,
        help=("select n files at random from input list"),
        default=[None],
    )
    parser.add_argument(
        "-n",
        metavar=("n_jobs"),
        dest="njobs",
        type=int,
        nargs=1,
        help="for parallel processing of multiple files",
        default=[1],
    )

    return parser.parse_args()


def print_args(args):
    print("Input arguments:")

    for arg in vars(args).items():
        print(arg)


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


# -----------------------------------------------------------

""" Compiled functions. """


@jit(nopython=True)
def i_index(n, k):
    return int(n - (4.0 * n ** 2 - 4 * n - 8 * k + 1) ** 0.5 / 2 - 0.5)


@jit(nopython=True)
def j_index(n, k, i):
    return int(k + i * (i + 3 - 2 * n) / 2 + 1)


@jit(nopython=True)
def get_pair_indices(n, kk):
    """Map k-index of condensed mat to i,j-indices of distance mat.

    n : number of points (dist_mat.shape[0]).
    kk : indices of elements in condensed matrix.
    """
    N = kk.shape[0]
    ii = np.empty(N, int32)
    jj = np.empty(N, int32)

    for m in range(N):
        k = kk[m]
        i = i_index(n, k)
        j = j_index(n, k, i)
        ii[m] = i
        jj[m] = j

    return ii, jj


@jit(nopython=True)
def get_dist_indices(dist, lag, tol):
    """Find indices of distances within lag +/- tol."""
    l1 = lag - tol
    l2 = lag + tol
    N = dist.shape[0]
    idx = np.empty(N, int32)
    k = 0

    for i in range(N):
        d = dist[i]

        if d >= l1 and d < l2:
            idx[k] = i
            k = k + 1

    return idx[:k]


@jit(nopython=True)
def cov_xy(x, y):
    """Covariance of x,y -> average of pair products."""
    N = x.shape[0]
    psum = 0

    for i in range(N):
        psum = psum + x[i] * y[i]

    return psum / N


@jit(nopython=True)
def dist_cov(x, y, dists, lags, tol):
    """Calculate sample distance convariance.

    x, y: pair of variables to compute covariance from.
    dists: distances between x and y (condensed matrix).
    lags: discrete lag values to estimate sample covariances.
    tol: margin to determine width of lag interval: width = lag +/- tol.
    """
    n_pairs = x.shape[0]
    n_lags = lags.shape[0]
    cov = np.empty(n_lags, float64)

    for i_lag in range(n_lags):
        lag = lags[i_lag]
        kk = get_dist_indices(dists, lag, tol)

        if len(kk) == 0:
            cov[i_lag] = np.nan

            continue

        ii, jj = get_pair_indices(n_pairs, kk)
        cov[i_lag] = cov_xy(x[ii], y[jj])

    return cov


# ----------------------------------------------------------

""" Helper functions. """


def list_files(string, sort_by=False):
    """Expand (sub)strings into list of file names."""
    path_list = string.split()  # list of substrings (paths)
    file_sets = [glob(p) for p in path_list]  # path list -> lists of files
    # Sort (in place) each set by key (keep set order as given)

    if sort_by:
        [sort_by_key(file_set, sort_by) for file_set in file_sets]
    # Generate flat list with file names

    return [f for file_set in file_sets for f in file_set]


def sort_by_key(files, key="tile"):
    """Sort list by 'key_num' for given 'key'."""
    natkey = lambda s: int(re.findall(key + "_\d+", s)[0].split("_")[-1])

    return files.sort(key=natkey)  # sort inplace


def subset_data(t, x, y, z, tlim=(0, 1), xlim=(-1, 1), ylim=(-1, 1)):
    """Subset data domain (add NaNs)."""
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    (ii,) = np.where(tt & xx & yy)

    return t[ii], x[ii], y[ii], z[ii]


def remove_invalid(t, x, y, z):
    """Remove NaNs and Zeros."""
    (ii,) = np.where((z != 0) & ~np.isnan(z))

    return t[ii], x[ii], y[ii], z[ii]


# ----------------------------------------------------------

# Parser argument to variable
args = get_args()

# Read input from terminal
ifile = args.ifile[:]
ofile_ = args.ofile[0]
ext = args.ext[0]
tvar = args.vnames[0]
xvar = args.vnames[1]
yvar = args.vnames[2]
zvar = args.vnames[3]
tlim_ = args.tlim[:]
xlim_ = args.xlim[:]
ylim_ = args.ylim[:]
dmin = args.dlim[0] * 1e3  # km -> m
dmax = args.dlim[1] * 1e3  # km -> m
dx = args.dx[0] * 1e3  # km -> m
proj = args.proj[0]
nrand = args.nrand[0]
njobs = args.njobs[0]

# Print parameters to screen
print_args(args)

if len(ifile) == 1:
    ifile = list_files(ifile[0])

if nrand:
    ifile = np.random.choice(ifile, nrand)

print("files to process:", len(ifile))


def main(ifile):

    ofile, tlim, xlim, ylim = ofile_, tlim_, xlim_, ylim_

    with h5py.File(ifile, "r") as f:
        step = 1
        time = f[tvar][::step]
        lon = f[xvar][::step]
        lat = f[yvar][::step]
        obs = f[zvar][::step]

    if None in tlim:
        tlim = [np.nanmin(time), np.nanmax(time)]

    if None in xlim:
        xlim = [np.nanmin(lon), np.nanmax(lon)]

    if None in ylim:
        ylim = [np.nanmin(lat), np.nanmax(lat)]

    if ofile is None:
        ofile = os.path.splitext(ifile)[0] + ext

    # Subset data in space and time
    data = [time, lon, lat, obs]
    data = subset_data(*data, tlim=tlim, xlim=xlim, ylim=ylim)
    time, lon, lat, obs = remove_invalid(*data)

    if len(obs) < 15:
        print("not sufficient data points!")

        return

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

    try:
        dist = pdist(X, metric="euclidean")  # -> condensed dist_mat (vec)
    except:
        print("MemoryError: skipping file ->", ifile)  # insufficient data?

        return

    print("time:", tim() - t0)

    lag = np.arange(dmin, dmax + dx, dx)  # distance lags
    tol = dx / 2.0  # half width of distance interval
    obs -= np.nanmean(obs)  # center data

    print("calculating discov ...")
    t0 = tim()

    cov = dist_cov(obs, obs, dist, lag, tol)

    print("time:", tim() - t0)

    """ Save sample covariances. """

    if 1:
        np.savetxt(ofile, np.column_stack((lag, cov)), fmt="%.6f")
        print("file out ->", ofile)

    if 0:
        # Plot (for testing)
        plt.plot(lag, cov, "o")
        plt.show()


# Run main program

if njobs == 1:
    print("running serial code ...")
    [main(f) for f in ifile]
else:
    print("running parallel code (%d jobs) ..." % njobs)
    from joblib import Parallel, delayed

    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in ifile)

print("done.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate empirical temporal covariance from data.

Example:
    covt.py ~/data/ers1/floating/filt_scat_det/joined_pts_a.h5_ross -v t_year lon lat h_res -x -160 0 -l 0 3 -d 0.1 -r 2.5

"""

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt
from .time import time as tim
from scipy.spatial import cKDTree


def get_args():
    """ Get command-line arguments. """

    des = 'Optimal Interpolation of spatial data'
    parser = argparse.ArgumentParser(description=des)

    parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='name of input file (in HDF5)')

    parser.add_argument(
            '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
            help='name of output file (in ASCII)',
            default=[None])

    parser.add_argument(
            '-v', metavar=('t', 'x','y', 'z'),
            dest='vnames', type=str, nargs=4,
            help=('name of time/x/y/obs variables in the HDF5'),
            default=[None], required=True)

    parser.add_argument(
            '-x', metavar=('x1', 'x2'), dest='xlim', type=float, nargs=2,
            help=('x/lon span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-y', metavar=('y1', 'y2'), dest='ylim', type=float, nargs=2,
            help=('y/lat span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-t', metavar=('t1', 't2'), dest='tlim', type=float, nargs=2,
            help=('time span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-l', metavar=('dmin', 'dmax'), dest='dlim', type=float, nargs=2,
            help=('distance span to calculate sample covariances'),
            default=[0,10],)

    parser.add_argument(
            '-d', metavar=('dt'), dest='dt', type=float, nargs=1,
            help=('interval to define a distance class'),
            default=[1],)

    parser.add_argument(
            '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
            help=('search radius for time series extraction (km)'),
            default=[1],)

    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

    return parser.parse_args()


def print_args(args):
    print('Input arguments:')
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
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)

#-----------------------------------------------------------

""" Main functions. """

def autocorr(y):
    """ Calculate autocorrelation function. """
    result = np.correlate(y, y, mode='full')
    return result[result.size/2:]


def timecov(t, y):
    """ Calculate sample time covariance. """
    ii = np.argsort(t)
    t_ = t[ii]
    y_ = y[ii]
    corr = autocorr(y_)
    lags = t_ - t_[0]
    var = np.var(y_)
    cov = corr * var
    return lags, cov 


def lag_intervals(lag, cov, lags, tol):
    """ Convert irregularly lagged covs to evenly spaced lags. """
    covs = np.full_like(lags, np.nan)
    for k, l in enumerate(lags):
        ii, = np.where( (lag >= l-tol) & (lag < l+tol) )
        if len(ii) > 0:
            covs[k] = np.nanmean(cov[ii])
    return covs

#-----------------------------------------------------------

""" Helper functions. """

def subset_data(t, x, y, z,
        tlim=(1995.25, 1995.5), xlim=(-1, 1), ylim=(-1, 1)):
    """ Subset data domain (add NaNs). """
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    ii, = np.where(tt & xx & yy)
    return t[ii], x[ii], y[ii], z[ii]


def remove_invalid(t, x, y, z):
    """ Mask NaNs and Zeros. """
    ii, = np.where((z != 0) & ~np.isnan(z))
    return t[ii], x[ii], y[ii], z[ii]


def detrend(x, y, poly=0):
    """
    Remove polynomial trend from time series data.

    Return:
        y_resid, y_trend: residuals and trend.
    """
    ii, = np.where(~np.isnan(x) & ~np.isnan(y))
    x_, y_ = x[ii], y[ii]

    # Detrend using OLS polynomial fit
    x_mean = np.nanmean(x)
    p = np.polyfit(x_ - x_mean, y_, poly)
    y_trend = np.polyval(p, x - x_mean)

    if np.isnan(y_trend).all():
        y_trend = np.zeros_like(x)

    return y-y_trend, y_trend


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
dt = args.dt[0]
radius = args.radius[0] * 1e3  # km -> m
proj = args.proj[0]

# Print parameters to screen
print_args(args)


print("reading input file ...")

with h5py.File(ifile, 'r') as f:

    step = 2
    time = f[tvar][::step]
    lon = f[xvar][::step]
    lat = f[yvar][::step]
    obs = f[zvar][::step]

    if 1:
        # Remove uncorrected data (this should be done before applying this code?)
        b = f['h_bs'][::step] 
        obs[b==0] = np.nan
        obs[np.isnan(b)] = np.nan

if None in tlim:
    tlim = [np.nanmin(time), np.nanmax(time)]

if None in xlim:
    xlim = [np.nanmin(lon), np.nanmax(lon)]

if None in ylim:
    ylim = [np.nanmin(lat), np.nanmax(lat)]

if ofile is None:
    path, ext = os.path.splitext(ifile)
    ofile = path + '.covt'

dmin, dmax = dlim

# Subset data in space and time
time, lon, lat, obs = subset_data(time, lon, lat, obs, tlim=tlim, xlim=xlim, ylim=ylim)
time, lon, lat, obs = remove_invalid(time, lon, lat, obs)

if len(obs) < 100:
    print('not sufficient data points!')
    sys.exit()

# Convert to stereo coordinates
x, y = transform_coord(4326, proj, lon, lat)

# Construct cKDTree - points
Tree = cKDTree(list(zip(x, y)))

x0 = np.nanmean(x)
y0 = np.nanmean(y)

# Get indexes from Tree 
idx = Tree.query_ball_point((x0, y0), radius)

if len(idx) < 100:
    print('time series not long enough!')
    sys.exit()

# Get data within search radius (time series)
tp = time[idx]
xp = x[idx]
yp = y[idx]
zp = obs[idx]

""" Calculate sample covariance. """

# Half width of distance interval
tol = dt/2.

# Get anomalies
zp, _ = detrend(tp, zp, poly=2) 

# Filter outliers
#zp = median_filter(zp, n_median=3)

# Center data
zp -= np.nanmean(zp)

# Compute time-lag covariance
lag, cov = timecov(tp, zp)

# Irregular lags -> evenly-spaced lags
lags = np.arange(dmin, dmax+dt, dt)

covs = lag_intervals(lag, cov, lags, tol)

""" Save sample covariances. """

if 0:
    #np.savetxt(ofile, np.column_stack((lag, cov)), fmt='%.6f')
    np.savetxt(ofile, np.column_stack((lags, covs)), fmt='%.6f')
    print('file out ->', ofile)

# Plot (for testing)
if 1:
    plt.figure()
    plt.plot(lag, cov, '.')
    plt.plot(lags, covs, 'o')
    plt.title('Covariance x Lag')

    plt.figure()
    y_corr = autocorr(zp)
    plt.plot(y_corr, '.')
    plt.title('Autocorrelation')

    plt.show()

print('done.')

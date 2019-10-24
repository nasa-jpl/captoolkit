#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:29:20 2017

@author: nilssonj
"""
import warnings

warnings.filterwarnings("ignore")
import sys
import netCDF4
import pyproj
import h5py
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shapefile
import argparse
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates
from scipy.ndimage import generic_filter
import pandas as pd
from progress.bar import Bar

def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def rlsq(x, y, n=1, i=5):
    """
        Robust version of polyfit, no weights available
    """

    # Test solution
    if len(x[~np.isnan(y)]) <= (n + 1):

        if n == 0:

            # Set output to Nan
            p = np.nan
            s = np.nan

        else:
            # Set output to NaN
            p = np.zeros((1, n)) * np.nan
            s = np.nan

        return p, s

    # Test model order
    if n == 0:

        # Mean only
        A = np.ones(len(x))

    else:

        # Empty array
        A = np.empty((0, len(x)))

    # Create counter
    i = 0

    # Construct design matrix
    while i <= n:

        # Stack coefficients
        A = np.vstack((A, x ** i))

        # Update counter
        i += 1

    # Solve system of equations
    try:

        # Robust least squares fit
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=i)

        # polynomial coefficients
        p = fit.params

        # RMS of the residuals
        s = mad_std(fit.resid)

    except:

        # Solution did't work
        print('Solution invalid!')
        return

    return p[::-1], s


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


# Description of program
description = ('Program for post-adjustment of Geosat data')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    'ifile', metavar='ifile', type=str, nargs=1,
    help='input file to process')

parser.add_argument(
    '-b', metavar='bias', dest='bias', type=int, nargs=1,
    help=('bias adjustment time in decimal years'),
    default=[None], )

parser.add_argument(
    '-r', metavar=('tref'), dest='tref', type=float, nargs=1,
    help='reference time of fit',
    default=[None], )

parser.add_argument(
    '-v', metavar=('t', 'Z'), dest='vnames',
    type=str, nargs=2, help=('name of t/z/ variables in the HDF5'),
    default=[None], required=True)

# Add arguments to container
args = parser.parse_args()

# Input and output file name for now
ifile = args.ifile[0]
tbias = args.bias[0]
tref = args.tref[0]
names = args.vnames[:]

# Print arguments to screen
print('parameters:')
for p in vars(args).items(): print(p)

# Get variable names
tvar, zvar = names

# Open file to plot
with h5py.File(ifile, 'r') as fx:

    # Read in needed variables
    X = fx['X'][:]
    Y = fx['Y'][:]
    Zts = fx[zvar][:]
    time = fx[tvar][:]

# Reference time
if tref is None:

    # Set to mean time
    tref = time.mean()

# Reference time
dt = time - tref

# Get shape of cube
(K, N, M) = Zts.shape

# Progress bar for spatial filtering
bar = Bar('Processing', max=N*M, suffix='%(index)d/%(max)d - %(percent).1f%%')

# Get flagged values
Inan = Zts == -9999

# Set those values to NaN
Zts[Inan] = np.nan

# Transform coordinates
Lon, Lat = transform_coord('3031', '4326', X, Y)

# Loop trough cube
for i in range(N):
    for j in range(M):

        # Get time series
        z = Zts[:, i, j]
        t = time.copy()
        zo = z.copy()

        # Skip if we don't have data
        if len(time[time < tbias]) == 0 or np.all(np.isnan(z)) or np.abs(Lat[i,j]) > 72:
            next(bar)
            continue

        # Create offset
        ci = np.zeros(time.shape)

        # Set Geosat values to one
        ci[time < tbias] = 1.

        # Setup design matrix - trend and acceleration
        A = np.vstack((np.ones(t.shape), t, 0.5 * t ** 2, np.cos(2 * np.pi * t), \
                           np.sin(2 * np.pi * t), ci)).T

        # Try to solve system of equations
        try:

            # Solve for model parameters
            p = sm.RLM(z, A, missing='drop').fit(maxiter=5).params

        except:
            next(bar)
            continue

        # Compute residuals
        dh = z - np.dot(A,p)

        # Get time spans
        igeo = (t > 1985) & (t < 1990) & ~np.isnan(z)
        iers = (t > 1990) & (t < 1996) & ~np.isnan(z)

        # Apply correction
        z -= ci * p[-1]

        # Fit model
        try:

            # Fit model to values
            p_geo, s_geo = rlsq(t[igeo], dh[igeo], 1)
            p_ers, s_ers = rlsq(t[iers], dh[iers], 1)

            b_geo = np.polyval(p_geo, t[igeo][-1])
            b_ers = np.polyval(p_ers, t[igeo][-1])

            # Apply correction
            z -= (b_geo - b_ers)

        except:

            # Did work
            pass

        # Apply the correction
        Zts[:, i, j] = z

        # Increase counter
        next(bar)

    # Increase counter
    next(bar)

# Add back the -9999 to cube
Zts[Inan] = -9999

# Save vars to file again
f = netCDF4.Dataset(ifile, 'r+')
f.variables[zvar][:] = Zts
f.close()

# Filtering finished
bar.finish()

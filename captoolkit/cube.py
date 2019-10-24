#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:27:10 2017

@author: nilssonj
"""
import warnings
warnings.filterwarnings("ignore")

import re
import sys
import glob
import h5py
import numpy as np
import netCDF4
import pyproj
import argparse
import matplotlib.pyplot as plt
from numba import jit
from gdalconst import *
from osgeo import gdal, osr
from datetime import datetime
from numpy.core import multiarray
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates
from gdalconst import *
from osgeo import gdal, osr
from scipy.interpolate import griddata

def lscip(x, y, z, s, h, Xi, Yi, Hi, d, n, ad, ah):
    """2D interpolation using ordinary kriging"""

    # Cast as int!
    n = int(n)

    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()
    hi = Hi.ravel()

    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    # Create KDTree
    tree = cKDTree(np.c_[x, y])

    # Convert to meters
    ad *= 1e3
    ah *= 1e3
    d  *= 1e3

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
        hc = h[idx]

        # Need minimum of two observations
        if len(zc) < 2: continue

        # Estimate local median (robust) and local variance of data
        m0 = np.nanmedian(zc)
        c0 = np.nanvar(zc)

        # Center the height to grid
        dh = np.abs(hc - hi[i])

        # Covariance function for Dxy
        Cxy = c0 * np.exp(-dxy / ad) * np.exp(-dh / ah)

        # Compute pair-wise distance
        dxx = cdist(np.c_[xc, yc], np.c_[xc, yc], "euclidean")

        # Create array
        dhh = np.zeros((len(hc), len(hc)))

        # Create pair-wise height
        for ki in range(len(hc)):
            for kj in range(len(hc)):
                dhh[ki, kj] = hc[ki] - hc[kj]

        # Take absolute value of time
        dhh = np.abs(dhh)

        # Covariance function Dxx
        Cxx = c0 * np.exp(-dxx / ad) * np.exp(-dhh / ah)

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


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + proj1)
    proj2 = pyproj.Proj("+init=EPSG:" + proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def geotiffread(ifile):
    """ Read geotiff rasters."""

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
        
    (Xp, Yp) = np.meshgrid(Xp,Yp)
        
    X = trans[0] + (Xp+0.5)*trans[1] + (Yp+0.5)*trans[2]
    Y = trans[3] + (Xp+0.5)*trans[4] + (Yp+0.5)*trans[5]
    
    band = file.GetRasterBand(1)
    
    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj, trans


def interp2d(x, y, z, xi, yi, **kwargs):
    """Raster to point interpolation."""

    x = np.flipud(x)
    y = np.flipud(y)
    z = np.flipud(z)

    x = x[0, :]
    y = y[:, 0]

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


#@jit(nopython=True)
def clean(image, kernel, thres=3.0):
    """
        3-sigma filter estimated from local
        variability.
    """
    # Get index of center coordinate
    ki = int(np.floor(0.5 * kernel))

    # Shape of new array
    (n, m) = image.shape

    # Loop trough raster
    for i in range(ki, n - ki, 1):
        for j in range(ki, m - ki, 1):

            # Get window
            img = image[i - ki:i + ki + 1, j - ki:j + ki + 1].copy()

            # Check for all NaN's
            if np.all(np.isnan(img)): continue

            # Set center to NaN
            img[ki, ki] = np.nan

            # Value at kernel center
            img_val = image[i, j]

            # Median value and standard deviation of kernel
            m_l = np.nanmedian(img)
            s_l = 1.4826 * np.nanmedian(np.abs(img - m_l))

            # Detect outliers
            if np.abs(img_val - m_l) > (thres * s_l):

                # Set to NaN
                img[img == img_val] = np.nan

                # Set value to NaN
                image[i, j] = np.nanmean(img)

    # Return filtered image
    return image

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


# Description of program
description = ('Create a 3D cube of a series of tif-files')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        '-i', metavar='ipath', dest='tsfiles', type=str, nargs='+',
        help='tif file(s) for time series values to process',
        required=True)

parser.add_argument(
        '-e', metavar='epath', dest='esfiles', type=str, nargs='+',
        help='tif file(s) for error to process',
        required=True)

parser.add_argument(
        '-n', metavar='npath', dest='nsfiles', type=str, nargs='+',
        help='tif file(s) for nobs files',
        required=True)

parser.add_argument(
        '-f', metavar='fpath', dest='flgfiles', type=str, nargs='+',
        help='tif file(s) for fill flag files',
        default=None)

parser.add_argument(
        '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
        help='name of output netcdf file',)

parser.add_argument(
        '-t', metavar=('t_min','t_max'), dest='tspan', type=float, nargs=2,
        help=('time span of cube in decimal years'),
        default=[-9999,9999],)

parser.add_argument(
        '-r', metavar=('rfile'), dest='rfile', type=str, nargs=1,
        help=('name of file containing rates'),
        default=[None],)

parser.add_argument(
        '-d', metavar=('dfile'), dest='dfile', type=str, nargs=1,
        help=('DEM file'),
        default=[None],)

parser.add_argument(
        '-p', metavar=('proj'), dest='proj', type=str, nargs=1,
        help=('projection'),
        default=['3031'],)

# Parser argument to variable
args = parser.parse_args()

# Extract filename
tsfiles = args.tsfiles
esfiles = args.esfiles
nsfiles = args.nsfiles
flfiles = args.flgfiles
ofile   = args.ofile[0]
tmin    = args.tspan[0]
tmax    = args.tspan[1]
rfile   = args.rfile[0]
proj    = args.proj[0]
fdem     = args.dfile[0]

# Start timing of script
startTime = datetime.now()

# Loop trough the files
for i in range(len(tsfiles)):

    # Read files from memory
    (Xt, Yt, Zt, dx, dy, Proj, Transform) = geotiffread(tsfiles[i])
    (Xe, Ye, Ze, dx, dy, Proj, Transform) = geotiffread(esfiles[i])
    (Xn, Yn, Zn, dx, dy, Proj, Transform) = geotiffread(nsfiles[i])

    # Test if we have fill flag
    if flfiles is not None:

        # Read file from memory
        (Xf, Yf, Zf, dx, dy, Proj, Transform) = geotiffread(flfiles[i])

    # Create data cube from first file
    if i == 0:

        # Get grid shape
        (N, M) = Xt.shape

        # Data output cubes
        Zts = np.zeros((len(tsfiles), N, M)) * np.nan
        Zes = np.zeros((len(tsfiles), N, M)) * np.nan
        Zns = np.zeros((len(tsfiles), N, M)) * np.nan

        # Test if we have fill flag
        if flfiles is not None:

            # Create output cube
            Zfl = np.zeros((len(tsfiles), N, M)) * np.nan

        # Create time vector
        time = np.zeros(len(tsfiles)) * np.nan

        # Convert time string to float
        time[i] = float(re.findall("\d+\.\d+", tsfiles[i])[0])

        # Save data to output
        Zts[i, :, :] = Zt
        Zes[i, :, :] = Ze
        Zns[i, :, :] = Zn

        # Test if we have fill flags
        if flfiles is not None:

            # Save data
            Zfl[i, :, :] = Zf

    else:

        # Convert time string to float
        time[i] = float(re.findall("\d+\.\d+", tsfiles[i])[0])

        # Save data to output
        Zts[i, :, :] = Zt
        Zes[i, :, :] = Ze
        Zns[i, :, :] = Zn

        # Test if we have fill flags
        if flfiles is not None:

            # Save data
            Zfl[i, :, :] = Zf

"""
# Set -9999 to NaN
Zts[Zts ==- 9999] = np.nan

# Open file to plot
with h5py.File(fdem, 'r') as fx:

    # Read in needed variables
    Xd = fx['X'][:]
    Yd = fx['Y'][:]
    Zd = fx['Z'][:]

# Interpolate and get DEM values
Zh = np.flipud(interp2d(Xd, Yd, Zd, Xt.ravel(), Yt.ravel(), order=1).reshape(Xt.shape))

print 'Loading rate file ...'

# Load the rate file
ds = np.loadtxt('trend.txt')

# Get needed variables
lon, lat, rate, tref = ds[:, 0], ds[:, 1], ds[:, 2], ds[:, -1]

# Transform points
xp, yp = transform_coord('4326', proj, lon, lat)

# Get height values at points
hp = interp2d(Xd, Yd, Zd, xp, yp, order=1)

# Small outlier editing
rate[np.abs(rate) > 100] = np.nan
rate[rate > 1.0] = np.nan

# Set data to NaN
inan = ~np.isnan(rate)

print 'Interpolating rates ...'

from scipy.interpolate import griddata

Zt = griddata((xp[inan], yp[inan]), rate[inan],(Xt, Yt), 'linear', fill_value=np.nan)

Zt = clean(Zt.copy(), kernel=9, thres=5.0)

# Interpolate rates to grid
#Zt = lscip(xp[inan], yp[inan], rate[inan], np.ones(rate[inan].shape) * 0.05, hp[inan], Xt, Yt, Zh, n=25, d=25, ah=0.5, ad=10)[0].reshape(Xt.shape)

Lon, Lat = transform_coord('3031', '4326', Xt, Yt)
Zi = Zt.copy()
Zi[np.isnan(Zts[-1,::])] = np.nan
Zi[np.abs(Lat)>81.5]=np.nan

print 'volume change', np.around(np.nansum(Zi) * 10e3 * 10e3 * 1e-9, 1)

# Reference time
dt = time - tref[0]

# Get shape of cube
(K, N, M) = Zts.shape

print 'Adding interpolated rates back ...'

# Loop trough cube
for i in xrange(N):
    for j in xrange(M):
        # Add trend back to array
        Zts[:, i, j] += Zt[i, j] * dt
"""

# Create output file
foo = netCDF4.Dataset(ofile, 'w', clobber=True)

# Spatial coordinates
x, y = Xt[0,:], Yt[:,0]

# Select time span
i_t = (time > tmin) & (time < tmax)

# Cube Dimensions 
nx, ny, nz = len(x), len(y), len(time[i_t])

# Create dimensions
foo.createDimension('x', nx)
foo.createDimension('y', ny)
foo.createDimension('time', nz)

# Create spatial variables
xi = foo.createVariable('x', 'd', ('x',),    zlib=True)
yi = foo.createVariable('y', 'd', ('y',),    zlib=True)
Xi = foo.createVariable('X', 'd', ('y','x'), zlib=True)
Yi = foo.createVariable('Y', 'd', ('y','x'), zlib=True)
ti = foo.createVariable('time', 'd', ('time',), zlib=True)

print('Grid Extent: ', x.min(), x.max(), y.min(), y.max(), ('m'))

# Populate arrays
xi[:], yi[:], ti[:] = x, y, time[i_t]

# Populate arrays
Xi[:], Yi[:] = Xt, Yt

# Create output arrays
Z = foo.createVariable('dH_height', 'd', ('time','y','x'), zlib=True)
E = foo.createVariable('dE_height', 'd', ('time','y','x'), zlib=True)
N = foo.createVariable('Ni_height', 'd', ('time','y','x'), zlib=True)

# Populate arrays
Z[:], E[:], N[:] = Zts[i_t,:,:], Zes[i_t,:,:], Zns[i_t,:,:]

# Test if we have fill flags
if flfiles is not None:

    # Create variable for fill flag
    F = foo.createVariable('FillFlag', 'd', ('time', 'y', 'x'), zlib=True)

    # Populate array
    F[:] = Zfl[i_t, :, :]

# Set projection for netcdf
Z.grid_mapping = 'proj'
E.grid_mapping = 'proj'
N.grid_mapping = 'proj'

# Test if we have fill flags
if flfiles is not None:
    F.grid_mapping = 'proj'

proj = foo.createVariable('proj', 'c')
proj.spatial_ref  = Proj
proj.GeoTransform = Transform

# Close file
foo.close()

# How long did the code run    
print("Execution Time: ", str(datetime.now() - startTime))

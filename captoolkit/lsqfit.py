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


# Writing raster files
def geotiffwrite(OFILE, X, Y, Z, dx, dy, proj, format):
    N, M = Z.shape

    driver = gdal.GetDriverByName("GTiff")

    if format == 'int':
        datatype = gdal.GDT_Int32

    if format == 'float':
        datatype = gdal.GDT_Float32

    ds = driver.Create(OFILE, M, N, 1, datatype)

    src = osr.SpatialReference()

    src.ImportFromEPSG(proj)

    ulx = np.min(np.min(X)) - 0.5 * dx

    uly = np.max(np.max(Y)) + 0.5 * dy

    geotransform = [ulx, dx, 0, uly, 0, -dy]

    ds.SetGeoTransform(geotransform)

    ds.SetProjection(src.ExportToWkt())

    ds.GetRasterBand(1).SetNoDataValue(np.nan)

    ds.GetRasterBand(1).WriteArray(Z)


def bilinear2d(xd, yd, data, xq, yq, **kwargs):
    """Raster to point interpolation."""

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0, :]
    yd = yd[:, 0]

    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1] - xd[0]), (yd[1] - yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq * ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq * ones(xq.size)

    xp = (xq - xd[0]) * (nx - 1) / (xd[-1] - xd[0])
    yp = (yq - yd[0]) * (ny - 1) / (yd[-1] - yd[0])

    coord = np.vstack([yp, xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + proj1)
    proj2 = pyproj.Proj("+init=EPSG:" + proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Construct output grid-coordinates."""

    # Setup grid dimensions
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    # Initiate x/y vectors for grid
    x_i = np.linspace(xmin, xmax, num=Ne)
    y_i = np.linspace(ymin, ymax, num=Nn)

    return np.meshgrid(x_i, y_i)


# Description of program
description = ('Program for fitting and filling data cubes using least-squares')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    'ifile', metavar='ifile', type=str, nargs=1,
    help='input file to process')

parser.add_argument(
    '-m', metavar=('model'), dest='model', type=int, nargs=1,
    help=('Choice of least squares model'),
    default=[1], )

parser.add_argument(
    '-r', metavar=('tref'), dest='tref', type=float, nargs=1,
    help='reference time of fit',
    default=[None], )

parser.add_argument(
    '-w', metavar=('win'), dest='win', type=float, nargs=1,
    help='smoohing window in years',
    default=[None], )

parser.add_argument(
    '-t', metavar=('tmin', 'tmax'), dest='tspan', type=float, nargs=2,
    help='time span for model fit',
    default=[-9999, 9999], )

parser.add_argument(
    '-v', metavar=('X', 'Y', 't', 'Z'), dest='vnames',
    type=str, nargs=4, help=('name of x/y/t/z variables in the HDF5'),
    default=[None], required=True)

parser.add_argument(
    '-s', metavar=None, dest='save', type=str, nargs=1,
    help=('select data to save'),
    choices=('par','ras'), default=['par'], )

parser.add_argument(
    '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
    help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
    default=[str(3031)], )

parser.add_argument(
    '-i', metavar=('niter'), dest='niter', type=int, nargs=1,
    help=('number of model iterations'),
    default=[1], )

parser.add_argument(
    '-o', metavar=('ofile'), dest='ofile', type=str, nargs=1,
    help=('name of output parameter file if "par" is active'),
    default=['param.h5'], )

parser.add_argument(
    '-p', dest='plot', action='store_true',
    help=('inspect fitted parameters'),
    default=False)

parser.add_argument(
    '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
    help=('bounding box for geograph. region (deg or m), optional'),
    default=[None],)

# Add arguments to container
args = parser.parse_args()

# Input and output file name for now
ifile = args.ifile[0]
model = args.model[0]
ofile = args.ofile[0]
tref = args.tref[0]
tmin = args.tspan[0]
tmax = args.tspan[1]
save = args.save[0]
names = args.vnames[:]
proj = args.proj[0]
niter = args.niter[0]
win = args.win[0]
plot = args.plot
bbox = args.bbox[:]

# Print arguments to screen
print('parameters:')
for p in vars(args).items(): print(p)

# Get variable names
xvar, yvar, tvar, zvar = names

# Open file to plot
with h5py.File(ifile, 'r') as fx:

    # Read in needed variables
    X = fx[xvar][:]
    Y = fx[yvar][:]
    Zts = fx[zvar][:]
    time = fx[tvar][:]

# Set all -9999 to NaN-value
Zts[Zts < -2000] = np.nan
Zts[Zts == 0] = np.nan

# Check dimensions
if X.ndim == 1:
    
    # Grid resolution
    dx = np.abs(X[0] - X[1])
    dy = np.abs(Y[0] - Y[1])

    # Construct grid from vectors
    X, Y = make_grid(X.min(), X.max(), Y.min(), Y.max(), dx, dy)

else:
    
    # Grid resolution
    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

# Check for bounding box
if bbox[0] is not None:
    
    # Get bounding box
    xmin, xmax, ymin, ymax = bbox
    
    # Get bounding
    i_bbox = (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax)
    
    # Set to NaN's
    Zts[:,~i_bbox] = np.nan

# Size of grid
(K, N, M) = Zts.shape

# Output rate array
a0 = np.zeros((N, M)) * np.nan
a1 = np.zeros((N, M)) * np.nan
a2 = np.zeros((N, M)) * np.nan
a3 = np.zeros((N, M)) * np.nan
a4 = np.zeros((N, M)) * np.nan
s1 = np.zeros((N, M)) * np.nan
ei = np.zeros((N, M)) * np.nan

# Reference time
if tref is None:

    print('Centering to mean time!')

    # Set to mean time
    tref = time.mean()

# Reference time
dt = time - tref

# Loop trough cube
for i in range(N):
    for j in range(M):

        # Test for data density
        if np.all(np.isnan(Zts[:, i, j])):
            continue
        else:
            pass

        # Get time series
        zts = Zts[:, i, j]

        # Apply a window
        if win is not None:

            # Only use data within the time span
            it = (time < (tmin - win)) | (time > (tmax + win))

        else:

            # Only use data within the time span
            it = (time < tmin) | (time > tmax)

        # Extract wanted time span
        t = dt
        z = zts

        # Set unwanted vales to nan's
        z[it] = np.nan
                
        # Make sure we have enough points
        if len(z[~np.isnan(z)]) <= 3:
            continue
    
        # Sort indices
        it = np.argsort(time[~np.isnan(zts)])
        
        
        # Setup design matrix - trend and acceleration
        A = np.vstack((np.ones(t.shape), t, 0.5 * t ** 2, np.cos(2 * np.pi * t), \
                       np.sin(2 * np.pi * t))).T
                       
        # Determine model to fit
        if model == 1:

            # Set to zero
            A[:,2:5] = 0

            # Columns to add back
            cols = [0, 1]

        elif model == 2:

            # Set to zero
            A[:,3:5] = 0

            # Columns to add back
            cols = [0, 1, 2]

        elif model == 3:

            # Set to zero
            A[:,2] = 0

            # Columns to add back
            cols = [0, 1, 3, 4]

        else:

            # Columns to add back
            cols = [0, 1, 2, 3, 4]
                
        # Model solution
        try:

            # Solve for model parameters
            lsq = sm.RLM(z, A, missing='drop').fit(maxiter=niter, tol=0.001)

        except:

            # Print to terminal
            continue

        # Coefficients of model
        Cm = lsq.params

        # Significance of each coeff.
        pvals = lsq.pvalues

        # Compute RMS
        rms = np.nanstd(z - np.dot(A,Cm))
        
        # Seasonal parameters
        if Cm[-1] == 0:

            # Not available, set to zero
            amp = np.nan
            phs = np.nan

        else:

            # Compute values for amplitude and phase
            amp = np.sqrt(Cm[3]**2 + Cm[4]**2)
            phs = int(365.25 * np.arctan(Cm[4] / Cm[3]) / (2.0 * np.pi))

        # Estimated bias
        a0[i, j] = Cm[0]

        # Estimated rate
        a1[i, j] = Cm[1]

        # Estimated acceleration
        a2[i, j] = Cm[2]

        # Estimated amplitude
        a3[i, j] = amp

        # Estimated phase
        a4[i, j] = phs if phs > 0 else phs + 365

        # Significance of trend
        s1[i, j] = pvals[1]

        # RMS of model fit
        ei[i, j] = rms

    # Print progress
    print(str(i) + '/' + str(N))

if save == 'par':

    # Save solution to disk
    with h5py.File(ofile, 'w') as f:

        # Save meta data
        f['X']    = X
        f['Y']    = Y
        f['p0']   = a0
        f['p1']   = a1
        f['p2']   = a2
        f['p3']   = a3
        f['p4']   = a4
        f['time'] = time
        f['tref'] = tref
        f['s1']   = s1
        f['rms']  = ei

else:

    # Construct output names
    OFILE_RATE = ifile.replace('.nc', '_RATE.tif')
    OFILE_ACCE = ifile.replace('.nc', '_ACCE.tif')
    OFILE_RMSE = ifile.replace('.nc', '_RMSE.tif')

    # Save rate and acceleration to tif files
    geotiffwrite(OFILE_RATE, X, Y, (a1), dx, dy, int(proj), "float")
    geotiffwrite(OFILE_ACCE, X, Y, (a2), dx, dy, int(proj), "float")
    geotiffwrite(OFILE_RMSE, X, Y, (s1), dx, dy, int(proj), "float")

# Change projection
Lon,Lat = transform_coord('3031', '4326', X, Y)

# Remove the pole hole data
a1[np.abs(Lat)>81.5] = np.nan
a2[np.abs(Lat)>81.5] = np.nan

# Print Volume Change Estimates
print('Volume Change :', np.around(np.nansum(a1) * dx * dy * 1e-9, 0), 'km^3/yr')
print('Acceleration  :', np.around(np.nansum(a2) * dx * dy * 1e-9, 0), 'km^3/yr')

# Plot solution for quality control
if plot:

    # Plot acceleration
    plt.figure(figsize=(10, 8))
    plt.title("Acceleration: " + str(np.around(np.nanmean(a2*100), 2)) + ' cma$^{-1}$')
    plt.imshow(a2, cmap='jet', extent=(X.min() * 1e-3, X.max() * 1e-3, Y.min() * 1e-3, Y.max() * 1e-3));
    plt.colorbar()
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")

    # Plot rate
    plt.figure(figsize=(10, 8))
    plt.title("Rate: " + str(np.around(np.nanmean(a1*100), 2)) + ' cma$^{-1}$')
    plt.imshow(a1, cmap='jet', extent=(X.min() * 1e-3, X.max() * 1e-3, Y.min() * 1e-3, Y.max() * 1e-3));
    plt.colorbar()
    plt.clim([-1, 1])
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.show()






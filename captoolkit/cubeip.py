#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:27:10 2017

@author: nilssonj
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import pyproj
import h5py
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from gdalconst import *
from osgeo import gdal, osr
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from datetime import datetime
from scipy.ndimage import generic_filter
from scipy import stats

#
# TODO
# 1. RANDOMIZE THE SELECTION OF POINTS FOR LSC
# 2. SAVE OUTPUT AS H5 FILES THAT WE LATER CAN MERGE INTO CUBE
# 3. TEST THE DIFFERENT KERNELS EXP VERSUS MATERN
#

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


def vec2grd(x, y, z, e, Xi, Yi, dr):
    """ Add data to closest node """
    
    # Output arrays
    Zi = np.zeros(Xi.shape)*np.nan
    Ze = np.zeros(Yi.shape)*np.nan
    Zn = np.zeros(Yi.shape)*np.nan
    
    # Ravel ouput arrays
    zi = Zi.ravel()
    ze = Ze.ravel()
    zn = Zn.ravel()
    
    # Zip data to vector
    coord = list(zip(x, y))

    # Construct KD-Tree
    tree = cKDTree(coord)
    
    # Ravel arrays
    xi = Xi.ravel()
    yi = Yi.ravel()
    
    # Change to meters
    dr *= 1e3
    
    # Loop trough points
    for i in range(len(Xi.ravel())):

        # Get closest data to node
        (dxy,idx) = tree.query((xi[i], yi[i]),k=int(1))  

        # Reject if to far away
        if dxy > dr: continue
      
        # Save data to new arrays
        zi[i] = z[idx] 
        ze[i] = e[idx]
        zn[i] = 1
        
    # Return data arrays
    return zi, ze, zn


def lscip(x, y, z, s, Xi, Yi, d, a, n, m):
    """Interpolation by least squares collocation."""

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
    tree = cKDTree(list(zip(x, y)))

    # Convert to meters
    a *= 1e3
    d *= 1e3
    
    # Loop through observations
    for i in range(len(xi)):

        # Determine type
        if m == 0:

            # Find closest number of observations
            (dxy, idx) = tree.query((xi[i], yi[i]), k=n)
            
            # Test if closest point to far away
            if np.min(dxy) > d:
                continue
            elif len(dxy) > len(x):
                continue
            else:
                pass
        else:

            # Find closest within search radius
            idx = tree.query_ball_point((xi[i], yi[i]), d)

            # Test if closest point to far away
            if len(idx) == 0: continue

            # Compute distance from grid node to obs.
            dxy = np.sqrt((x[idx] - xi[i]) * (x[idx] - xi[i]) + \
                          (y[idx] - yi[i]) * (y[idx] - yi[i]))
        
        # Get parameters
        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]
        
        # Need minimum of two observations 
        if len(zc) < 2: continue

        # Minimum error from allowed
        sc[sc < 0.07] = 0.07

        # Estimate local median (robust) and local variance of data
        m0 = np.nanmedian(zc)
        c0 = np.nanmean(sc) ** 2
        
        # Covariance function for Dxy
        Cxy = c0 * (1 + (dxy / a) - 0.5 * (dxy / a) ** 2) * np.exp(-dxy / a)

        # Compute pair-wise distance
        dxx = cdist(list(zip(xc, yc)), list(zip(xc, yc)), "euclidean")

        # Covariance function Dxx
        Cxx = c0 * (1 + (dxx / a) - 0.5 * (dxx / a) ** 2) * np.exp(-dxx / a)

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


def lscip3d(x, y, t, z, s, h, Xi, Yi, Ti, Hi, d, at, ad, ah, n):
    """Interpolation by least squares collocation."""
    
    # Cast as int!
    n = int(n)
    
    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()
    ti = Ti.ravel()
    hi = Hi.ravel()

    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan
    
    # Create KDTree
    tree = cKDTree(list(zip(x, y)))
    
    # Convert to meters
    ah *= 1e3
    ad *= 1e3
    d  *= 1e3
    
    # Loop through observations
    for i in range(len(xi)):

        # Find closest within search radius
        idx = tree.query_ball_point((xi[i], yi[i]), d)

        # Check number of data points
        if len(idx) > 150:

            # Find closest number of observations
            (dxy, idx) = tree.query((xi[i], yi[i]), k=150)

            if np.min(dxy) > 3 * d:
                continue
            if len(dxy) > len(x):
                continue

        else:

            # Compute distance from grid node to obs.
            dxy = np.sqrt((x[idx] - xi[i]) * (x[idx] - xi[i]) + \
                          (y[idx] - yi[i]) * (y[idx] - yi[i]))

        # Get parameters
        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]
        tc = t[idx]
        hc = h[idx]

        # Need minimum of two observations
        if len(zc) < 2: continue

        # Minimum error from allowed
        sc[sc < 0.07] = 0.07
        
        # Estimate local median (robust) and local variance of data
        m0 = np.nanmedian(zc)
        c0 = np.nanmean(sc) ** 2

        # Center time to current time stamp
        dti = np.abs(tc - ti[i])
        dhi = np.abs(hc - hi[i])

        # Cross-covariance for time and space
        #Cxy = c0 * np.exp(-dti/at) * np.exp(-dxy**2/(2.*ad**2)) * np.exp(-dhi**2/(2.*ah**2))
        Cxy = c0 * np.exp(-dti/at) * np.exp(-dxy**2/(2.*ad**2)) * np.exp(-dhi / ah)

        # Covariance for time
        dtt = np.zeros((len(tc), len(tc)))
        dhh = np.zeros((len(tc), len(tc)))

        # Create pair-wise distance for time
        for ki in range(len(tc)):
            for kj in range(len(tc)):
                dtt[ki, kj] = tc[ki] - tc[kj]
                dhh[ki, kj] = hc[ki] - hc[kj]

        # Take absolute value of time
        dtt = np.abs(dtt)
        dhh = np.abs(dhh)

        # Compute pair-wise distance
        dxx = cdist(list(zip(xc, yc)), list(zip(xc, yc)), "euclidean")

        # Auto-covariance space and time
        #Cxx = c0 * np.exp(-dti/at) * np.exp(-dxx**2/(2.*ad**2)) * np.exp(-dhh**2/(2.*ad**2))
        Cxx = c0 * np.exp(-dti/at) * np.exp(-dxx**2/(2.*ad**2)) * np.exp(-dhh / ad)

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


def blockip(x, y, z, s, Xi, Yi, d, a, n, m):
    """2D interp. by weighted average ."""

    # Ravel grid coord.
    xi = Xi.ravel()
    yi = Yi.ravel()
    
    # Create output vectors
    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    # Create kdtree
    tree = cKDTree(list(zip(x,y)))
    
    # Convert to meters
    a *= 1e3
    d *= 1e3
    
    # Loop through observations
    for i in range(len(xi)):
        
        # Determine type
        if m == 0:
            
            # Find closest number of observations
            (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

            # Test if closest point to far away
            if np.min(dxy) > d:
                continue
            elif len(dxy) > len(x):
                continue
            else:
                pass
        else:
            
            # Find closest within search radius
            idx = tree.query_ball_point((xi[i], yi[i]), d)

            # Test if closest point to far away
            if len(idx) == 0: continue
            
            # Compute distance from grid node to obs.
            dxy = np.sqrt((x[idx] - xi[i]) * (x[idx] - xi[i]) + \
                          (y[idx] - yi[i]) * (y[idx] - yi[i]))

        # Get parameters
        zc = z[idx]
        sc = s[idx]
        
        # Check for empty solutions
        if len(zc[~np.isnan(zc)]) == 0: continue

        # Hard code min error
        sc[sc < 0.07] = 0.07

        # Error weighted average
        wc = (1.0 / (sc ** 2)) * np.exp(-dx/a)
        
        # Predicted value
        zi[i] = np.nansum(wc * zc) / np.nansum(wc)
        #zi[i] = np.nanmedian(zc)
        
        # Create output errors
        es = np.sqrt(1.0 / np.nansum(wc))
        #es = np.nanmean(sc)
        
        # Predicted error
        ei[i] = es if es > 0.07 else 0.07
        
        # Number of data used for prediction
        ni[i] = len(zc)

    # Return interpolated points
    return zi, ei, ni


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


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
    
    return X, Y, Z, dx, dy, proj


def bilinear2d(xd,yd,data,xq,yq, **kwargs):
    """Raster to point interpolation."""

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0,:]
    yd = yd[:,0]

    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1]-xd[0]), (yd[1]-yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq*ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq*ones(xq.size)

    xp = (xq-xd[0])*(nx-1)/(xd[-1]-xd[0])
    yp = (yq-yd[0])*(ny-1)/(yd[-1]-yd[0])

    coord = np.vstack([yp,xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq


# Writing raster files
def geotiffwrite(OFILE, X, Y, Z, dx, dy, proj,format):

    N,M = Z.shape

    driver = gdal.GetDriverByName("GTiff")

    if format == 'int':
        datatype = gdal.GDT_Int32

    if format == 'float':
        datatype = gdal.GDT_Float32

    ds = driver.Create(OFILE, M, N, 1, datatype)

    src = osr.SpatialReference()

    src.ImportFromEPSG(proj)

    ulx = np.min(np.min(X)) - 0.5*dx

    uly = np.max(np.max(Y)) + 0.5*dy

    geotransform = [ulx, dx, 0, uly, 0, -dy]

    ds.SetGeoTransform(geotransform)

    ds.SetProjection(src.ExportToWkt())

    ds.GetRasterBand(1).SetNoDataValue(np.nan)

    ds.GetRasterBand(1).WriteArray(Z)

    ds = None


def spatial_filter(x, y, z, s, dx, dy, sigma=10.0):
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
        idx, = np.where((index == ind[i])  & (s < 1.0))
        
        # Get data
        xb = x[idx]
        yb = y[idx]
        zb = z[idx]
        
        # Make sure we have enough
        if len(zb[~np.isnan(zb)]) <= 3:
            continue
        
        # Centering of coordinates
        dxb, dyb = xb - xb.mean(), yb - yb.mean()
        
        # Design matrix bilinear plane
        Ab = np.vstack((np.ones(xb.shape), dxb, dyb)).T
        
        # Trying to solve for model coeff.
        try:
            
            # Generate model
            fit = sm.RLM(zb, Ab, missing='drop').fit(maxiter=5)
        
            # polynomial coefficients
            p = fit.params
    
            # Compute difference from plane
            dh = zb - np.dot(Ab, p)
        
        except:
            
            # Set to median of values
            dh = zb - np.nanmedian(zb)
            print('Using median instead of model!')
        
        # Identify outliers
        foo = np.abs(dh - np.nanmedian(dh)) > sigma * mad_std(dh)
        
        # Set to nan-value
        zb[foo] = np.nan
        
        # Replace data
        zo[idx] = zb

    return zo

# Description of program
description = ('Spatio-temporal interpolation of altimetry data for data cube')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'ifiles', metavar='ifiles', type=str, nargs=2,
        help='file(s) to process (only hdf5)')

parser.add_argument(
        'opath', metavar='opath', type=str, nargs='+',
        help='output path for data')

parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('prediction mode: (p)oint or (g)rid'),
        choices=('vec', 'lsc', 'idw'), default=['lsc'],)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('grid resolution (deg or meters)'),
        default=[10,10],)

parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
        help=('maximum allowed search radius'),
        default=[25],)

parser.add_argument(
        '-a', metavar='res_param', dest='resparam', type=float, nargs=1,
        help=('correlation length for interpolation (km)'),
        default=[1],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('number of closest point for interpolation'),
        default=[4],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(3031)],)

parser.add_argument(
        '-f', metavar=('mask.tif'), dest='mask',  type=str, nargs=1,
        help='filename of mask (".tif")',
        default=[None],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help='for parallel processing of multiple files, optional',
        default=[1],)

parser.add_argument(
        '-k', metavar=('kernel'), dest='kernel', type=int, nargs=1,
        help='smoothing kernel size, optional',
        default=[0],)

parser.add_argument(
        '-s', metavar=('m_sol'), dest='msol', type=int, nargs=1,
        help='nearest (0) or search radius (1) solution',
        default=[0],)

parser.add_argument(
        '-t', metavar=('val_lim,','rms_lim'), dest='threshold', type=float, nargs=2,
        help='remove observations with RMSE > threshold',
        default=[9999],)

parser.add_argument(
        '-q', metavar=('fslope'), dest='fslope', type=str, nargs=1,
        help='name of slope file',
        default=[None],)

parser.add_argument(
        '-l', metavar=('slim'), dest='slope_lim', type=float, nargs=1,
        help='slope limit in degrees (deg)',
        default=[9999],)

parser.add_argument(
        '-w', metavar=('dx','dy'), dest='ores',  type=float, nargs=2,
        help='resolution of cells used for outlier rejection (km)',
        default=[0,0],)

# Add arguments to container
args = parser.parse_args()

# Pass arguments
tfile  = args.ifiles[0]              # input file time series
efile  = args.ifiles[1]              # input file time series standard error
opath  = args.opath[0]               # output directory(s)
dx     = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy     = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax_  = args.radius[0]              # max search radius (km)
dres   = args.resparam[0]            # resolution param for weighting func (km)
nlim   = args.minobs[0]              # min obs for solution
proj   = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031) for OBS
mask   = args.mask[0]                # name of binary mask, needs to be ".tif"
njobs  = args.njobs[0]               # for parallel processing
mode   = args.mode[0]                # prediction mode: point or grid solution
kern   = args.kernel[0]              # size of median filter smoothing kernel, if zero (default) not applied
value  = args.threshold[0]           # Remove data with values larger than threshold (meters)
thrs   = args.threshold[1]           # Remove data with RMSE larger than threshold (meters)
m_sol  = args.msol[0]                # Solution type nearest (0) search radius (1)
fslope = args.fslope[0]              # Name of slope file
slim   = args.slope_lim[0]           # Max allowed slope
dxc    = args.ores[0]*1e3
dyc    = args.ores[1]*1e3

# Print arguments to screen
print('parameters:')
for p in vars(args).items(): print(p)

# Test for file type
if tfile.find("ts") or tfile.find("TS") > 0:
    
    # Kalman Smoother
    tsvar = 'ts'
    esvar = 'es'

else:
    
    # Weighted average
    tsvar = 'tw'
    esvar = 'ew'

# Load data from disk
with h5py.File(tfile, 'r') as fx:
    
    # Read in needed variables
    ts   = fx[tsvar][:]

with h5py.File(efile, 'r') as fy:
    
    # Read in needed variables
    es   = fy[esvar][:]

# EPSG number for lon/lat proj
projGeo = '4326'

# EPSG number for grid proj
projGrd = proj

# Remove all NaN rows
i_nan = np.where(np.isnan(ts[:, 3]))
ts = np.delete(ts.T, i_nan, 1).T

# Remove all NaN rows
i_nan = np.where(np.isnan(es[:, 3]))
es = np.delete(es.T, i_nan, 1).T

# Extract coordinates
(lonp, latp) = ts[:, 1], ts[:, 0]

# Time parameters
t_start  = ts[0,2]
t_stop   = ts[0,3]
n_months = ts[0,4]

# Compute time step
dt_step = (t_stop - t_start) / n_months

# Compute time vector
time = np.arange(t_start, t_stop, dt_step) + 0.5 * dt_step

# Extracts data arrays for time series 
TS = ts[:, 5:]
ES = es[:, 5:]

# Convert to equal-area coordinate system
(xp, yp) = transform_coord(projGeo, projGrd, lonp, latp)

# Data buffer
buff = 20e3

# Construct wanted grid
(Xi, Yi) = make_grid(xp.min()-buff, xp.max()+buff, yp.min()-buff, yp.max()+buff, dx, dy)

# Time vector
Ti = (np.ones(TS.shape) * time)

# Create long and lat variables
(Lon, Lat) = transform_coord(projGrd, projGeo, np.flipud(Xi), np.flipud(Yi))

# Check for mask
if mask:

    print("Reading mask ...")
    # Read in masking file
    Xm, Ym, Zm = geotiffread(mask)[0:3]

    # Re-sample masking grid to wanted resolution - nearest value
    Zm = np.flipud(bilinear2d(Xm, Ym, Zm, Xi.ravel(), Yi.ravel(), order=0).reshape(Xi.shape))

# Check for slope file
if fslope:

    print("Reading elevation ...")
    # Read in masking file
    Xs, Ys, Zs = geotiffread(fslope)[0:3]

    # Inter polate the DEM to points
    hp = bilinear2d(Xs, Ys, Zs, xp, yp, order=1)

    # Reformat to grid coordinates
    Zs = np.flipud(bilinear2d(Xs, Ys, Zs, Xi.ravel(), Yi.ravel(), order=0).reshape(Xi.shape))

# Remove outliers
if dxc != 0:

    # Set to dummy if we don't have it
    sp = np.ones(xp.shape) * 1e6

    # Loop trough spatial fields
    for ki in range(len(TS.T)):

        # Remove outliers
        TS[:, ki] = spatial_filter(xp.copy(), yp.copy(), TS[:,ki].copy(), sp.copy(), dx=dxc, dy=dyc)

        # Get NaN value locations
        nans = np.isnan(TS[:, ki])

        # Set errors to NaN
        ES[nans, ki] = np.nan

        print("Cleaning:", ki)

# Start timing of script
startTime = datetime.now()

# Create vectors
T   = (np.ones(TS.shape) * time)
HS  = (np.ones(TS.shape) * hp.reshape((len(xp), 1)))
XS  = (np.ones(TS.shape) * xp.reshape((len(xp), 1)))
YS  = (np.ones(TS.shape) * yp.reshape((len(yp), 1)))
LAT = (np.ones(TS.shape) * latp.reshape((len(latp), 1)))

# Run interpolation
def main(i, TS, ES, XS, YS, LAT, Xi, Yi, Ti, Zs, imax):

    # Select data
    x = XS[:, i]
    y = YS[:, i]
    z = TS[:, i]
    t =  T[:, i]
    s = ES[:, i]
    lat = LAT[:, i]
    dem = HS[:, i ]

    # Select data
    #x = XS[:, i - 2:i + 1].ravel()
    #y = YS[:, i - 2:i + 1].ravel()
    #z = TS[:, i - 2:i + 1].ravel()
    #s = ES[:, i - 2:i + 1].ravel()
    #t =  T[:, i - 2:i + 1].ravel()
    #lat = LAT[:, i - 2:i + 1].ravel()
    #dem = HS[:, i - 2:i + 1].ravel()

    # Find NaN's and remove large values
    inan = ~np.isnan(z) & (s < thrs) & (np.abs(z) < value)
    
    # Get available latitude data
    lat_i = lat[inan]

    # Test for all NaN's
    if np.all(np.isnan(z[inan])) or ((time[i] < 1992.2) and (time[i] > 1989.2)):

        # Get loop index
        index = '%03d' % i

        # Create NaN only arrays
        Zi, Ei, Ni = np.ones(Xi.shape)*np.nan, np.ones(Xi.shape)*np.nan, np.ones(Xi.shape)*np.nan

        # Write to tiff file
        geotiffwrite(opath + index + '_ts_' + str(time[i]) + '.tif', Xi, Yi, Zi, dx, dy, int(proj), "float")
        geotiffwrite(opath + index + '_es_' + str(time[i]) + '.tif', Xi, Yi, Ei, dx, dy, int(proj), "float")
        geotiffwrite(opath + index + '_ns_' + str(time[i]) + '.tif', Xi, Yi, Ni, dx, dy, int(proj), "float")

        print("Only nan's in array!")
        return

    # Set dmax
    dmax = dmax_

    if time[i] < 1993:
        tres = 1./12.
    else:
        tres = 1/12.

    # Interpolate data to grid
    if mode == 'lsc':

        # LSC-algorithm
        #(zi, ei, ni) = lscip(x[inan], y[inan], z[inan], s[inan], Xi, Yi, dmax, dres, nlim, m_sol)
        (zi, ei, ni) = lscip3d(x[inan], y[inan], t[inan], z[inan], s[inan], dem[inan], \
                               Xi, Yi, Ti, Zs, dmax, tres, dres, 100./1000, nlim)

    elif mode =='idw':

        # IDW-algorithm
        (zi, ei, ni) = blockip(x[inan], y[inan], z[inan], s[inan], Xi, Yi, dmax, dres, nlim, m_sol)

    else:

        # vec2grd-algorithm
        (zi, ei, ni) = vec2grd(x[inan], y[inan], z[inan], s[inan], Xi, Yi, dmax)

    # Reshape output
    Zi = np.flipud(zi.reshape(Xi.shape))
    Ei = np.flipud(ei.reshape(Xi.shape))
    Ni = np.flipud(ni.reshape(Xi.shape))

    # Apply filter kernel
    if kern > 0:
        
        # Apply median filter to raster
        Zi = generic_filter(Zi.copy(), np.nanmedian, kern)
    
    # Check if mask should be applied
    if mask:
    
        # Apply mask
        Zi[(Zm != 1)] = np.nan
        Ei[(Zm != 1)] = np.nan
        Ni[(Zm != 1)] = np.nan

    # Check if we should fill pole hole for Antarctica
    if 1:

        # Pole hole clipping
        if np.all(np.abs(lat_i) < 75.0):

            # Remove data larger than 81.5 deg
            Zi[np.abs(Lat) > 72.0] = -9999
            Ei[np.abs(Lat) > 72.0] = -9999
            Ni[np.abs(Lat) > 72.0] = -9999

            try:
                # Set slopes larger than one to NaNs
                Zi[Zs > slim] = np.nan
                Ei[Zs > slim] = np.nan
                Ni[Zs > slim] = np.nan
            except:
                    pass

        # Pole hole clipping
        elif np.all(np.abs(lat_i) < 83.0):

            # Remove data larger than 81.5 deg
            Zi[np.abs(Lat) > 81.5] = -9999
            Ei[np.abs(Lat) > 81.5] = -9999
            Ni[np.abs(Lat) > 81.5] = -9999
            try:
                # Set slopes larger than one to NaNs
                Zi[Zs > slim] = np.nan
                Ei[Zs > slim] = np.nan
                Ni[Zs > slim] = np.nan
            except:
                pass
                    
        # Pole hole clipping
        elif np.all(np.abs(lat_i) < 87.0):

            # Remove data larger than 86.0 deg
            Zi[np.abs(Lat) > 86.0] = -9999
            Ei[np.abs(Lat) > 86.0] = -9999
            Ni[np.abs(Lat) > 86.0] = -9999

            try:
                # Set slopes larger than one to NaNs
                Zi[Zs > slim] = np.nan
                Ei[Zs > slim] = np.nan
                Ni[Zs > slim] = np.nan
            except:
                pass

        # Pole hole clipping
        else:
            
            try:
                # Remove data larger than 88.0 deg
                Zi[np.abs(Lat) > 88.0] = -9999
                Ei[np.abs(Lat) > 88.0] = -9999
                Ni[np.abs(Lat) > 88.0] = -9999
            except:
                pass

    # Flip data to align with data raster
    Xi, Yi = np.flipud(Xi), np.flipud(Yi)
    
    # Get loop index
    index = '%03d' % i

    # Write to tiff file 
    geotiffwrite(opath+index+'_ts_'+str(time[i])+'.tif', Xi, Yi, Zi, dx, dy, int(proj), "float")
    geotiffwrite(opath+index+'_es_'+str(time[i])+'.tif', Xi, Yi, Ei, dx, dy, int(proj), "float")
    geotiffwrite(opath+index+'_ns_'+str(time[i])+'.tif', Xi, Yi, Ni, dx, dy, int(proj), "float")

    # Display which datum is being processed
    print(str(i+1),'/',str(len(TS.T)))

# Run main program!
if njobs == 1:
    
    # Single core
    print('running sequential code ...')
    [main(i,TS,ES,XS,YS,LAT,Xi,Yi,Ti,Zs,len(TS.T)) for i in range(len(TS.T))]

# Run main program in parallel 
else:
    
    # Multiple cores
    print('running parallel code (%d jobs) ...' % njobs)
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(i,TS,ES,XS,YS,LAT,Xi,Yi,Ti,Zs,len(TS.T)) for i in range(len(TS.T)))

# Print execution time
print("Execution Time: ", str(datetime.now()-startTime))
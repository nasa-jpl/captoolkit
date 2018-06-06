#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:15:38 2015

@author: nilssonj
"""

import sys
import pyproj
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from gdalconst import *
from osgeo import gdal, osr

"""
Program for gridding irregular spaced point data by the means of least-squares collocation to a regular grid off a given
projection. The program uses; either N nearest neighbours search around each grid node (divided in eight quadrants) or
a fixed search radius (no sectors) for each prediction. The covariance structure inside each solution area is modelled
using a 3rd order Gauss-Markov model, based on distance and given a correlation length. For each solution an iterative
outlier rejection scheme can be applied if needed, given the correct input. When a fixed search radius is used the data
is thinned by random selection of obs. inside the search cap to speed up the inversion, but keeping spatial coverage
intact (also reduces striping). The program takes as input either (1) ordinary ascii files ".txt" or (2) python binary
files ".npy" for faster I/O and outputs the gridded data to geotiff.The program provides powerful, flexible and fast
gridding of large spatial datasets for anytype spatial dataset.

INPUT:
------

ifile   :   Name of input (x,y,z) file.
ofile   :   Name of output raster(s).
bbox    :   String for bounding box and grid resolution "xmin, xmax, ymin, ymax, dx, dy".
proj    :   Projection type (EPSG number).
mode    :   Prediction mode: (1) nearest-neighbour (2) max. search radius.
nobs    :   Number of obs. for each cell (mode 1) or max. distance in km (mode 2).
dmax    :   Exclude data with minimum distance > dmax (km).
alpha   :   Correlation/Resolution length parameter (km).
sigma   :   RMS-noise: if s>0 all sigma values < sigma set to sigma, or s<0 all values given same sigma.
icols   :   String of input columns in input file: "x y z s", if (s < 1) sigma_i = sigma .
tol     :   Tolerance for outlier rejection (in %), if tol < 0 no outlier editing
thres   :   Sigma threshold (thres*sigma), if negative no outlier editing applied

USAGE:
------

lscip.py ifile.txt ofile.tif "69 84 -75 -10 0.01 0.025" 4326 1 100 10 1 "2 1 3 4" 50

"""


def geotiffwrite(OFILE, X, Y, Z, dx, dy, proj,format):
    """Writing to raster file"""
    (N,M) = Z.shape
    
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


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def iterfilt(x, xmin, xmax, tol, alpha):
    """ Iterative outlier filter """
    
    # Set default value
    tau = 100.0
    
    # Remove data outside selected range
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan
    
    # Initiate counter
    k = 0
    
    # Outlier rejection loop
    while tau > tol:
        
        # Compute initial rms
        rmse_b = mad_std(x)
        
        # Compute residuals
        dh_abs = np.abs(x - np.nanmedian(x))
        
        # Index of outliers
        io = dh_abs > alpha * rmse_b
        
        # Compute edited rms
        rmse_a = mad_std(x[~io])
        
        # Determine rms reduction
        tau = 100.0 * (rmse_b - rmse_a) / rmse_a
        
        # Remove data if true
        if tau > tol or k == 0:
            
            # Set outliers to NaN
            x[io] = np.nan
            
            # Update counter
            k += 1
    
    return x

def rand(x, n):
    """Draws random samples from array"""

    # Determine data density
    if len(x) > n:

        # Draw random samples from array
        I = np.random.choice(np.arange(len(x)), n, replace=False)
    
    else:

        # Output boolean vector - true
        I = np.ones(len(x), dtype=bool)

    return I


def sort_dist(d, n):
    """ Sort array by distance"""
    
    # Determine if sorting needed
    if len(d) >= n:
        
        # Sort according to distance
        I = np.argsort(d)
    
    else:
        
        # Output boolean vector - true
        I = np.ones(len(x), dtype=bool)

    return I

# Description of algorithm
des = 'Optimal interpolation of scattered data using Least-Squares Collocation'

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
        'input', metavar='ifile', type=str, nargs='+',
        help='name of i-file, numpy binary or ascii (for binary ".npy")')

parser.add_argument(
        'output', metavar='ofile', type=str, nargs='+',
        help='name of o-file, numpy binary or ascii (for binary ".npy")')

parser.add_argument(
        '-i', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m), optional'),
        default=[],)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid (deg or km)'),
        default=[1, 1],)

parser.add_argument(
        '-n', metavar='nobs', dest='nobs', type=int, nargs=1,
        help=('number of obs. for each quadrant'),
        default=[1],)

parser.add_argument(
        '-r', metavar='radius', dest='radius', type=float, nargs=1,
        help=('maximum search radius (km)'),
        default=[1],)

parser.add_argument(
        '-a', metavar='alpha', dest='alpha', type=float, nargs=1,
        help=('correlation length (km)'),
        default=[1],)

parser.add_argument(
        '-e', metavar='sigma', dest='sigma', type=float, nargs=1,
        help=('rms noise of obs. (m)'),
        default=[1],)

parser.add_argument(
        '-f', metavar=('min','max','tol','thrs'), dest='filt', type=float, nargs=4,
        help=('reject obs: obs_min, obs_max, tolerance (pct), N*rms'),
        default=[-9999,9999,5,3],)

parser.add_argument(
        '-c', metavar=(0,1,2,3), dest='cols', type=int, nargs=4,
        help=('data cols (lon, lat, obs, error), -1 = dont use'),
        default=[2,1,3,-1],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('prediction mode: nearest (nearest) or search radius (r).'),
        choices=('n', 'r'), default=['r'],)

parser.add_argument(
        '-s', metavar=None, dest='select', type=str, nargs=1,
        help=('sampling mode: random (r) or distance (s).'),
        choices=('r', 's'), default=['s'],)

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifile = args.input[0]
ofile = args.output[0]
bbox  = args.bbox
dx    = args.dxy[0] * 1e3
dy    = args.dxy[1] * 1e3
proj  = args.proj[0]
nobs  = args.nobs[0]
dmax  = args.radius[0] * 1e3
alpha = args.alpha[0] * 1e3
sigma = args.sigma[0]
icols = args.cols
zmin  = args.filt[0]
zmax  = args.filt[1]
tol   = args.filt[2]
thres = args.filt[3]
mode  = args.mode[0]
selec = args.select[0]

# Print parameters to screen
print 'parameters:'
for p in vars(args).iteritems(): print p

# Extract columns indexes
(cx, cy, cz, cs) = icols

# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")

# Make pyproj format
projfull = "+init=EPSG:"+proj

# Projection - prediction grid
projGrd = pyproj.Proj(projfull)

# Start timing of script
startTime = datetime.now()

print "reading input file ..."

# Read data from differetn file formats
if ifile.endswith('.npy'):
    
    # Load Numpy binary file to memory
    Points = np.load(ifile)

else:
    
    # Load ASCII file to memory
    Points = pd.read_csv(ifile, engine="c", header=None, delim_whitespace=True)
                         
    # Convert to numpy array
    Points = pd.DataFrame.as_matrix(Points)

# Convert into stenographic coordinates
(xp, yp) = pyproj.transform(projGeo, projGrd, Points[:, cx], Points[:, cy])

# Test for different types of input
if len(bbox) == 6:

    # Extract bounding box elements
    (xmin, xmax, ymin, ymax) = bbox

else:

    # Create bounding box limits
    (xmin, xmax, ymin, ymax) = (xp.min() - 10.0*dx), (xp.max() + 10.0*dx), (yp.min() - 10.0*dy), (yp.max() + 10.0*dy)

# Setup grid dimensions
Nn = int((np.abs(ymax - ymin)) / dy) + 1
Ne = int((np.abs(xmax - xmin)) / dx) + 1

# Initiate lat/lon vectors for grid
x_i = np.linspace(xmin, xmax, num=Ne)
y_i = np.linspace(ymin, ymax, num=Nn)

# Construct output grid-coordinates
(Xi, Yi) = np.meshgrid(x_i, y_i)

# Flatten prediction grid
xi = Xi.ravel()
yi = Yi.ravel()

# Geographical projection
if np.abs(ymax) < 100:
    
    # Convert to stereographic coord.
    (xi, yi) = pyproj.transform(projGeo, projGrd, xi, yi)

print "setting up kd-tree ..."

# Construct cKDTree - points
TreeP = cKDTree(zip(xp, yp))

# Markov-model parameter
a = 0.9132 * alpha

# Compute noise variance
crms = sigma * sigma

# Output vectors
zi = np.ones(len(xi)) * np.nan
ei = np.ones(len(xi)) * np.nan
ni = np.ones(len(xi)) * np.nan

# Extract observations
zp = Points[:, cz]

print "looping grid nodes ..."

# Enter prediction loop
for i in xrange(len(xi)):
    
    # Detect mode
    if mode == 's':
        
        # Get indexes from Tree 
        idx = TreeP.query_ball_point((xi[i], yi[i]), dmax)
        
        # Compute distance from grid node to obs.
        dxy = np.sqrt((xp[idx] - xi[i]) * (xp[idx] - xi[i]) + (yp[idx] - yi[i]) * (yp[idx] - yi[i]))
                    
    else:
        
        # Find closest observations
        (dxy, idx) = TreeP.query((xi[i], yi[i]), nobs * 8)
        
        # Test if closest point to far away
        if np.min(dxy) > dmax: continue
    
    # Test for empty cell
    if len(zp[idx]) == 0: continue
    
    # Obs. before editing
    nb = len(zp[idx])

    # Quick outlier editing
    Io = ~np.isnan(iterfilt(zp[idx].copy(), zmin, zmax, tol, thres))

    # Parameters
    x = xp[idx][Io]
    y = yp[idx][Io]
    z = zp[idx][Io]
    
    # Distance editing
    dxy = dxy[Io]
                    
    # Obs. after editing
    na = len(z)

    # Outliers removed
    dn = nb - na

    # Test for empty cell
    if len(z) == 0: continue

    # Noise handling
    if cs < 0:

        # Provide all obs. the same RMS
        c = np.ones(len(x)) * crms

    else:

        # Set all obs. errors < crms to crms
        c = Points[idx, cs][Io] ** 2
        c[c < crms] = crms

    # Compute angle to data points
    theta = (180.0 / np.pi) * np.arctan2(y - yi[i], x - xi[i]) + 180

    # Get index for data in 8-sectors
    IQ1 = (theta > 0) & (theta < 45)
    IQ2 = (theta > 45) & (theta < 90)
    IQ3 = (theta > 90) & (theta < 135)
    IQ4 = (theta > 135) & (theta < 180)
    IQ5 = (theta > 180) & (theta < 225)
    IQ6 = (theta > 225) & (theta < 270)
    IQ7 = (theta > 270) & (theta < 315)
    IQ8 = (theta > 315) & (theta < 360)

    # Merge all data to sectors
    Q1 = np.vstack((x[IQ1], y[IQ1], z[IQ1], c[IQ1], dxy[IQ1])).T
    Q2 = np.vstack((x[IQ2], y[IQ2], z[IQ2], c[IQ2], dxy[IQ2])).T
    Q3 = np.vstack((x[IQ3], y[IQ3], z[IQ3], c[IQ3], dxy[IQ3])).T
    Q4 = np.vstack((x[IQ4], y[IQ4], z[IQ4], c[IQ4], dxy[IQ4])).T
    Q5 = np.vstack((x[IQ5], y[IQ5], z[IQ5], c[IQ5], dxy[IQ5])).T
    Q6 = np.vstack((x[IQ6], y[IQ6], z[IQ6], c[IQ6], dxy[IQ6])).T
    Q7 = np.vstack((x[IQ7], y[IQ7], z[IQ7], c[IQ7], dxy[IQ7])).T
    Q8 = np.vstack((x[IQ8], y[IQ8], z[IQ8], c[IQ8], dxy[IQ8])).T
    
    # Determine sampling strategy
    if selec == 'r':
        
        # Draw random samples from each sector
        I1 = rand(Q1[:, 0], nobs)
        I2 = rand(Q2[:, 0], nobs)
        I3 = rand(Q3[:, 0], nobs)
        I4 = rand(Q4[:, 0], nobs)
        I5 = rand(Q5[:, 0], nobs)
        I6 = rand(Q6[:, 0], nobs)
        I7 = rand(Q7[:, 0], nobs)
        I8 = rand(Q8[:, 0], nobs)

    else:
        
        # Draw closest samples from each sector
        I1 = rand(Q1[:, 4], nobs)
        I2 = rand(Q2[:, 4], nobs)
        I3 = rand(Q3[:, 4], nobs)
        I4 = rand(Q4[:, 4], nobs)
        I5 = rand(Q5[:, 4], nobs)
        I6 = rand(Q6[:, 4], nobs)
        I7 = rand(Q7[:, 4], nobs)
        I8 = rand(Q8[:, 4], nobs)

    # Stack the data
    Q18 = np.vstack((Q1[I1, :], Q2[I2, :], Q3[I3, :], Q4[I4, :], Q5[I5, :], Q6[I6, :], Q7[I7, :], Q8[I8, :]))

    # Extract position and data
    xc = Q18[:, 0]
    yc = Q18[:, 1]
    zc = Q18[:, 2]
    cc = Q18[:, 3]

    # Distance from grid node to data
    Dxy = Q18[:, 4]

    # Estimate local median (robust) and local variance of data
    m0 = np.median(zc)
    c0 = np.var(zc) 

    # Covariance function for Dxy
    Cxy = c0 * (1 + (Dxy / a) - 0.5 * (Dxy / a) ** 2) * np.exp( -Dxy / a)

    # Compute pair-wise distance 
    Dxx = cdist(zip(xc, yc), zip(xc, yc), "euclidean")
    
    # Covariance function Dxx
    Cxx = c0 * (1 + (Dxx / a) - 0.5 * (Dxx / a) ** 2) * np.exp( -Dxx / a)

    # Measurement noise matrix
    N = np.eye(len(Cxx)) * np.diag(cc)

    # Matrix inversion of Cxy(Cxy+N)^(-1)
    CxyCxxi = np.linalg.solve((Cxx + N).T, Cxy.T)

    # Predicted value
    zi[i] = np.dot(CxyCxxi, zc) + (1 - np.sum(CxyCxxi)) * m0
    
    # Predicted error
    ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))
    
    # Number of data used for prediction    
    ni[i] = len(zc)

    # Print progress to terminal
    if (i % 500) == 0:
        
        # N-predicted values
        print str(i)+'/'+str(len(xi))+' Pred: '+str(np.around(zi[i],2))+'  Nsol: '+str(ni[i])+'  Dmax: ' \
              +str(np.around(1e-3 * dxy.max(),2))

# Convert back to arrays
Zi = np.flipud(zi.reshape(Xi.shape))
Ei = np.flipud(ei.reshape(Xi.shape))
Ni = np.flipud(ni.reshape(Xi.shape))

# Flip coordinates
Xi = np.flipud(Xi)
Yi = np.flipud(Yi)

# Set output names
OFILE_1 = ofile[:-4]+'_PRED'+'.tif'
OFILE_2 = ofile[:-4]+'_RMSE'+'.tif'
OFILE_3 = ofile[:-4]+'_NSOL'+'.tif'

print "saving data ..."

# Write data to geotiff-format
geotiffwrite(OFILE_1, Xi, Yi, Zi, dx, dy, int(proj), "float")
geotiffwrite(OFILE_2, Xi, Yi, Ei, dx, dy, int(proj), "float")
geotiffwrite(OFILE_3, Xi, Yi, Ni, dx, dy, int(proj), "float")

# Print execution time of script
print 'Execution time: '+ str(datetime.now()-startTime)
#!/usr/bin/env python

import sys
import pyproj
import pandas as pd
import numpy as np
import argparse
import warnings
import h5py
from datetime import datetime
from scipy.spatial import cKDTree
from gdalconst import *
from osgeo import gdal, osr
import matplotlib.pyplot as plt

"""
Program for gridding irregular point data using distance weighted average interpolation. It uses a "gaussian" distance
weighting scheme, with the option of including error weights for each individual observation. For each predication a
iterative outlier rejection scheme is applied to remove outliers from the mean. The program takes as input either (1)
ordinary ascii files ".txt" or python binary files ".npy" for faster I/O and outputs the gridded data to geotiff. The
program provides the option to select different projections of the output rasters. The program provides powerful,
flexible and fast gridding of large spatial datasets for any type spatial dataset.
"""

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


# Description of algorithm
des = 'Interpolation of scattered data using distance and error weighted average'

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
        'ifile', metavar='ifile', type=str, nargs='+',
        help='name of i-file, numpy binary or ascii (for binary ".npy")')

parser.add_argument(
        'ofile', metavar='ofile', type=str, nargs='+',
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
        help=('number of obs. for each quadrant, only "nearest" '),
        default=[4],)

parser.add_argument(
        '-r', metavar='radius', dest='radius', type=float, nargs=1,
        help=('search radius (km), or distance cutoff "nearest"'),
        default=[1],)

parser.add_argument(
        '-a', metavar='alpha', dest='alpha', type=float, nargs=1,
        help=('correlation length (km)'),
        default=[1],)

parser.add_argument(
        '-f', metavar=('tol','thrs'), dest='filt', type=float, nargs=2,
        help=('outlier rejection: thres * sigma and tolerance (pct)'),
        default=[100,100],)

parser.add_argument(
        '-c', metavar=(0,1,2,3), dest='cols', type=int, nargs=4,
        help=("data cols (lon, lat, obs, sigma), -1 = don't use"),
        default=[2,1,3,-1],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('radius or nearest neighbour interpolation'),
        choices=('r', 'n'), default=['r'],)

parser.add_argument(
        '-o', metavar=None, dest='otype', type=str, nargs=1,
        help=('output as point (ascii or binarey) or grid (tif) file'),
        choices=('p', 'g'), default=['grid'],)

parser.add_argument(
        '-v', metavar=('x','y','z','s'), dest='vnames', type=str, nargs=4,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','t_year','h_cor','h_rms'],)


# Parser argument to variable
args = parser.parse_args()

# Supress all warnings
warnings.filterwarnings('ignore')

# Read input from terminal
ifile = args.ifile[0]
ofile = args.ofile[0]
bbox  = args.bbox
dx    = args.dxy[0] * 1e3
dy    = args.dxy[1] * 1e3
proj  = args.proj[0]
mode  = args.mode[0]
nobs  = args.nobs[0]
dmax  = args.radius[0]
alpha = args.alpha[0]
icol  = args.cols
tol   = args.filt[0]
thres = args.filt[1]
otype = args.otype[0]
vicol = args.vnames[:]

# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")

# Make pyproj format
projfull = '+init=EPSG:'+proj

# Projection - prediction grid
projGrd = pyproj.Proj(projfull)

# Start timing of script
startTime = datetime.now()

print "reading data ..."

# Binary file
if ifile.endswith('.npy'):
    
    # Extract columns indexes from string
    (cx, cy, cz, cs) = icol

    # Load binary file to memory
    Points = np.load(ifile)

    # Convert into sterographic coordinates
    (xp, yp) = pyproj.transform(projGeo, projGrd, Points[:, cx], Points[:, cy])

    # Data column
    zp = Points[:, cz]
    
    # Include error in solution
    if cs <= 0:
        
        # Create array of ones
        s = np.ones(len(z))
    
    else:
        
        # Get a-priori errors
        sp = Points[:,cs]

# Hdf5-files
elif ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
    
    # Get variable names
    xvar, yvar, zvar, svar = vicol
        
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:
        
        # Get variables
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        zp = fi[zvar][:]
        sp = fi[svar][:] if svar in fi else np.ones(lon.shape)
        
        # Convert into sterographic coordinates
        (xp, yp) = pyproj.transform(projGeo, projGrd, lon, lat)

# Ascii file
else:
    
    # Get columns for data
    cx, cy, cz, cs = icol
    
    # Load input file to memory
    Points = pd.read_csv(ifile, engine="c", header=None, delim_whitespace=True)

    # Converte to numpy array
    Points = pd.DataFrame.as_matrix(Points)
    
    # Convert into sterographic coordinates
    (xp, yp) = pyproj.transform(projGeo, projGrd, Points[:, cx], Points[:, cy])
    
    # Data column
    zp = Points[:, cz]
    
    # Include error in solution
    if cs <= 0:
    
        # Create array of ones
        s = np.ones(len(z))
    
    else:
        
        # Get a-priori errors
        sp = Points[:,cs]

# Test for different types of input
if len(bbox) == 6:
    
    # Extract bounding box elements
    (xmin, xmax, ymin, ymax) = bbox

else:

    # Create bounding box limits
    (xmin, xmax, ymin, ymax) = (xp.min() - dx), (xp.max() + dx), (yp.min() - dy), (yp.max() + dy)

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

# Flatten data to vectors
P = zip(xp, yp)
G = zip(xi, yi)

# Geographical projection
if np.abs(ymax) < 100:

    # Convert to stereographic coord.
    (xi, yi) = pyproj.transform(projGeo, projGrd, xi, yi)

print "creating KDTree ..."

# Construct cKDTree
TreeP = cKDTree(P)

# Output vectors
zi = np.ones(len(xi)) * np.nan
ei = np.ones(len(xi)) * np.nan
ni = np.ones(len(xi)) * np.nan

# Enter prediction loop
for i in xrange(len(zi)):

    # Select prediction mode
    if mode == 'r':

        # Find observations within search cap
        idx = TreeP.query_ball_point(G[i], dmax * 1e3)
    
        # Test for zeros
        if len(xp[idx]) == 0:
            continue
    else:

        # Find closest observations
        (d, idx) = TreeP.query(G[i], nobs * 8)

        # Test if closest point to far away
        if np.min(d) > dmax * 1e3:
            continue

    # Extract data for solution
    x = xp[idx]
    y = yp[idx]
    z = zp[idx]
    s = sp[idx]

    # Test if cell is empty
    if len(z) == 0: continue

    # Detect outliers
    tmp = iterfilt(z,-9999,9999, 5.0, 5.0)

    # Detect NaN's
    Inan = ~np.isnan(tmp)

    # Remove outliers
    if len(z[Inan]) != 0:

        # Remove outliers
        x = x[Inan]
        y = y[Inan]
        z = z[Inan]
        s = s[Inan]

        if mode == 'n':

            # Remove outlier
            d = d[Inan]

    # Search radius - Fixed radius
    if mode == 'r':

        # Compute distance from grid node to obs.
        d = np.sqrt((x - xi[i]) * (x - xi[i]) + (y - yi[i]) * (y - yi[i]))

    # Nearest neighbour - Sector search
    else:

        # Compute angle to data points
        theta = (180.0 / np.pi) * np.arctan2(y - yi[i], x - xi[i]) + 180

        # Get index for data in four sectors
        IQ1 = (theta > 0) & (theta < 90)
        IQ2 = (theta > 90) & (theta < 180)
        IQ3 = (theta > 180) & (theta < 270)
        IQ4 = (theta > 270) & (theta < 360)

        # Merge data in different sectors
        Q1 = np.vstack((x[IQ1], y[IQ1], z[IQ1], s[IQ1], d[IQ1])).T
        Q2 = np.vstack((x[IQ2], y[IQ2], z[IQ2], s[IQ2], d[IQ2])).T
        Q3 = np.vstack((x[IQ3], y[IQ3], z[IQ3], s[IQ3], d[IQ3])).T
        Q4 = np.vstack((x[IQ4], y[IQ4], z[IQ4], s[IQ4], d[IQ4])).T

        # Sort according to distance
        I1 = np.argsort(Q1[:, 4])
        I2 = np.argsort(Q2[:, 4])
        I3 = np.argsort(Q3[:, 4])
        I4 = np.argsort(Q4[:, 4])

        # Select only the nobs closest
        if len(Q1) >= nobs:

            Q1 = Q1[I1, :]
            Q1 = Q1[:nobs, :]

        else:

            Q1 = Q1[I1, :]
    
        if len(Q2) >= nobs:

            Q2 = Q2[I2, :]
            Q2 = Q2[:nobs, :]

        else:

            Q2 = Q2[I2, :]
    
        if len(Q3) >= nobs:

            Q3 = Q3[I3, :]
            Q3 = Q3[:nobs, :]

        else:

            Q3 = Q3[I3, :]
    
        if len(Q4) >= nobs:

            Q4 = Q4[I4, :]
            Q4 = Q4[:nobs, :]

        else:

            Q14 = Q4[I4, :]

        # Stack the data
        Q14 = np.vstack((Q1, Q2, Q3, Q4))

        # Extract sectored data
        x = Q14[:, 0]
        y = Q14[:, 1]
        z = Q14[:, 2]
        s = Q14[:, 3]
        d = Q14[:, 4]

    # Resolution parameter to meters
    alpha *= 1e3

    # Distance weighting vector using sigma (S) and distance (D)
    w = 1.0 / ((s ** 2) * (1 - (d / alpha) ** 2))

    # Avoid zero weights
    w += 1e-6

    # Prediction at grid node
    zi[i] = np.average(z, weights=w)
    
    # Prediction error - weighted std.dev
    ei[i] = np.sqrt(np.abs(np.average((z - zi[i]) ** 2, weights=w)))

    # Number of obs. in solution
    ni[i] = len(z)

    # Only print every n:th number
    if (i % 500) == 0:
        
        # print progress to terminal
        print str(i+1)+"/"+str(len(zi))

# Save to raster to point file
if otype == "p":

    # Transform coordinates
    (lon, lat) = pyproj.transform(projGrd, projGeo, Xi.ravel(), Yi.ravel())

    # Remove empty grid cells
    I = ~np.isnan(zi)

    # Save data to binary
    if ofile.endswith('.npy'):

        # Save data
        np.save(ofile,np.vstack((lat[I], lon[I], zi[I], ei[I], ni[I])).T)

    # Save data to hdf5
    elif ofile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):

        # Append new columns to original file
        with h5py.File(ofile, 'w') as fo:

            # Save variables
            fo['lat'] = lat[I]
            fo['lon'] = lon[I]
            fo['zi']  = zi[I]
            fo['ei']  = ei[I]
            fo['ni']  = ni[I]

    # Save data to ASCII
    else:

        # Save data
        np.savetxt(ofile,np.vstack((lat[I], lon[I], zi[I], ei[I], ni[I])).T, delimiter=" ", fmt="%8.5f")

# Save to grid file
else:

    # Converte back to arrays
    Zi = np.flipud(zi.reshape(Xi.shape))
    Ei = np.flipud(ei.reshape(Xi.shape))
    Ni = np.flipud(ni.reshape(Xi.shape))

    # Flip coordinates
    Xi = np.flipud(Xi)
    Yi = np.flipud(Yi)

    # Set output names
    OFILE_1 = ofile[:-4] + "_PRED" + ".tif"
    OFILE_2 = ofile[:-4] + "_RMSE" + ".tif"
    OFILE_3 = ofile[:-4] + "_NSOL" + ".tif"

    geotiffwrite(OFILE_1, Xi, Yi, Zi, dx, dy, int(proj), "float")
    geotiffwrite(OFILE_2, Xi, Yi, Ei, dx, dy, int(proj), "float")
    geotiffwrite(OFILE_3, Xi, Yi, Ni, dx, dy, int(proj), "float")

# Print execution time of script
print 'Execution time: ' + str(datetime.now()-startTime)
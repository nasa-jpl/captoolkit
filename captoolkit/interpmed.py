#!/usr/bin/env python
"""
Interpolation of scattered data using the median

The program uses nearest neighbors interpolation and selects data from four
quadrants around the prediction point, with a correlation length provided by
the user.

Provides the possibility of pre-cleaning of the data using a spatial n-sigma
filter before interpolation.

Takes as input a h5df file with needed data in geographical coordinates
and a-priori error if needed. The user provides the wanted projection
using the EPSG projection format.

Output consists of an hdf5 file containing the predictions, rmse and the
number of points used in the prediction, and the epsg number for the
projection.
 
Notes:
    If the error/std.dev is not provided as input the variable name should
    be set a dummy name. Then the error is set as an array of ones.

    If the error/std.dev is provided as input the prediction RMSE is the RSS of
    the a-priori error and the variability of the data used for the
    prediction (if no a-priori error provided the array is set to zero
    before RSS).
 
Example:
    python interpmed.py ifile.h5 ofile.h5 -d 10 10 -n 25 -r 50 -a 25 -p 3031\
        -c 50 10 -v lon lat dhdt dummy
    python interpmed.py ifile.h5 ofile.h5 -d 10 10 -n 25 -r 50 -a 25 -p 3031\
        -c 50 10 -v lon lat dhdt rmse
 
Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
 
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
 
    Jet Propulsion Laboratory, California Institute of Technology
"""

import sys
import pyproj
import numpy as np
import argparse
import h5py
from scipy import stats
from scipy.spatial import cKDTree

def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Construct output grid-coordinates. """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1  # ny
    Ne = int((np.abs(xmax - xmin)) / dx) + 1  # nx
    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)
    return np.meshgrid(xi, yi)


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


# Description of algorithm
des = 'Interpolation of scattered data using the median'

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
        'ifile', metavar='ifile', type=str, nargs='+',
        help='name of input file (h5-format)')

parser.add_argument(
        'ofile', metavar='ofile', type=str, nargs='+',
        help='name of ouput file (h5-format)')

parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m), optional'),
        default=[None],)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid (deg or km)'),
        default=[1, 1],)

parser.add_argument(
        '-n', metavar='nobs', dest='nobs', type=int, nargs=1,
        help=('number of obs. for each quadrant'),
        default=[None],)

parser.add_argument(
        '-r', metavar='radius', dest='radius', type=float, nargs=1,
        help=('cut off distance cutoff in (km)'),
        default=[None],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-c', metavar=('dim','thres'), dest='filter', type=float, nargs=2,
        help=('dimension of filter in km and sigma thres'),
        default=[0,0],)

parser.add_argument(
        '-v', metavar=('x','y','z','s'), dest='vnames', type=str, nargs=4,
        help=('name of vars in the HDF5-file'),
        default=['lon','lat','h_cor','h_rms'],)

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifile = args.ifile[0]
ofile = args.ofile[0]
bbox  = args.bbox
dx    = args.dxy[0] * 1e3
dy    = args.dxy[1] * 1e3
proj  = args.proj[0]
nobs  = args.nobs[0]
dmax  = args.radius[0]
vicol = args.vnames[:]
dxy   = args.filter[0] * 1e3
thres =  args.filter[1]

# Print parameters to screen
print('parameters:')
for p in vars(args).items(): print(p)

print("reading data ...")

# Get variable names
xvar, yvar, zvar, svar = vicol

# Load all 1d variables needed
with h5py.File(ifile, 'r') as fi:

    # Get variables
    lon = fi[xvar][:]
    lat = fi[yvar][:]
    zp  = fi[zvar][:]
    sp  = fi[svar][:] if svar in fi else np.ones(lon.shape)

    # Remove data wiht NaN's
    lon, lat, zp, sp = lon[~np.isnan(zp)],lat[~np.isnan(zp)],\
                zp[~np.isnan(zp)], sp[~np.isnan(zp)]

# Transform coordinates to wanted projection
xp, yp = transform_coord('4326', proj, lon, lat)

# Test for different types of input
if bbox[0] is not None:

    # Extract bounding box elements
    (xmin, xmax, ymin, ymax) = bbox

else:

    # Create bounding box limits
    (xmin, xmax, ymin, ymax) = (xp.min() - 50.*dx), (xp.max() + 50.*dx), \
        (yp.min() - 50.*dy), (yp.max() + 50.*dy)

# Construct the grid
Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

# Flatten prediction grid
xi = Xi.ravel()
yi = Yi.ravel()

print("-> creating KDTree ...")

# Construct cKDTree
TreeP = cKDTree(np.c_[xp, yp])

# Output vectors
zi = np.ones(len(xi)) * np.nan
ei = np.ones(len(xi)) * np.nan
ni = np.ones(len(xi)) * np.nan

# Check if we should filter
if dxy != 0:

    print('-> cleaning data ...')

    # Clean the data in the spatial domain
    zp = spatial_filter(xp.copy(), yp.copy(), zp.copy(), dxy, dxy, sigma=thres)

# Enter prediction loop
for i in range(len(zi)):

    # Find closest observations
    (d, idx) = TreeP.query((xi[i],yi[i]), nobs * 5)

    # Test if closest point to far away
    if np.min(d) > dmax * 1e3: continue

    # Extract data for solution
    x = xp[idx]
    y = yp[idx]
    z = zp[idx]
    s = sp[idx]

    # Test if cell is empty
    if len(z) == 0: continue

    # Compute angle to data points
    theta = (180.0 / np.pi) * np.arctan2(y - yi[i], x - xi[i]) + 180

    # Get index for data in four sectors
    IQ1 = (theta > 00) & (theta < 90)
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
          
    # Predicted value
    zi[i] = np.nanmedian(z)

    # Compute random error
    sigma_r = np.nanstd(z)

    # Compute systematic error
    sigma_s = 0 if np.all(s == 1) else np.nanmean(s)

    # Prediction error at grid node
    ei[i] = np.sqrt(sigma_r ** 2 + sigma_s ** 2)

    # Number of obs. in solution
    ni[i] = len(z)

 # Converte back to arrays
Zi = np.flipud(zi.reshape(Xi.shape))
Ei = np.flipud(ei.reshape(Xi.shape))
Ni = np.flipud(ni.reshape(Xi.shape))

# Flip coordinates
Xi = np.flipud(Xi)
Yi = np.flipud(Yi)

print('-> saving prediction to file...')

# Save data to file
with h5py.File(ofile, 'w') as foo:

    foo['X'] = Xi
    foo['Y'] = Yi
    foo['Z_pred'] = Zi
    foo['Z_rmse'] = Ei
    foo['Z_nobs'] = Ni
    foo['epsg'] = int(proj)

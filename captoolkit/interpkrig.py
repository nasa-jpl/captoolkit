#!/usr/bin/env python
"""
Interpolation of scattered data using ordinary kriging/collocation

The program uses nearest neighbors interpolation and selects data from eight
quadrants around the prediction point and uses a third-order Gauss-Markov
covariance model, with a correlation length defined by the user.

Provides the possibility of pre-cleaning of the data using a spatial n-sigma
filter before interpolation.

Observations with provided noise/error estimates (for each observation) are
added to the diagonal of the covariance matrix if provided. User can also
provide a constant rms-noise added to the diagonal.

Takes as input a h5df file with needed data in geographical coordinates
and a-priori error if needed. The user provides the wanted projection
using the EPSG projection format.

Output consists of an hdf5 file containing the predictions, rmse and the
number of points used in the prediction, and the epsg number for the
projection.

Notes:
    If both the a-priori errors are provided and constant rms all values
    smaller then provided rms is set to this value providing a minimum
    error for the observations.

    To reduce the impact of highly correlated along-track measurements
    (seen as streaks in the interpolated raster) the 'rand' option
    can be used. This randomly samples N-observations in each quadrant
    instead of using the closest data points.

Example:

    python interpkrig.py ifile.h5 ofile.h5 -d 10 10 -n 25 -r 50 -a 25 -p 3031 \
        -c 50 10 -v lon lat dhdt dummy -e 0.1 -m dist

    python interpkrig.py ifile.h5 ofile.h5 -d 10 10 -n 25 -r 50 -a 25 -p 3031 \
        -c 50 10 -v lon lat dhdt rmse -e 0.1 -m rand

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""

import h5py
import pyproj
import argparse
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + proj1)
    proj2 = pyproj.Proj("+init=EPSG:" + proj2)

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
    f_bin = stats.binned_statistic_2d(x, y, z, bins=(Ne, Nn))

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
des = 'Interpolation of scattered data using ordinary kriging/collocation'

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
    'ifile', metavar='ifile', type=str, nargs='+',
    help='name of input file (h5-format)')

parser.add_argument(
    'ofile', metavar='ofile', type=str, nargs='+',
    help='name of ouput file (h5-format)')

parser.add_argument(
    '-b', metavar=('w', 'e', 's', 'n'), dest='bbox', type=float, nargs=4,
    help=('bounding box for geograph. region (deg or m), optional'),
    default=[None], )

parser.add_argument(
    '-d', metavar=('dx', 'dy'), dest='dxy', type=float, nargs=2,
    help=('spatial resolution for grid (deg or km)'),
    default=[1, 1], )

parser.add_argument(
    '-n', metavar='nobs', dest='nobs', type=int, nargs=1,
    help=('number of obs. for each quadrant'),
    default=[None], )

parser.add_argument(
    '-r', metavar='radius', dest='radius', type=float, nargs=1,
    help=('cut off distance cutoff in (km)'),
    default=[None], )

parser.add_argument(
    '-a', metavar='alpha', dest='alpha', type=float, nargs=1,
    help=('correlation length (km)'),
    default=[None], )

parser.add_argument(
    '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
    help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
    default=['3031'], )

parser.add_argument(
    '-c', metavar=('dim', 'thres'), dest='filter', type=float, nargs=2,
    help=('dim. of filter in km and sigma thres'),
    default=[0, 0], )

parser.add_argument(
    '-v', metavar=('x', 'y', 'z', 's'), dest='vnames', type=str, nargs=4,
    help=('name of varibales in the HDF5-file'),
    default=['lon', 'lat', 'h_cor', 'h_rms'], )

parser.add_argument(
    '-e', metavar='sigma', dest='sigma', type=float, nargs=1,
    help=('constant rms noise value'),
    default=[0], )

parser.add_argument(
    '-m', metavar=None, dest='mode', type=str, nargs=1,
    help=('sampling mode: random (rand) or distance (dist).'),
    choices=('rand', 'dist'), default=['dist'], )

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifile = args.ifile[0]
ofile = args.ofile[0]
bbox = args.bbox
dx = args.dxy[0] * 1e3
dy = args.dxy[1] * 1e3
proj = args.proj[0]
nobs = args.nobs[0]
dmax = args.radius[0] * 1e3
alpha = args.alpha[0] * 1e3
sigma = args.sigma[0]
dxy = args.filter[0] * 1e3
thres = args.filter[1]
mode = args.mode[0]
vicol = args.vnames[:]

# Print parameters to screen
print('parameters:')
for p in list(vars(args).items()): print(p)

# Get variable names
xvar, yvar, zvar, svar = vicol

# Load all 1d variables needed
with h5py.File(ifile, 'r') as fi:
    # Get variables
    lon = fi[xvar][:]
    lat = fi[yvar][:]
    zp = fi[zvar][:]
    sp = fi[svar][:] if svar in fi else np.ones(lon.shape)

    # Remove data with NaN's
    lon, lat, zp, sp = lon[~np.isnan(zp)], lat[~np.isnan(zp)], \
                       zp[~np.isnan(zp)], sp[~np.isnan(zp)]

# Transform coordinates to wanted projection
xp, yp = transform_coord('4326', proj, lon, lat)

# Test for different types of input
if bbox[0] is not None:

    # Extract bounding box elements
    xmin, xmax, ymin, ymax = bbox

else:

    # Create bounding box limits
    xmin, xmax, ymin, ymax = (xp.min() - 50. * dx), (xp.max() + 50. * dx), \
                               (yp.min() - 50. * dy), (yp.max() + 50. * dy)

# Construct the grid
Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

# Flatten prediction grid
xi = Xi.ravel()
yi = Yi.ravel()

# Markov-model parameter
a = 0.9132 * alpha

# Signal variance of entire field
c0 = np.nanvar(zp)

# Compute noise variance
crms = sigma * sigma

# Output vectors
zi = np.ones(len(xi)) * np.nan
ei = np.ones(len(xi)) * np.nan
ni = np.ones(len(xi)) * np.nan

# Determine nobs for tree
if mode == 'rand':
    n_quad = 16
else:
    n_quad = 8

# Check if we should filter
if dxy != 0:

    print('-> cleaning data ...')

    # Clean the data in the spatial domain
    zp = spatial_filter(xp.copy(), yp.copy(), zp.copy(), dxy, dxy, sigma=thres)

    # Remove data with NaN's
    xp, yp, zp, sp = xp[~np.isnan(zp)], yp[~np.isnan(zp)], zp[~np.isnan(zp)], \
        sp[~np.isnan(zp)]

print("-> creating KDTree ...")

# Construct cKDTree
TreeP = cKDTree(np.c_[xp, yp])

# Enter prediction loop
for i in range(len(xi)):

    # Find closest observations
    (dr, idx) = TreeP.query((xi[i], yi[i]), nobs * n_quad)

    # Test if closest point to far away
    if (np.min(dr) > dmax) or (len(zp[idx]) < 2): continue

    # Parameters inside cap
    x = xp[idx]
    y = yp[idx]
    z = zp[idx]
    s = sp[idx]

    # Noise handling
    if np.all(sp == 1):

        # Provide all obs. with the same RMS
        c = np.ones(len(x)) * crms

    else:

        # Set all obs. errors < crms to crms
        c = s ** 2
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
    Q1 = np.vstack((x[IQ1], y[IQ1], z[IQ1], c[IQ1], dr[IQ1])).T
    Q2 = np.vstack((x[IQ2], y[IQ2], z[IQ2], c[IQ2], dr[IQ2])).T
    Q3 = np.vstack((x[IQ3], y[IQ3], z[IQ3], c[IQ3], dr[IQ3])).T
    Q4 = np.vstack((x[IQ4], y[IQ4], z[IQ4], c[IQ4], dr[IQ4])).T
    Q5 = np.vstack((x[IQ5], y[IQ5], z[IQ5], c[IQ5], dr[IQ5])).T
    Q6 = np.vstack((x[IQ6], y[IQ6], z[IQ6], c[IQ6], dr[IQ6])).T
    Q7 = np.vstack((x[IQ7], y[IQ7], z[IQ7], c[IQ7], dr[IQ7])).T
    Q8 = np.vstack((x[IQ8], y[IQ8], z[IQ8], c[IQ8], dr[IQ8])).T

    # Sampling strategy
    if mode == 'rand':

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
    Q18 = np.vstack((Q1[I1, :], Q2[I2, :], Q3[I3, :], Q4[I4, :], Q5[I5, :], \
                     Q6[I6, :], Q7[I7, :], Q8[I8, :]))

    # Extract position and data
    xc = Q18[:, 0]
    yc = Q18[:, 1]
    zc = Q18[:, 2]
    cc = Q18[:, 3]

    # Distance from grid node to data
    Dxy = Q18[:, 4]

    # Estimate local median (robust) and local variance of data
    m0 = np.nanmedian(zc)

    # Covariance function for Dxy
    Cxy = c0 * (1 + (Dxy / a) - 0.5 * (Dxy / a) ** 2) * np.exp(-Dxy / a)

    # Compute pair-wise distance 
    Dxx = cdist(list(zip(xc, yc)), list(zip(xc, yc)), "euclidean")

    # Covariance function Dxx
    Cxx = c0 * (1 + (Dxx / a) - 0.5 * (Dxx / a) ** 2) * np.exp(-Dxx / a)

    # Measurement noise matrix
    N = np.eye(len(Cxx)) * np.diag(cc)

    # Matrix solution of Cxy(Cxy + N)^(-1), instead of inverse.
    CxyCxxi = np.linalg.solve((Cxx + N).T, Cxy.T)

    # Predicted value
    zi[i] = np.dot(CxyCxxi, zc) + (1 - np.sum(CxyCxxi)) * m0

    # Predicted error
    ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))

    # Number of data used for prediction    
    ni[i] = len(zc)

# Convert back to arrays
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

#!/usr/bin/env python
"""
Spatio-temporal interpolation of scattered data using an gaussian and
exponential kernel.

The program uses a search radius for interpolation and selects data from four
quadrants around the prediction point, with correlation lengths provided by
the user.

Provides the possibility of pre-cleaning of the data using a spatial n-sigma
filter before interpolation.

Takes as input a h5df file with needed data in geographical coordinates and
a-priori error if needed. The user provides the wanted projection using
the EPSG projection format.

Output consists of an hdf5 file containing the predictions, rmse and the
number of points used in the prediction, the epsg number for the
projection and the time vector.
 
Notes:
    If the error/std.dev is not provided as input the variable name should be
    set a dummy name. Then the error is set as an array of ones.
    If the error/std.dev is provided as input the prediction rmse is the RSS of
    the a-priori error and the variability of the data used for the prediction
    (if no a-priori error provided the array is set to zero before RSS). As
    most altimetry datasets are of monthly resolution the program expects
    that the temporal resolution/correlation is given as a monthly integer
    which is divided by 12 internally: monthly = 1/12 -> input as 1.

Example:
    python interpgaus.py ifile.h5 ofile.h5 -d 10 10 -r 50 -a 25 -p 3031\
        -c 50 10 -v lon lat dh time dummy -s 10
    python interpgaus.py ifile.h5 ofile.h5 -d 10 10 -r 50 -a 25 -p 3031\
        -c 50 10 -v lon lat dh time rmse -s 10
 
Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
 
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
 
    Jet Propulsion Laboratory, California Institute of Technology
"""

import h5py
import pyproj
import numpy as np
import argparse
from scipy import stats
from scipy.spatial import cKDTree
from numba import jit

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


def spatial_filter(x, y, z, dx, dy, sigma=5.0, vmax=100):
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
        foo = np.abs(dh) > sigma*np.nanstd(dh)
    
        # Set to nan-value
        zb[foo] = np.nan
        
        # Replace data
        zo[idx] = zb
    
    # Return filtered array
    return zo

@jit(nopython=True)
def fwavg(w, z):
    return np.nansum(w*z)/np.nansum(w)

@jit(nopython=True)
def fwstd(w, z, zm):
    return np.nansum(w*(z-zm)**2)/np.nansum(w)

@jit(nopython=True)
def make_weights(dr, dt, ad, at):
    ed = np.exp(-(dr ** 2)/(2 * ad ** 2))
    et = np.exp(-(dt ** 2)/(2 * at ** 2))
    return ed, et

@jit(nopython=True)
def square_dist(x, y, xi, yi):
    return np.sqrt((x - xi)**2 + (y - yi)**2)

@jit(nopython=True)
def fast_sort(x):
    return np.argsort(x)

# Description of algorithm
des = 'Spatio-temporal interpolation of irregular data'

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
        '-t', metavar=('tmin','tmax','dt'), dest='time', type=float, nargs=3,
        help=('temporal resolution for grid (months)'),
        default=[-9999, 9999, 1],)

parser.add_argument(
        '-r', metavar='radius', dest='radius', type=float, nargs=1,
        help=('search radius (km)'),
        default=[None],)

parser.add_argument(
        '-a', metavar=('alpha_d','alpha_t'), dest='alpha', type=float, nargs=2,
        help=('spatial and temporal corr. length (km and months)'),
        default=[None,None],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-s', metavar=('n_sample'), dest='n_sample', type=int, nargs=1,
        help=('sample every n:th point in dataset'),
        default=[1],)

parser.add_argument(
        '-c', metavar=('dim','thres','max'), dest='filter', type=float, nargs=3,
        help=('dim. of filter in km, sigma thres and max-value'),
        default=[0,0,9999],)

parser.add_argument(
        '-v', metavar=('x','y','z','t','s'), dest='vnames', type=str, nargs=5,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','h_cor','t_year','h_rms'],)

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifile   = args.ifile[0]
ofile   =  args.ofile[0]
bbox    = args.bbox
dx      = args.dxy[0] * 1e3
dy      = args.dxy[1] * 1e3
proj    = args.proj[0]
dmax    = args.radius[0] * 1e3
alpha_d = args.alpha[0] * 1e3
alpha_t = args.alpha[1] / 12.
vicol   = args.vnames[:]
dxy     = args.filter[0] * 1e3
thres   = args.filter[1]
vmax    = args.filter[2]
tmin    = args.time[0]
tmax    = args.time[1]
tres    = args.time[2]/12.
nsam    = args.n_sample[0]

# Print parameters to screen
print('parameters:')
for p in vars(args).items(): print(p)

print("-> reading data ...")

# Get variable names
xvar, yvar, zvar, tvar, svar = vicol

# Load all 1d variables needed
with h5py.File(ifile, 'r') as fi:

    # Get variables and sub-sample if needed
    lon = fi[xvar][::nsam]
    lat = fi[yvar][::nsam]
    zp  = fi[zvar][::nsam]
    tp  = fi[tvar][::nsam]
    sp  = fi[svar][::nsam] if svar in fi else np.ones(lon.shape)

    # Find all NaNs and do not select them
    no_nan = ~np.isnan(zp)
    
    # Remove data wiht NaN's
    lon, lat, zp, tp, sp = lon[no_nan], lat[no_nan], zp[no_nan],\
                           tp[no_nan], sp[no_nan]
    
# Transform coordinates to wanted projection
xp, yp = transform_coord('4326', proj, lon, lat)

# Test for different types of input
if bbox[0] is not None:

    # Extract bounding box elements
    (xmin, xmax, ymin, ymax) = bbox

else:
    
    # Create bounding box limits
    xmin, xmax, ymin, ymax = xp.min(), xp.max(), yp.min(), yp.max()

# Time vector
ti = np.arange(tmin, tmax + tres, tres)

# Construct the grid
Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

# Shape of grid
Nx, Ny = Xi.shape

# Length of time vector
Nt = len(ti)

# Output vectors
Zi = np.ones((Nt, Nx, Ny)) * np.nan
Ei = np.ones((Nt, Nx, Ny)) * np.nan
Ni = np.ones((Nt, Nx, Ny)) * np.nan

# Check if we should filter
if dxy != 0:

    print('-> cleaning data ...')

    # Global filtering before cleaning 
    i_o = np.abs(zp) < vmax

    # Remove all NaNs 
    xp, yp, zp, tp, sp = xp[i_o], yp[i_o], zp[i_o], tp[i_o], sp[i_o]
    
    # Clean the data in the spatial domain
    zp = spatial_filter(xp.copy(), yp.copy(), zp.copy(), dxy, dxy, sigma=thres)

print("-> creating kdtree ...")

# Construct cKDTree
tree = cKDTree(np.c_[xp, yp])

print('-> interpolating data ...')

import matplotlib.pyplot as plt

# Enter prediction loop
for i in range(int(Nx)):
    for j in range(int(Ny)):

        # Find closest observations for entire stack
        idx = tree.query_ball_point([Xi[i,j], Yi[i,j]], r=dmax)

        # Test if empty
        if len(idx) == 0: continue

        # Extract data for solution
        xt = xp[idx]
        yt = yp[idx]
        zt = zp[idx]
        tt = tp[idx]
        st = sp[idx]

        # Loop trough time 
        for k in range(int(Nt)):

            # Difference in time 
            dt = np.abs(tt - ti[k])
            
            # Distance from center 
            dr = square_dist(xt, yt, Xi[i,j], Yi[i,j])
             
            # Compute the weighting factors
            ed, et = make_weights(dr, dt, alpha_d, alpha_t)
           
            # Combine weights and scale with error
            w = (1. / st ** 2) * ed * et
            
            # Add something small to avoid division by zero
            w += 1e-6
        
            # Predicted value
            zi = fwavg(w, zt)
            
            # Test for signular values 
            if np.abs(zi) < 1e-6: continue
            
            # Compute random error
            sigma_r = fwstd(w, zt, zi)

            # Compute systematic error
            sigma_s = 0 if np.all(st == 1) else np.nanmean(st)

            # Prediction error at grid node
            ei = np.sqrt(sigma_r ** 2 + sigma_s ** 2)

            # Number of obs. in solution
            ni = len(zt)

            # Save data to output
            Zi[k,i,j] = zi
            Ei[k,i,j] = ei
            Ni[k,i,j] = ni

print('-> saving predictions to file...')

# Save data to file
with h5py.File(ofile, 'w') as foo:

    foo['X'] = Xi
    foo['Y'] = Yi
    foo['time'] = ti
    foo['Z_pred'] = Zi
    foo['Z_rmse'] = Ei
    foo['Z_nobs'] = Ni
    foo['epsg'] = int(proj)

#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Program for surface elevation detrending
of satellite and airborne altimetry.

Example:

    python topofit.py /pth/to/file.txt -m grid -d 10 10 -r 1 1 \
            -p 1 -z 10 -t 2005 2015 -e 2010 -l 15 -q 1 -s 10 \
            -j 3031 -c 2 1 3 4 -1 -1 -x 't + 2000'


Created on Wed Apr  1 13:47:37 2015

@author: nilssonj


"""
__version__ = 0.2

import os
import h5py
import pyproj
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gdalconst import *
from datetime import datetime
from scipy.spatial import cKDTree
from statsmodels.robust.scale import mad

import warnings
warnings.filterwarnings("ignore")

# Defaul grid spacing in x and y (km)
DXY = [1, 1]

# Defaul min and max search radius (km)
RADIUS = [1]

# Default min obs within search radius to compute solution
MINOBS = 10

# Default number of iterations for solution
NITER = 5

# Default reference time for solution (yr), None = mean time
TREF = None

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ = 3031

# Default data columns (lon,lat,time,height,error,id)
COLS = ['lon', 'lat', 't_sec', 'h_ellip', 'h_sigma']

# Default expression to transform time variable
EXPR = None

# Defaul maximum value of residuals
DZLIM = 9999

# Default njobs for parallel processing
NJOBS = 1

# Output description of solution
description = ('Program for computing surface elevation residuals from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (ASCII, HDF5 or Numpy)')

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or m)'),
        default=DXY,)

parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
        help=('min and max search radius (km)'),
        default=RADIUS,)

parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations for model solution'),
        default=[NITER],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-e', metavar=('ref_time'), dest='tref', type=float, nargs=1,
        help=('time to reference the solution to (yr), optional'),
        default=[TREF],)

parser.add_argument(
        '-l', metavar=('dzlim'), dest='dzlim', type=float, nargs=1,
        help=('maximum allowed magnitude of residual'),
        default=[DZLIM],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','s'), dest='vnames', type=str, nargs=5,
        help=('name of lon/lat/t/h/s in the HDF5'),
        default=COLS,)

parser.add_argument(
        '-x', metavar=('expr'), dest='expr',  type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[EXPR],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files, optional",
        default=[NJOBS],)

args = parser.parse_args()

# Pass arguments
files = args.files                  # input file(s)
dx    = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy    = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax  = args.radius[0] * 1e3        # min search radius (km -> m)
nlim  = args.minobs[0]              # min obs for solution
niter = args.niter[0]               # number of iterations for solution
tref_ = args.tref[0]                # ref time for solution (d.yr)
proj  = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031)
icol  = args.vnames[:]              # data input cols (x,y,t,h,err,id) [4]
expr  = args.expr[0]                # expression to transform time
njobs = args.njobs[0]               # for parallel processing
dzlim = args.dzlim[0]               # maximum allowd residual values


print 'parameters:'
for p in vars(args).iteritems(): print p

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


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def iterfilt(x, xmin, xmax, tol, thres):
    """Iterativ outlier rejection"""

    tau = 100.0

    xi = x.copy()

    xi[(xi < xmin) & (xi > xmax)] = np.nan

    while tau > tol:

        stdpre = mad_std(xi)

        diff = np.abs(xi - np.nanmean(xi))

        Io = diff > thres * stdpre

        xi[Io] = np.nan

        stdpost = mad_std(xi)

        tau = 100.0 * (stdpre - stdpost) / stdpost

        if tau > tol:

            x[Io] = np.nan

    return x


# Main function for computing parameters
def main(ifile, n=''):
    
    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print 'input file is empty!'
        return
    
    # Start timing of script
    startTime = datetime.now()

    print 'loading data ...'

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print "Input file must be in hdf5-format"
        return
    
    # Input variables
    xvar, yvar, tvar, zvar, svar = icol
    
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:
 
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        time = fi[tvar][:]
        height = fi[zvar][:]
        sigma = fi[svar][:] if svar in fi else np.ones(lon.shape)

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print 'converting lon/lat to x/y ...'

    # Convert into stereographic coordinates
    (x, y) = transform_coord(projGeo, projGrd, lon, lat)

    # Get bbox from data
    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Define time in years
    #time /= (3600. * 24. * 365.25)

    # Apply transformation to time
    if expr:

        time = eval(expr.replace('t', 'time'))

    # Time interval = all data
    t1lim, t2lim = time.min(), time.max()

    # Grid solution - defined by nodes
    (Xi, Yi) = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Flatten prediction grid
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Zip data to vector
    coord = zip(x.ravel(), y.ravel())

    # Construct cKDTree
    print 'building the k-d tree ...'
    Tree = cKDTree(coord)

    # Create output containers
    dh_topo = np.ones(height.shape) * np.nan
    de_topo = np.ones(height.shape) * np.nan
    mi_topo = np.ones(height.shape) * np.nan
    hm_topo = np.ones(height.shape) * np.nan
    sx_topo = np.ones(height.shape) * np.nan
    sy_topo = np.ones(height.shape) * np.nan

    # Enter prediction loop
    print 'predicting values ...'
    for i in xrange(len(xi)):

        # Query the Tree with grid-node coordinates
        idx = Tree.query_ball_point((xi[i], yi[i]), dmax)
        
        # Length of data in search cap
        nobs = len(x[idx])
        
        # Continue to next solution if true
        if (nobs <= nlim): continue

        # Query the Tree with updated centroid
        idx = Tree.query_ball_point((np.median(x[idx]), np.median(y[idx])), dmax)

        # Length of data in search cap
        nobs = len(x[idx])
    
        # Continue to next solution if true
        if (nobs <= nlim): continue

        # Parameters for model-solution
        xcap  = x[idx]
        ycap  = y[idx]
        tcap  = time[idx]
        hcap  = height[idx]
        scap  = sigma[idx]

        # Copy original height vector
        h_org = hcap.copy()

        # Compute variance
        Var = scap * scap

        # Centroid node
        xc = np.median(xcap)
        yc = np.median(ycap)

        # If reference time not given, use mean
        tref = tref_ if tref_ else np.mean(tcap)

        # Design matrix elements
        c0 = np.ones(len(xcap))
        c1 = xcap - xc
        c2 = ycap - yc
        c3 = c1 * c2
        c4 = c1 * c1
        c5 = c2 * c2
        c6 = tcap - tref
        c7 = c6 * c6

        # Weighting factor for solution
        Wcap = 1.0 / Var

        # Topography and linear trend
        Acap = np.vstack((c0, c1, c2, c3, c4, c5, c6)).T
        
        # Model identifier for biquadratic surface
        mi = 1
        
        # Initiate counter
        k = 0

        # Initialize outlier boolean vector
        Io = np.ones(hcap.shape, dtype=bool)

        # Length before editing
        nb = len(hcap)
        
        # Set model order
        if nb <= 7:
            
            # Bilinear topography and trend
            Acap = Acap[:,[0, 1, 2, 6]]
            
            # Model identifier for bilinear surface
            mi = 2

        # Enter iterative solution
        while k <= niter:

            # Remove outlier
            xcap = xcap[Io]
            ycap = ycap[Io]
            tcap = tcap[Io]
            hcap = hcap[Io]
            Acap = Acap[Io]
            Wcap = Wcap[Io]
            
            # Check constrains before solving
            if (len(xcap) == 0): break
            
            # Construct model object
            linear_model = sm.WLS(hcap, Acap, weights=Wcap)

            # Fit the model to the data,
            linear_model_fit = linear_model.fit()

            # Residuals dH = H - A * Cm
            residual = linear_model_fit.resid
            
            # Outlier indexing
            Io = ~np.isnan(iterfilt(residual.copy(), -dzlim, dzlim, 5.0, 3.0))
            
            # Exit loop if no outliers found
            if len(residual[~Io]) < 3: break
            
            # Update counter
            k += 1
            
        # Length after editing
        na = len(hcap)

        # Coefficients
        Cm = linear_model_fit.params
        
        # RMSE of the residuals
        rms_new = mad(residual) / np.sqrt(na)

        # Select all data for editing
        ind = np.ones(de_topo[idx].shape, dtype=bool)

        # Check if already populated
        if np.any(~np.isnan(de_topo[idx])):

            # Determine previous and current noise estimate
            rms_old = de_topo[idx][0].copy()
            
            # Do not overwrite values if true
            if rms_old < rms_new:

                # Do not overwrite data with these indices
                ind = ~np.isnan(de_topo[idx])
        
        # Modelled topography
        if mi == 1:
            
            # Biquadratic surface
            h_model = np.dot(np.vstack((c0, c1, c2, c3, c4, c5)).T, Cm[[0, 1, 2, 3 ,4 ,5]])

        else:
            
            # Bilinear surface
            h_model = np.dot(np.vstack((c0, c1, c2)).T, Cm[[0, 1, 2]])

        # Remove modelled surface topography
        dh_topo[idx] = h_org[ind] - h_model[ind]
        de_topo[idx] = rms_new
        hm_topo[idx] = h_model[ind]
        mi_topo[idx] = mi
        sx_topo[idx] = np.arctan(Cm[1]) * (180 / np.pi)
        sy_topo[idx] = np.arctan(Cm[2]) * (180 / np.pi)

        # Print progress (every N iterations)
        if (i % 100) == 0:
            
            print str(i) + "/" + str(len(xi)) + ' Iterations: '+str(k) + ' Model: ' + str(mi) + ' Nobs: ' + str(nb)

    # Append new columns to original file
    with h5py.File(ifile, 'a') as fi:
        
        # Check if we have varibales in file
        try:
            
            # Save variables
            fi['h_res'] = dh_topo
            fi['h_mod'] = hm_topo
            fi['e_res'] = de_topo
            fi['m_deg'] = mi_topo
            fi['slp_x'] = sx_topo
            fi['slp_y'] = sy_topo

        except:
            
            # Update variables
            fi['h_res'][:] = dh_topo
            fi['h_mod'][:] = hm_topo
            fi['e_res'][:] = de_topo
            fi['m_deg'][:] = mi_topo
            fi['slp_x'][:] = sx_topo
            fi['slp_y'][:] = sy_topo

    # Print some statistics
    print '************************************************************************'
    print('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %.2f %s' %
    ('* Statistics','Mean:',np.nanmedian(dh_topo),'Std.dev:',mad_std(dh_topo),'Min:',
        np.nanmin(dh_topo),'Max:',np.nanmax(dh_topo), 'RMSE:',np.nanmean(de_topo),'*'))
    print '************************************************************************'

    # Print execution time of algorithm
    print 'Execution time: '+ str(datetime.now()-startTime)

if njobs == 1:
    print 'running sequential code ...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

    '''
    from dask import compute, delayed
    from distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=8, threads_per_worker=None,
                          scheduler_port=8002, diagnostics_port=8003)
    client = Client(cluster)  # connect to cluster
    print client

    #values = [delayed(main)(f) for f in files]
    #results = compute(*values, get=client.get)
    values = [client.submit(main, f) for f in files]
    results = client.gather(values)
    '''

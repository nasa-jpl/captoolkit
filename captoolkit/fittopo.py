#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""

Surface topography detrending of satellite and airborne altimetry

Program computes surface elevation residuals, containing only the temporal
component, by removing the static topography.

Depending on the number of observations in each solution one of three models
are used to solve for the topography (1) Bi-quadratic, (2) Bilinear and (3)
the average.

User specifies a grid resolution, search radius and the number of
relocations that should be used to detrend the observations. Inside each
search area the model is centered (relocated) to the centroid of the data,
given the provided number of allowed relocations.

Given the possible overlap between solutions the solution with the smallest
RMS is used and data of poorer quality overwritten.

Notes:
    For mission in reference track configuration a dx = dy = 250 m and a
    search radius of 350 m is appropriate, and less than n=3 relocations is
    usually needed to center the data (depends on search radius)

    This program can be run in parallel to processes several files at the same
    time (tiles or missions etc).

    Good threshold ("-m" option) for switching from biquadratic to bilinear
    model is around 10-15 points.

Example:

    python fittopo.py /path/to/files/*.h5 -v lon lat t_year h_cor \
            -d 1 1 -r 1 -q 3 -i 5 -z 5 -m 15 -k 1 -t 2012 -j 3031 -n 2

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""

import warnings
warnings.filterwarnings("ignore")
import os
import h5py
import pyproj
import argparse
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from scipy.spatial import cKDTree
from statsmodels.robust.scale import mad

# Defaul grid spacing in x and y (km)
DXY = [1, 1]

# Defaul min and max search radius (km)
RADIUS = [1]

# Default min obs within search radius to compute solution
MINOBS = 10

# Default number of iterations for solution
NITER = 5

# Default ref time for solution: 'year' | 'fixed'=full mean t | 'variable'=cap mean t
TREF = 'fixed'

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ = 3031

# Default data columns (lon,lat,time,height,error,id)
COLS = ['lon', 'lat', 't_sec', 'h_cor', 'h_rms']

# Default expression to transform time variable
EXPR = None

# Default order of the surface fit model 
ORDER = 2

# Default numbe rof obs. to change to mean solution
MLIM = 10

# Default njobs for parallel processing of *tiles*
NJOBS = 1

# Maximum slope allowed from the solution, replaced by SLOPE
SLOPE = 1.0

# Output description of solution
description = ('Compute surface elevation residuals '
               'from satellite/airborne altimetry.')

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
        '-q', metavar=('n_reloc'), dest='nreloc', type=int, nargs=1,
        help=('number of relocations for search radius'),
        default=[0],)

parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations for model solution'),
        default=[NITER],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-m', metavar=('mod_lim'), dest='mlim', type=int, nargs=1,
        help=('minimum obs for higher order models'),
        default=[MLIM],)

parser.add_argument(
        '-k', metavar=('mod_order'), dest='order', type=int, nargs=1,
        help=('order of the surface fit model: 1=lin or 2=quad'),
        default=[ORDER],)

parser.add_argument(
        '-t', metavar=('ref_time'), dest='tref', type=str, nargs=1,
        help=('time to reference the solution to: year|fixed|variable'),
        default=[TREF],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h'), dest='vnames', type=str, nargs=4,
        help=('name of lon/lat/t/h/s in the HDF5'),
        default=COLS,)

parser.add_argument(
        '-x', metavar=('expr'), dest='expr',  type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[EXPR],)

parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[NJOBS],)

parser.add_argument(
        '-s', metavar=('slope_lim'), dest='slplim', type=float, nargs=1,
        help="slope limit for x/y direction (deg)",
        default=[SLOPE],)

args = parser.parse_args()

# Pass arguments
files  = args.files                  # input file(s)
dx     = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy     = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax   = args.radius[0] * 1e3        # min search radius (km -> m)
nreloc = args.nreloc[0]              # number of relocations 
nlim   = args.minobs[0]              # min obs for solution
mlim   = args.mlim[0]                # minimum value for parametric verusu men model
niter  = args.niter[0]               # number of iterations for solution
tref_  = args.tref[0]                # ref time for solution (d.yr)
proj   = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031)
icol   = args.vnames[:]              # data input cols (x,y,t,h,err,id) [4]
expr   = args.expr[0]                # expression to transform time
njobs  = args.njobs[0]               # for parallel processing of tiles
order  = args.order[0]               # max order of the surface fit model
slplim = args.slplim[0]              # max allowed surface slope in deg.

print('parameters:')
for p in list(vars(args).items()):
    print(p)

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


def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0,
        min_months=24, max_reloc=3, time=None, height=None):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the center of cell 
    idx = Tree.query_ball_point((x0, y0), r)

    #print 'query #: 1 ( first search )'

    if len(idx) < 2:
        return idx

    if time is not None:
        n_reloc = max_reloc

    if n_reloc < 1:
        return idx
    
    # Relocate center of search radius and query again 
    for k in range(n_reloc):

        # Compute new search location => relocate initial center
        x0_new, y0_new = np.median(x[idx]), np.median(y[idx])

        # Compute relocation distance
        reloc_dist = np.hypot(x0_new-x0, y0_new-y0)

        # Do not allow total relocation to be larger than the search radius
        if reloc_dist > r:
            break

        #print 'query #:', k+2, '( reloc #:', k+1, ')'
        #print 'relocation dist:', reloc_dist

        idx = Tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

        # If time provided, keep relocating until time-coverage is sufficient 
        if time is not None:

            t_b, x_b = binning(time[idx], height[idx], dx=1/12., window=1/12.)[:2]

            print(('months #:', np.sum(~np.isnan(x_b))))

            # If sufficient coverage, exit
            if np.sum(~np.isnan(x_b)) >= min_months:
                break

    return idx


def rlsq(x, y, n=1):
    """ Fit a robust polynomial of n:th deg."""

    # Test solution
    if len(x[~np.isnan(y)]) <= (n + 1):

        if n == 0:
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1, n)) * np.nan
            s = np.nan

        return p, s

    # Empty array
    A = np.empty((0, len(x)))

    # Create counter
    i = 0

    # Determine if we need centering
    if n > 1:
        # Center x-axis
        x -= np.nanmean(x)

    # Special case
    if n == 0:

        # Mean offset
        A = np.ones(len(x))

    else:

        # Make design matrix
        while i <= n:
            # Stack coefficients
            A = np.vstack((A, x ** i))

            # Update counter
            i += 1

    # Test to see if we can solve the system
    try:

        # Robust least squares fit
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=5, tol=0.001)

        # polynomial coefficients
        p = fit.params

        # RMS of the residuals
        s = mad_std(fit.resid)

    except:

        # Set output to NaN
        if n == 0:
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1, n)) * np.nan
            s = np.nan

    return p[::-1], s


# Main function for computing parameters
def main(ifile, n=''):
    
    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print('input file is empty!')
        return
    
    # Start timing of script
    startTime = datetime.now()

    print('loading data ...')

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print("Input file must be in hdf5-format")
        return
    
    # Input variables
    xvar, yvar, tvar, zvar = icol
    
    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        lon = fi[xvar][:]
        lat = fi[yvar][:]
        time = fi[tvar][:]
        height = fi[zvar][:]

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print('converting lon/lat to x/y ...')

    # Convert into stereographic coordinates
    (x, y) = transform_coord(projGeo, projGrd, lon, lat)

    # Get bbox from data
    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Apply transformation to time
    if expr: time = eval(expr.replace('t', 'time'))

    # Overall (fixed) mean time
    t_mean = np.round(np.nanmean(time), 2)

    # Grid solution - defined by nodes
    (Xi, Yi) = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Flatten prediction grid
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Zip data to vector
    coord = list(zip(x.ravel(), y.ravel()))

    # Construct cKDTree
    print('building the k-d tree ...')
    Tree = cKDTree(coord)

    # Create output containers
    dh_topo = np.full(height.shape, np.nan)
    de_topo = np.full(height.shape, 999999)
    mi_topo = np.full(height.shape, np.nan)
    hm_topo = np.full(height.shape, np.nan)
    sx_topo = np.full(height.shape, np.nan)
    sy_topo = np.full(height.shape, np.nan)
    tr_topo = np.full(height.shape, np.nan)
    
    # Set slope limit
    slp_lim = np.tan(np.deg2rad(slplim))
    
    # Enter prediction loop
    print('predicting values ...')
    for i in range(len(xi)):

        x0, y0 = xi[i], yi[i]

        # Get indexes of data within search radius or cell bbox
        idx = get_radius_idx(
                x, y, x0, y0, dmax, Tree, n_reloc=nreloc,
                min_months=18, max_reloc=3, time=None, height=None)

        # Length of data in search cap
        nobs = len(x[idx])
            
        # Check data density
        if (nobs < nlim): continue

        # Parameters for model-solution
        xcap = x[idx]
        ycap = y[idx]
        tcap = time[idx]
        hcap = height[idx]

        # Copy original height vector
        h_org = hcap.copy()

        # Centroid node
        xc = np.median(xcap)
        yc = np.median(ycap)

        # If reference time not given, use fixed or variable mean
        if tref_ == 'fixed':
            tref = t_mean
        elif tref_ == 'variable':
            tref = np.nanmean(tcap)
        else:
            tref = np.float(tref_)

        # Design matrix elements
        c0 = np.ones(len(xcap))
        c1 = xcap - xc
        c2 = ycap - yc
        c3 = c1 * c2
        c4 = c1 * c1
        c5 = c2 * c2
        c6 = tcap - tref

        # Length before editing
        nb = len(hcap)

        # Determine model order
        if order == 2 and nb >= mlim * 2:

            # Biquadratic surface and linear trend
            Acap = np.vstack((c0, c1, c2, c3, c4, c5, c6)).T

            # Model identifier
            mi = 1

        # Set model order
        elif nb >= mlim:

            # Bilinear surface and linear trend
            Acap = np.vstack((c0, c1, c2, c6)).T
            
            # Model identifier
            mi = 2

        else:

            # Model identifier
            mi = 3
        
        # Modelled topography
        if mi == 1:
            
            # Construct model object
            linear_model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')

            # Fit the model to the data,
            linear_model_fit = linear_model.fit(maxiter=niter, tol=0.001)
           
            # Coefficients
            Cm = linear_model_fit.params

            # Biquadratic surface
            h_model = np.dot(np.vstack((c0, c1, c2, c3, c4, c5)).T, Cm[[0, 1, 2, 3, 4, 5]])

            # Compute along and across track slope
            sx = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            sy = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]

            # Mean height
            h_avg = Cm[0]
        
        elif mi == 2:
            
            # Construct model object
            linear_model = sm.RLM(hcap, Acap, M=sm.robust.norms.HuberT(), missing='drop')

            # Fit the model to the data,
            linear_model_fit = linear_model.fit(maxiter=niter, tol=0.001)
           
            # Coefficients
            Cm = linear_model_fit.params
            
            # Bilinear surface
            h_model = np.dot(np.vstack((c0, c1, c2)).T, Cm[[0, 1, 2]])

            # Compute along and across track slope
            sx = np.sign(Cm[1]) * slp_lim if np.abs(Cm[1]) > slp_lim else Cm[1]
            sy = np.sign(Cm[2]) * slp_lim if np.abs(Cm[2]) > slp_lim else Cm[2]

            # Mean height
            h_avg = Cm[0]
    
        else:
                        
            # Mean surface from median
            h_avg = np.median(hcap)

            # Compute distance estimates from centroid
            s_dx = (xcap - xc) + 1e-3
            s_dy = (ycap - yc) + 1e-3

            # Center surface height
            dh_i = h_org - h_avg
        
            # Compute along-track slope
            px, rms_x = rlsq(s_dx, dh_i, 1)
            py, rms_x = rlsq(s_dy, dh_i, 1)

            # Set along-track slope
            s_x = 0 if np.isnan(px[0]) else px[0]
                
            # Set across-track slope to zero
            s_y = 0 if np.isnan(py[0]) else py[0]
            
            # Compute along and across track slope
            sx = np.sign(s_x) * slp_lim if np.abs(s_x) > slp_lim else s_x
            sy = np.sign(s_y) * slp_lim if np.abs(s_y) > slp_lim else s_y
            
            # Compute the surface height correction
            h_model = h_avg + (sx * s_dx) + (sy * s_dy)

        # Compute full slope
        slope = np.arctan(np.sqrt(sx**2 + sy**2)) * (180 / np.pi)

        # Compute residual
        dh = h_org - h_model

        # Number of observations
        na = len(dh)

        # RMSE of the residuals
        RMSE = mad(dh) / np.sqrt(na)

        # Overwrite errors
        iup = RMSE < de_topo[idx]

        # Create temporary variables
        dh_cap = dh_topo[idx].copy()
        de_cap = de_topo[idx].copy()
        hm_cap = hm_topo[idx].copy()
        mi_cap = mi_topo[idx].copy()
        tr_cap = tr_topo[idx].copy()

        # Update variables
        dh_cap[iup] = dh[iup]
        de_cap[iup] = RMSE
        hm_cap[iup] = h_avg 
        mi_cap[iup] = mi
        tr_cap[iup] = tref
        
        # Update with current solution
        dh_topo[idx] = dh_cap
        de_topo[idx] = de_cap
        hm_topo[idx] = hm_cap
        mi_topo[idx] = mi_cap
        tr_topo[idx] = tr_cap
        sx_topo[idx] = np.arctan(sx) * (180 / np.pi)
        sy_topo[idx] = np.arctan(sy) * (180 / np.pi)

        # Print progress (every N iterations)
        if (i % 100) == 0:

            # Print message every i:th solution
            print(('%s %i %s %2i %s %i %s %03d %s %.3f %s %.3f' % \
                    ('#',i,'/',len(xi),'Model:',mi,'Nobs:',nb,'Slope:',\
                    np.around(slope,3),'Residual:',np.around(mad_std(dh),3))))

    # Print percentage of not filled
    print(('Total NaNs (percent): %.2f' % \
            (100 * float(len(dh_topo[np.isnan(dh_topo)])) / float(len(dh_topo)))))

    # Print percentage of each model
    one = np.sum(mi_topo == 1)
    two = np.sum(mi_topo == 2)
    tre = np.sum(mi_topo == 3)
    N = float(len(mi_topo))

    print(('Model types (percent): 1 = %.2f, 2 = %.2f, 3 = %.2f' % \
            (100 * one/N, 100 * two/N, 100 * tre/N)))

    # Append new columns to original file
    with h5py.File(ifile, 'a') as fi:

        # Check if we have variables in file
        try:
            
            # Save variables
            fi['h_res'] = dh_topo
            fi['h_mod'] = hm_topo
            fi['e_res'] = de_topo
            fi['m_deg'] = mi_topo
            fi['t_ref'] = tr_topo
            fi['slp_x'] = sx_topo
            fi['slp_y'] = sy_topo

        except:
            
            # Update variables
            fi['h_res'][:] = dh_topo
            fi['h_mod'][:] = hm_topo
            fi['e_res'][:] = de_topo
            fi['m_deg'][:] = mi_topo
            fi['t_ref'][:] = tr_topo
            fi['slp_x'][:] = sx_topo
            fi['slp_y'][:] = sy_topo

    # Rename file
    os.rename(ifile, ifile.replace('.h5', '_TOPO.h5'))

    # Print some statistics
    print(('*' * 75))
    print(('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' % \
        ('Statistics',
         'Mean:', np.nanmedian(dh_topo),
         'Std.dev:', mad_std(dh_topo),
         'Min:', np.nanmin(dh_topo),
         'Max:', np.nanmax(dh_topo),
         'RMSE:', np.nanmedian(de_topo[dh_topo!=999999]),)))
    print(('*' * 75))
    print('')

    # Print execution time of algorithm
    print(('Execution time: '+ str(datetime.now()-startTime)))

if njobs == 1:
    print('running sequential code ...')
    [main(f, n) for n,f in enumerate(files)]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
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

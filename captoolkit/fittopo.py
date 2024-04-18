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

An a-priori DEM can be proivded to perform intial detrending of the data where
fittopo.py then removes any residual toporaphy and references the data. DEM must
have the same projection as provided by "-j" option.

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
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from statsmodels.robust.scale import mad
from altimutils import tiffread
from altimutils import interp2d
from altimutils import make_grid
from altimutils import transform_coord
from altimutils import mad_std
from altimutils import binning
from altimutils import median_filter
from altimutils import lstsq

# Defaul grid spacing in x and y (km)
DXY = [1, 1]

# Defaul min and max search radius (km)
RADIUS = [1, 0.5]

# Default min obs within search radius to compute solution
MINOBS = 3

# Default number of iterations for solution
NITER = 5

# Default ref time for sol: user provided or mean of data in radius
TREF = None

# Default time limit: use all data
TLIM = 0

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ = 3031

# Default data columns (lon,lat,time,height,error,id)
COLS = ['lon', 'lat', 't_year', 'h_elv']

# Default expression to transform time variable
EXPR = None

# Default order of the surface fit model
ORDER = 2

# Default numbe rof obs. to change to mean solution
MLIM = 10

# Default njobs for parallel processing of *tiles*
NJOBS = 1

# Default outlier filter settings
NSIGMA, THRES = [None, None]

# Default for  DEM file
DEM = [None]

# Output description of solution
description = ('Compute surface elevation residuals '
               'from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or km)'),
        default=DXY,)

parser.add_argument(
        '-r', metavar=('rmax','rcor'), dest='radius', type=float, nargs=2,
        help=('max search radius and corr. length (km)'),
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
        help=('ref. time for fit (default mean of solution)'),
        default=[TREF],)

parser.add_argument(
        '-l', metavar=('tlim'), dest='tlim', type=float, nargs=1,
        help=('min time span of solution (e.g reject single orbits)'),
        default=[TLIM],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h'), dest='vnames', type=str, nargs=4,
        help=('name of lon/lat/t/h in the HDF5'),
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
        '-e', metavar=('nsigma','thres'), dest='filter', type=float, nargs=2,
        help="Number of std.dev and cutoff values to filter data",
        default=[NSIGMA, THRES],)

parser.add_argument(
        '-p', dest='pshow', action='store_true',
        help=('print diagnostic information to terminal'),
        default=False)

args = parser.parse_args()

# Pass arguments
files  = args.files                  # input file(s)
dx     = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy     = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
dmax   = args.radius[0] * 1e3        # max search radius (km -> m)
dcor   = args.radius[1] * 1e3        # correlation length (km-> m)
nreloc = args.nreloc[0]              # number of relocations
nlim   = args.minobs[0]              # min obs for solution
nmod   = args.mlim[0]                # minimum value for changing model
niter  = args.niter[0]               # number of iterations for solution
tref_  = args.tref[0]                # ref time for solution (d.yr)
proj   = args.proj[0]                # EPSG number (GrIS=3413, AnIS=3031)
icol   = args.vnames[:]              # data input cols (x,y,t,h)
expr   = args.expr[0]                # expression to transform time
njobs  = args.njobs[0]               # for parallel processing of tiles
order  = args.order[0]               # max order of the surface fit model
diag   = args.pshow                  # print diagnostics to terminal
nsigma = args.filter[0]              # number of std.dev's
thres  = args.filter[1]              # cutoff value for filter
dtlim  = args.tlim[0]                # minimum time span of data

print('parameters:')
for p in list(vars(args).items()):
    print(p)

def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0,
        min_months=24, max_reloc=3, time=None, height=None):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the center of cell
    idx = Tree.query_ball_point((x0, y0), r)

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

        # Query the KD-tree at location
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


# Main function
def main(ifile, n=''):

    # Use trend in model
    set_use = 1

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print('-> Input file is empty!')
        return

    # Start timing of script
    startTime = datetime.now()

    print('-> Loading data ...')

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print("-> Input file must be in hdf5-format")
        return

    # Input variables
    xvar, yvar, tvar, zvar = icol

    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        lon = fi[xvar][:]
        lat = fi[yvar][:]
        time = fi[tvar][:]
        height = fi[zvar][:]

    print('-> Converting lon/lat to x/y ...')

    # Convert into stereographic coordinates
    (x, y) = transform_coord('4326', proj, lon, lat)

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
    print('-> Building the KD-tree ...')
    Tree = cKDTree(coord)

    # Create output containers
    dh_topo = np.full(height.shape, np.nan)
    de_topo = np.full(height.shape, 999999.)
    mi_topo = np.full(height.shape, np.nan)
    hm_topo = np.full(height.shape, np.nan)
    sx_topo = np.full(height.shape, np.nan)
    sy_topo = np.full(height.shape, np.nan)
    tr_topo = np.full(height.shape, np.nan)

    # Enter prediction loop
    print('-> Predicting values ...')

    # Loop through the grid
    for i in range(len(xi)):

        # Get indexes of data within search radius or cell bbox
        idx = get_radius_idx(x, y, xi[i], yi[i], dmax, Tree, n_reloc=nreloc)

        # Length of data in search cap
        nobs = len(time[idx])

        # Check data density
        if (nobs < nlim):
            continue

        # Get time window
        trad = time[idx]

        # Get time-span of cap
        tmin, tmax = trad.min(), trad.max()

        # Break if enough data avalibale
        if (tmax - tmin) < dtlim:
            continue

        # Parameters for model-solution
        xcap = x[idx]
        ycap = y[idx]
        tcap = time[idx]
        hcap = height[idx]

        # Find centroid of data inside cap
        x0 = np.median(xcap)
        y0 = np.median(ycap)

        # Determine time span
        tmin, tmax = tcap.min(), tcap.max()

        # Reject solution if to short time span
        if (tmax - tmin) < dtlim:
            continue

        # Copy original height vector
        h_org = hcap.copy()

        # Find centroid of data inside cap
        xc = np.median(xcap)
        yc = np.median(ycap)

        # Set reference time
        if tref_ is not None:
            tref = np.float(tref_)
        else:
            tref = 0
            set_use = 0

        # Design matrix elements
        c0 = np.ones(len(xcap))
        c1 = xcap - xc
        c2 = ycap - yc
        c3 = c1 * c2
        c4 = c1 * c1
        c5 = c2 * c2
        c6 = (tcap - tref) * set_use

        # Length before editing
        nb = len(hcap)

        # Bilinear surface and linear trend
        Acap = np.vstack((c6, c0, c1, c2)).T

        # Model identifier
        mi = 2

        # Design matrix - Quadratic
        if nobs > nmod and order > 1:

            # Biquadratic surface and linear trend
            Acap = np.vstack((c6, c0, c1, c2, c3, c4, c5)).T

            # Model identifier
            mi = 1

        # Test for weighted resolution
        if dcor > 0:

            # Distance for estimation point
            dr = np.sqrt((xcap - x0)**2 + (ycap - y0)**2)

            # Gaussian weights - distance from node
            wcap = 1.0 / (1.0 + (dr / dcor)**2)

        else:

            # Don't use weights
            wcap = None
            
        # Solve least-squares iterativly
        x_hat, e_hat, i_bad = lstsq(Acap, hcap, w=wcap, \
                            n_iter=niter, n_sigma=nsigma)
            
        # Model values for topography only
        h_mod = np.dot(Acap[:,1:], x_hat[1:])

        # Slope x/y direction
        sx, sy = x_hat[2], x_hat[3]

        # Intercept value and error
        h0 = x_hat[1]

        # Compute slope
        slope = np.arctan(np.sqrt(sx**2 + sy**2)) * (180 / np.pi)

        # Compute topographical residuals
        dh = h_org - h_mod

        # Number of observations
        na = len(dh)

        # RMSE of the residuals
        RMSE = mad_std(dh)

        # Remove outliers from residuals obtained from model
        if nsigma is not None:
            dh[i_bad] = np.nan

        # Remove residuals above threshold
        if thres is not None:
            dh[np.abs(dh) > thres] = np.nan
            RMSE = mad_std(dh)
            if np.isnan(RMSE):
                continue
            if RMSE > thres:
                continue

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
        hm_cap[iup] = h0
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
        if (i % 100) == 0 and diag is True:
            # Print message every i:th solution
            print(('%s %i %s %2i %s %i %s %03d %s %.3f %s %.3f' % \
                    ('#',i,'/',len(xi),'Model:',mi,'Nobs:',nb,'Slope:',\
                    np.around(slope,3),'Residual:',np.around(RMSE,3))))

    """
    dh_topo = spatial_filter(x, y, dh_topo.copy(), dx=10e3, dy=10e3, n_sigma=3)
    plt.figure()
    plt.scatter(x,y,s=1, c=dh_topo,cmap='coolwarm_r')
    xbb, ybb = binning(time.copy(),dh_topo.copy(),window=3./12,dx=1./12, median=True)[0:2]
    plt.figure()
    plt.plot(xbb,ybb,'-o')
    p0 = np.polyfit(xbb,ybb,1)
    plt.title(p0*100)
    plt.show()
    """

    # Print percentage of not filled
    print(('Total NaNs (percent): %.2f' % \
            (100 * float(len(dh_topo[np.isnan(dh_topo)])) /\
             float(len(dh_topo)))))

    # Print percentage of each model
    one = np.sum(mi_topo[~np.isnan(dh_topo)] == 1)
    two = np.sum(mi_topo[~np.isnan(dh_topo)] == 2)
    tre = np.sum(mi_topo[~np.isnan(dh_topo)] == 3)

    # Total number of data
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
    if ifile.find('TOPO') < 0:
        os.rename(ifile, ifile.replace('.h5', '_TOPO.h5'))

    # Print some statistics
    print(('*' * 75))
    print(('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' % \
        ('Statistics',
         'Mean:',    np.nanmedian(dh_topo),
         'Std.dev:', mad_std(dh_topo),
         'Min:',     np.nanmin(dh_topo),
         'Max:',     np.nanmax(dh_topo),
         'RMSE:',    np.nanmedian(de_topo[dh_topo!=999999]),)))
    print(('*' * 75))
    print('')

    # Print execution time of algorithm
    print(('Execution time: '+ str(datetime.now()-startTime)))

if njobs == 1:
    print('running sequential code ...')
    [main(f, n) for n,f in enumerate(files)]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) \
            for n, f in enumerate(files))

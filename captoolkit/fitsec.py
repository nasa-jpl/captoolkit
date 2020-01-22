#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Compute surface height changes from satellite and airborne altimetry.

Example:
    python secfit.py /path/to/files/*.h5 -v lon lat t_year h_cor None None h_bs \
        -m g -d 1 1 -r 1 3 -a 0.5 -q 2 -z 50 -f fixed -s 1 -p 2 -n 16
"""
__version__ = 0.3

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

# Default output file name, None = same as input
OUTFILE = None

# Standard name for variables
VNAMES = ['lon', 'lat', 't_year', 'h_cor', 'm_rms','m_id','h_bs']

# Default solution mode
MODE = 'g'

# Default geographic domain for solution, [] = defined by data
BBOX = None

# Defaul grid spacing in x and y (km)
DXY = [2, 2]

# Defaul min and max search radius (km)
##NOTE: The effective resolution will be: res_param * accepted_radius
RADIUS = [2, 3]

# Default resolution param for weighting function (km)
# (a scaling factor applied to radius: res_param * search_rad)
RESPARAM = 0.75

# Default no relocations in grid solution => regular grid
NRELOC = 0

# Default min obs within search radius to compute solution
MINOBS = 25

# Min length of time series for solution
MINMONTHS = 6.

# Default number of iterations for solution
NITER = 5

# Apply 5-sigma filter to data before fit
SIGMAFILT = True

# Default time interval for solution [yr1, yr2], [] = defined by data
TSPAN = []

# Default reference time for solution (yr)
TREF = 'fixed'

# Default |dh/dt| limit accept estimate (m/yr)
DHDTLIM = 15

# Default time-span limit to accept estimate (yr)
DTLIM = 0.1

# Default ID for solution if merging (0=SIN, 1=LRM)
IDMISSION = 0

##DEPRECATED: Not being used!
# Default |residual| limit to accept estimate (m)
RESIDLIM = 100

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ_OBS = 3031

# Default expression to transform time variable
EXPR = None

# Default njobs for parallel processing
NJOBS = 1

# Default time resolution of binned time series (months)
TSTEP = 1.0

# Order of design matrix
ORDER = 2

# Output description of solution
description = ('Computes robust surface-height changes '
               'from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (ASCII, HDF5 or Numpy)')
parser.add_argument(
        '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
        help='output file name, default same as input',
        default=[OUTFILE],)
parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('prediction mode: (p)oint or (g)rid'),
        choices=('p', 'g'), default=[MODE],)
parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m)'),
        default=BBOX,)
parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or m)'),
        default=DXY,)
parser.add_argument(
        '-r', metavar=('r_min','r_max'), dest='radius', type=float, nargs=2,
        help=('min and max search radius (km)'),
        default=RADIUS,)
parser.add_argument(
        '-a', metavar='res_param', dest='resparam', type=float, nargs=1,
        help=('resolution param for weighting function (km)'),
        default=[RESPARAM],)
parser.add_argument(
        '-q', metavar=('n_reloc'), dest='nreloc', type=int, nargs=1,
        help=('number of relocations for search radius: 0 => grid'),
        default=[NRELOC],)
parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations to solve model'),
        default=[NITER],)
parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[MINOBS],)
parser.add_argument(
        '-t', metavar=('t_min','t_max'), dest='tspan', type=float, nargs=2,
        help=('min and max time for solution (yr)'),
        default=TSPAN,)
parser.add_argument(
        '-f', metavar=('ref_time'), dest='tref', type=str, nargs=1,
        help=('time to reference the solution to: year|fixed|variable'),
        default=[TREF],)
parser.add_argument(
        '-l', metavar=('dhdt_lim'), dest='dhdtlim', type=float, nargs=1,
        help=('discard estimate if |dh/dt| > dhdt_lim (m/yr)'),
        default=[DHDTLIM],)
parser.add_argument(
        '-k', metavar=('dt_lim'), dest='dtlim', type=float, nargs=1,
        help=('discard estimates if data-span < dt_lim (yr)'),
        default=[DTLIM],)
parser.add_argument(
        '-s', metavar=('t_step'), dest='tstep', type=float, nargs=1,
        help=('time resolution of binned time series (months)'),
        default=[TSTEP],)
parser.add_argument(
        '-u', metavar=('id_mission'), dest='idmission', type=int, nargs=1,                 ##FIXME: Remove this!!!
        help=('reference id for merging (0=sin, 1=lrm)'),
        default=[IDMISSION],)
parser.add_argument(
        '-w', metavar=('resid_lim'), dest='residlim', type=float, nargs=1,
        help=('discard residual if |residual| > resid_lim (m)'),
        default=[RESIDLIM],)
parser.add_argument(
        '-j', metavar=('epsg_num'), dest='projo', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ_OBS)],)
parser.add_argument(
        '-v', metavar=('x','y','t','h','s','i','c'), dest='vnames', type=str, nargs=7,
        help=('name of varibales in the HDF5-file'),
        default=[VNAMES],)
parser.add_argument(
        '-x', metavar=('expr'), dest='expr',  type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000')",
        default=[EXPR],)
parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files",
        default=[NJOBS],)
parser.add_argument(
        '-p', metavar=None, dest='model', type=int, nargs=1,
        help=('select design matrix (order of model fit)'),
        choices=(0,1,2,3), default=[ORDER],)
args = parser.parse_args()

# Pass arguments
mode = args.mode[0]                 # prediction mode: point or grid solution
files = args.files                  # input file(s)
ofile = args.ofile[0]               # output directory
bbox_ = args.bbox                   # bounding box EPSG (m) or geographical (deg)
dx = args.dxy[0] * 1e3              # grid spacing in x (km -> m)
dy = args.dxy[1] * 1e3              # grid spacing in y (km -> m)
tstep_ = args.tstep[0]              # time spacing in t (months)
dmin = args.radius[0] * 1e3         # min search radius (km -> m)
dmax = args.radius[1] * 1e3 + 1e-4  # max search radius (km -> m)
dres_ = args.resparam[0]            # resolution param for weighting func [1]
nreloc = args.nreloc[0]             # number of relocations 
nlim = args.minobs[0]               # min obs for solution
niter = args.niter[0]               # number of iterations for solution
tspan = args.tspan                  # min/max time for solution (d.yr)
tref_ = args.tref[0]                # ref time for solution (d.yr)
dtlim = args.dtlim[0]               # min time difference needed for solution
dhlim = args.dhdtlim[0]             # discard estimate if |dh/dt| > value (m)
nmidx = args.idmission[0]           # id to tie the solution to if merging [3]
slim = args.residlim[0]             # remove residual if |resid| > value (m)
projo = args.projo[0]               # EPSG number (GrIS=3413, AnIS=3031) for OBS
expr = args.expr[0]                 # expression to transform time
njobs = args.njobs[0]               # for parallel processing
model = args.model[0]               # model order lin=trend+accel, biq=linear+topo
names = args.vnames[:]              # Name of hdf5 parameters of interest

print('parameters:')
for p in list(vars(args).items()): print(p)

##NOTE
# [1] This defines the shape (correlation length) of the
# weighting function inside the search radius.
# [2] For Cryosat-2 only (to merge SIN and LRM mode).
# [3] ID for different mode data: 0=SIN, 1=LRM.
# [4] If err and id cols = -1, then they are not used.


def binning(x, y, xmin=None, xmax=None, dx=1/12., 
            window=3/12., interp=False, median=False):
    """Time-series binning (w/overlapping windows).

    Args:
        x,y: time and value of time series.
        xmin,xmax: time span of returned binned series.
        dx: time step of binning.
        window: size of binning window.
        interp: interpolate binned values to original x points.
    """
    if xmin is None: xmin = np.nanmin(x)
    if xmax is None: xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax+dx, dx)      # time steps
    bins = [(ti, ti+window) for ti in steps]  # bin limits

    N = len(bins)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in range(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0: continue

        ybv = y[idx]
        xbv = x[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1+t2)
        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        yb = np.interp(x, xb, yb)
        eb = np.interp(x, xb, eb)
        sb = np.interp(x, xb, sb)
        xb = x

    return xb, yb, eb, nb, sb


def detrend_binned(x, y, order=1, window=3/12.):
    """ Bin data (Med), compute trend (OLS) on binned, detrend original data. """
    x_b, y_b = binning(x, y, median=True, window=window, interp=False)[:2]
    i_valid = ~np.isnan(y_b) & ~np.isnan(x_b)
    x_b, y_b = x_b[i_valid], y_b[i_valid]
    try:
        coef = np.polyfit(x_b, y_b, order)
        y_trend = np.polyval(coef, x)  ##NOTE: Eval on full time
    except:
        y_trend = np.zeros_like(y)
    return y-y_trend, y_trend


def sigma_filter(x, y, order=1, window=3/12., n_iter=3, n_sigma=3):
    """ Detrend (binned) data, remove 3 sigma from residual, repeat. """
    y_filt, y_res = y.copy(), y.copy()
    for _ in range(n_iter):
        y_res, y_trend = detrend_binned(x, y_res, order=order, window=window)
        idx, = np.where(np.abs(y_res) > mad_std(y_res)*n_sigma)  # find where above n_sigma
        if len(idx) == 0: break  # if no data to filter, stop iterating
        y_res[idx] = np.nan
        if np.sum(~np.isnan(y_res)) < 10: break  ##NOTE: Arbitrary min obs
    y_filt[np.isnan(y_res)] = np.nan    
    return y_filt


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Construct output grid-coordinates. """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1  # grid dimensions
    Ne = int((np.abs(xmax - xmin)) / dx) + 1
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


def get_bbox(fname):
    """Extract bbox info from file name."""
    fname = fname.split('_')  # fname -> list
    i = fname.index('bbox')
    return list(map(float, fname[i+1:i+5]))  # m


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


##FIXME: This is temporary! Need to find a better option
def chisquared(model):
    return mad_std(model.resid)/np.sqrt(model.df_model)


def rsquared(model):
    return model.rsquared_adj


def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the node
    idx = Tree.query_ball_point((x0, y0), r)

    #print 'query #: 1 ( first search )'

    reloc_dist = 0.

    # Either no relocation or not enough points to do relocation
    if n_reloc < 1 or len(idx) < 2: return idx, reloc_dist

    # Relocate center of search radius and query again 
    for k in range(n_reloc):

        # Compute new search location => relocate initial center
        x0_new, y0_new = np.median(x[idx]), np.median(y[idx])

        # Compute relocation distance
        reloc_dist = np.hypot(x0_new-x0, y0_new-y0)

        # Do not allow total relocation to be larger than the search radius
        if reloc_dist > r: break

        #print 'query #:', k+2, '( reloc #:', k+1, ')'
        #print 'relocation dist:', reloc_dist

        # Query from the new location
        idx = Tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1: break

    return idx, reloc_dist


def n_months(tc, hc, tstep=1/12.):
    """ Bin at monthly intervals to check temporal sampling => nmonths, tspan """
    t_b, h_binned = binning(tc, hc, dx=tstep, window=tstep)[:2]
    return sum(~np.isnan(h_binned)), np.nanmax(tc)-np.nanmin(tc)


def is_empty(ifile):
    """If file is empty/corruted, return True."""
    try:
        with h5py.File(ifile, 'r') as f: return not bool(list(f.keys()))
    except:
        return True


# Main function for computing parameters
def main(ifile, n='', robust_fit=True, n_iter=niter):
    
    # Check for empty file
    if is_empty(ifile):
        print(('SKIP FILE: EMPTY OR CORRUPTED FILE:', ifile))
        return

    # Start timing of script
    startTime = datetime.now()

    print('loading data ...')

    xvar, yvar, tvar, zvar, svar, ivar, cvar = names

    with h5py.File(ifile, 'r') as fi:
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        time = fi[tvar][:]
        height = fi[zvar][:]
        sigma = fi[svar][:] if svar in fi else np.ones(lon.shape)
        id = fi[ivar][:] if ivar in fi else np.ones(lon.shape) * nmidx
        cal = fi[cvar][:] if cvar in fi else np.zeros(lon.shape)

    # Filter in time
    if 1:
        i_time, = np.where( (time > 1993.972) & (time < 1995.222) )       ##NOTE: To remove ERS-1 GM
        if len(i_time) > 0: height[i_time] = np.nan


    ##NOTE: Filter data based on 'cal' but DO NOT REMOVE NANs!
    if sum(cal) != 0:
        cal[np.isnan(cal)] = 0.  # keep values w/o correction
        height -= cal  # correct absolute H for bs

    # Filter NaNs
    if 1:
        i_valid = ~np.isnan(height)
        lon = lon[i_valid]
        lat = lat[i_valid]
        time = time[i_valid]
        height = height[i_valid]
        sigma = sigma[i_valid]
        id = id[i_valid]
        cal = cal[i_valid]

    projGeo = '4326'  # EPSG number for lon/lat proj
    projGrd = projo   # EPSG number for grid proj

    print('converting lon/lat to x/y ...')

    # If no bbox was given
    if bbox_ is None:
        try:
            bbox = get_bbox(ifile)  # Try reading bbox from file name
        except:
            bbox = None
    else:
        bbox = bbox_

    # Get geographic boundaries + max search radius
    if bbox:
        # Extract bounding box
        xmin, xmax, ymin, ymax = bbox

        # Transform coordinates
        x, y = transform_coord(projGeo, projGrd, lon, lat)

        # Select data inside bounding box
        Ig = (x >= xmin - dmax) & (x <= xmax + dmax) & (y >= ymin - dmax) & (y <= ymax + dmax)

        # Check bbox for obs.
        if len(x[Ig]) == 0:
	    print(('SKIP FILE: NO DATA POINTS INSIDE BBOX:', ifile))
            return
            
        print(('Number of obs. edited by bbox!', 'before:', len(x), 'after:', len(x[Ig])))

        # Only select wanted data
        x = x[Ig]
        y = y[Ig]
        id = id[Ig]
        time = time[Ig]
        height = height[Ig]
        sigma = sigma[Ig]
    else:
        # Convert into stereographic coordinates
        x, y = transform_coord(projGeo, projGrd, lon, lat)

        # Get bbox from data
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    # Apply transformation to time
    if expr: time = eval(expr.replace('t', 'time'))

    # Define time interval of solution
    if tspan:
        # Time interval = given time span
        t1lim, t2lim = tspan

        # Select only observations inside time interval
        Itime = (time > t1lim) & (time < t2lim)

        # Keep only data inside time span
        x = x[Itime]
        y = y[Itime]
        id = id[Itime]
        time = time[Itime]
        height = height[Itime]
        sigma = sigma[Itime]
    else:
        # Time interval = all data
        t1lim, t2lim = time.min(), time.max()

    if mode == 'p':
        # Point solution - all points
        xi, yi = np.copy(x), np.copy(y)
    else:
        # Grid solution - defined by nodes
        Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

        xi, yi = Xi.ravel(), Yi.ravel() 
        coord = list(zip(x.ravel(), y.ravel()))

        print('building the k-d tree ...')
        Tree = cKDTree(coord)

    # Overall (fixed) mean time
    t_mean = np.round(np.nanmean(time), 2)

    # Number of nodes
    nodes = len(xi)

    # Initialize bias param
    bias = np.ones(lon.shape) * np.nan

    # Temporal resolution: months -> years
    tstep = tstep_ / 12.0

    # Expected max number of months in time series
    months = len(np.arange(t1lim, t2lim+tstep, tstep))
    M = 5

    # Create output containers (data matrix)
    DATA0 = np.full((nodes, 21), np.nan)
    DATA1 = np.full((nodes, months+M), np.nan)
    DATA2 = np.full((nodes, months+M), np.nan)

    # Search radius array (dmax is slightly increased by 1e-4)
    dr = np.arange(dmin, dmax, 500)

    # Enter prediction loop
    print('predicting values ...')
    for i in range(len(xi)):

        xc, yc = xi[i], yi[i]  # Center coordinates

        # Loop through search radii
        for rad in dr:

            # Get indices of data within search radius (after relocation)
            i_cell, reloc_dist = get_radius_idx(x, y, xc, yc, rad, Tree, n_reloc=nreloc)

            if len(i_cell) < nlim: continue  # use larger radius

            tcap, hcap  = time[i_cell], height[i_cell] 

            Nb = sum(~np.isnan(hcap))  # length before editing

            # 3-sigma filter 
            if SIGMAFILT:
                #hcap = sigma_filter(tcap, hcap, order=1, n_sigma=3, n_iter=3)  ##NOTE: It removes too much!!!
                hcap[np.abs( hcap - np.nanmedian(hcap) ) > mad_std(hcap) * 3] = np.nan
                hcap[np.abs( hcap - np.nanmedian(hcap) ) > 300] = np.nan

            Na = sum(~np.isnan(hcap))  # Length after editing

            n_mon, t_span = n_months(tcap, hcap, tstep=tstep)

            ##NOTE: Not using n_mon and t_span to constrain the solution! <<<<<<<<<<<<<<<<<<<<<
            # If enough data accept radius 
            #if Na >= nlim and n_mon >= MINMONTHS and t_span >= dtlim:
            if Na >= nlim:
                break
            else:
                i_cell = []

        if not i_cell: continue

        # Parameters for model-solution
        xcap = x[i_cell]
        ycap = y[i_cell]
        tcap = time[i_cell]
        hcap = height[i_cell]
        mcap = id[i_cell]
        scap = sigma[i_cell]

        i_valid = ~np.isnan(hcap)
        if sum(i_valid) < nlim: continue

        xcap = xcap[i_valid]
        ycap = ycap[i_valid]
        tcap = tcap[i_valid]
        hcap = hcap[i_valid]
        mcap = mcap[i_valid]
        scap = scap[i_valid]

        if nreloc:
            xc = np.median(xcap)  # update inversion cell coords
            yc = np.median(ycap)

        # Define resolution param (a fraction of the accepted radius) 
        dres = dres_ * rad 

        # Estimate variance
        vcap = scap * scap

        # If reference time not given, use fixed or variable mean
        if tref_ == 'fixed':
            tref = t_mean
        elif tref_ == 'variable':
            tref = np.nanmean(tcap)
        else:
            tref = np.float(tref_)
            
        # Design matrix elements
        c0 = np.ones(len(xcap))  # intercept    (0)
        c1 = xcap - xc           # dx           (1)
        c2 = ycap - yc           # dy           (2)
        c3 = c1*c2               # dx**2
        c4 = c1*c1               # dx**2
        c5 = c2*c2               # dy**2
        c6 = tcap - tref         # trend        (6)
        c7 = 0.5 * (c6*c6)       # acceleration (7)
        c8 = np.sin(2*np.pi*c6)  # seasonal sin (8)
        c9 = np.cos(2*np.pi*c6)  # seasonal cos (9)

        # Compute distance from prediction point to data inside cap
        dist = np.sqrt((xcap-xc)*(xcap-xc) + (ycap-yc)*(ycap-yc))

        # Add small value to stabilize SVD solution
        vcap += 1e-6

        # Weighting factor: distance and error
        Wcap = 1.0 / (vcap * (1.0 + (dist/dres)*(dist/dres)))

        # Create some intermediate output variables
        sx, sy, at, ae, bi = np.nan, np.nan, np.nan, np.nan, np.nan 

        # Setup design matrix
        if model == 0:
            # Trend and seasonal
            Acap = np.vstack((c0, c8, c9, c6)).T
            mcol = [1, 2, 3]  # columns to add back
        elif model == 1:
            # Trend, acceleration and seasonal
            Acap = np.vstack((c0, c7, c8, c9, c6)).T
            mcol = [1, 2, 3, 4]
        elif model == 2:
            # Trend, acceleration, seasonal and bi-linear surface
            Acap = np.vstack((c0, c1, c2, c7, c8, c9, c6)).T
            mcol = [3, 4, 5, 6]
        else:
            # Trend, acceleration, seasonal and bi-quadratic surface (full model)
            Acap = np.vstack((c0, c1, c2, c3, c4, c5, c7, c8, c9, c6)).T
            mcol = [6, 7, 8, 9]

        has_bias = False  # bias flag
        
        # Check if bias is needed
        if len(np.unique(mcap)) > 1:
            # Add bias to design matrix
            Acap = np.vstack((Acap.T, mcap)).T
            has_bias = True

        ##NOTE: Not using t_span to constrain solution! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Check constrains before solving model (min_pts and min_tspan)
        #if len(hcap) < nlim or np.max(tcap)-np.min(tcap) < dtlim: continue
        if len(hcap) < nlim: continue

        """ Least-squares fit """

        if robust_fit:
            # Robust least squares
            try:
                model_fit = sm.RLM(hcap, Acap, missing='drop').fit(maxiter=n_iter, tol=0.001)
            except:
                print('SOMETHING WRONG WITH THE FIT... SKIPPING CELL!!!')
                continue
        else:
            # Weighted Least squares
            model_fit = sm.WLS(hcap, Acap, weights=Wcap, missing='drop').fit()

        Cm = model_fit.params    # coeffs
        Ce = model_fit.bse       # std err
        resid = model_fit.resid  # data - model
        
        # Check rate and error
        if np.abs(Cm[-1]) > dhlim or np.isinf(Ce[-1]): continue                            ##NOTE: Important for ICESat !!!
        
        # Residuals dH = H - A * Cm (remove linear trend)
        dh = hcap - np.dot(Acap, Cm)

        if robust_fit:
            chisq = chisquared(model_fit)
        else:
            chisq = rsquared(model_fit)

        # Compute amplitude of seasonal signal
        asea = np.sqrt(Cm[-2] * Cm[-2] + Cm[-3] * Cm[-3])

        # Compute phase offset
        psea = np.arctan2(Cm[-2], Cm[-3])

        # Convert phase to decimal years                                                   ##FIXME: Convert phase to days !!!
        psea /= (2*np.pi)

        # Compute root-mean-square of full model residuals
        rms = mad_std(resid)

        # Add back wanted model parameters
        dh += np.dot(Acap[:, mcol], Cm[mcol])

        # Simple binning of residuals
        tb, hb, eb, nb = binning(tcap.copy(), dh.copy(), t1lim, t2lim, tstep)[:4]         ##FIXME: Use Median to construct time series

        # Convert centroid location to latitude and longitude
        lon_c, lat_c = transform_coord(projGrd, projGeo, xc, yc)

        # Position
        DATA0[i,0] = lat_c
        DATA0[i,1] = lon_c

        # Elevation Change
        DATA0[i,2] = Cm[-1]  # trend
        DATA0[i,3] = Ce[-1]  # trend error

        # Compute acceleration and error
        if model > 0:
            at, ae  = Cm[-4], Ce[-4]

        DATA0[i,4] = at  # acceleration
        DATA0[i,5] = ae  # acceleration error

        # Surface Elevation
        DATA0[i,6] = Cm[0]
        DATA0[i,7] = Ce[0]

        # Model RMS
        DATA0[i,8] = rms

        # Compute x,y slopes in degrees
        if model > 1:
            sx, sy  = np.arctan(Cm[1])*(180 / np.pi), np.arctan(Cm[2])*(180 / np.pi)

        # Surface slope values
        DATA0[i,9] = sx
        DATA0[i,10] = sy

        # Time span of data in cap
        DATA0[i,11] = t_span
        DATA0[i,12] = tref

        # Seasonal signal
        DATA0[i,13] = asea
        DATA0[i,14] = psea

        # Bias magnitude
        if has_bias: bi = Cm[-1]

        # Aux-data from solution
        DATA0[i,15] = len(hcap)
        DATA0[i,16] = dmin
        DATA0[i,17] = rad
        DATA0[i,18] = Nb-Na
        DATA0[i,19] = chisq
        DATA0[i,20] = bi

        # Time series values
        DATA1[i,:] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(tb), hb))         ##FIXME: Think how to do this better
        DATA2[i,:] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(tb), eb))

        # Print progress (every N iterations)
        if (i % 200) == 0:
            print(('cell#', str(i) + "/" + str(len(xi)),  \
                  'trend:', np.around(Cm[mcol[-1]],2), 'm/yr', 'n_months:', n_mon, \
                  'n_pts:', len( resid), 'radius:', rad, 'reloc_dist:', reloc_dist))

    # Remove invalid entries from data matrix
    if mode == 'p':
        i_nan = np.where(np.isnan(DATA0[:,3]))
        DATA0 = np.delete(DATA0.T, i_nan, 1).T
        i_nan = np.where(np.isnan(DATA1[:,3]))
        DATA1 = np.delete(DATA1.T, i_nan, 1).T
        i_nan = np.where(np.isnan(DATA2[:,3]))
        DATA2 = np.delete(DATA2.T, i_nan, 1).T
    else:
        ##NOTE: NaNs are not removed in case a grid soluction (n_reloc=0) is selected.
        if not nreloc: grids = [d.reshape(Xi.shape) for d in DATA0.T]  # 1d -> 2d (grids) 

        variables = ['lat', 'lon', 'trend', 'trend_err', 'accel', 'accel_err',
                     'height', 'height_err', 'model_rms', 'slope_x', 'slope_y',
                     't_span', 't_ref', 'amp_seas', 'pha_seas', 'n_obs',
                     'd_min', 'd_ri', 'n_edited', 'chi2', 'bias']

    # Check if output arrays are empty
    if np.isnan(DATA0[:,3]).all():
        print(('SKIP FILE: NO PREDICTIONS TO SAVE:', ifile))
        return

    # Define output file name
    if ofile:
        outfile = ofile
    else:
        outfile = ifile

    # Output file names - strings
    path, ext = os.path.splitext(outfile)
    ofile0 = path + '_sf.h5'
    ofile1 = path + '_ts.h5'
    ofile2 = path + '_es.h5'

    print('saving data ...')

    # Save surface fit parameters
    with h5py.File(ofile0, 'w') as fo0:
        if mode == 'p':
            fo0['sf'] = DATA0                               # data matrix
        elif nreloc:
            for v,a in zip(variables, DATA0.T): fo0[v] = a  # 1d arrays
        else:
            for v,g in zip(variables, grids): fo0[v] = g    # 2d arrays
            fo0['x'], fo0['y'] = Xi[0,:], Yi[:,0] 

    # Save binned time series values
    with h5py.File(ofile1, 'w') as fo1:
        fo1['ts'] = DATA1

    # Save binned time series errors
    with h5py.File(ofile2, 'w') as fo2:
        fo2['es'] = DATA2

    # Print some statistics
    print(('*'*70))
    print(('%s %.5f %s %.2f %s %.2f %s %.2f %s %s' %
    ('Mean:',np.nanmean(DATA0[:,2]), 'Std:',np.nanstd(DATA0[:,2]), 'Min:',
         np.nanmin(DATA0[:,2]), 'Max:', np.nanmax(DATA0[:,2]), 'Model:', model)))
    print(('*'*70))
    print(('Execution time: '+ str(datetime.now()-startTime)))
    print(('Surface fit results ->', ofile0))
    print(('Time series values -> ', ofile1))
    print(('Time series errors -> ', ofile2))


# Run main program
if njobs == 1:
    print('running sequential code ...')
    [main(f) for f in files]
else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(f, n) for n, f in enumerate(files))

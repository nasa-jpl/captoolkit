#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:47:37 2015

@author: nilssonj
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
import sys
import pyproj
import h5py
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import weightedstats as ws
import matplotlib.pyplot as plt
import deepdish as dd
import scipy.interpolate as interp
from scipy.spatial import cKDTree
from datetime import datetime
from statsmodels.robust.scale import mad
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates
from pykalman import KalmanFilter
from numpy import ma
from scipy.interpolate import interp1d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import generic_filter
"""

Program for adaptive least-squares adjustment, cross-calibration and optimal merging of multi-mission altimetry data

"""

def binning(x, y, xmin, xmax, dx, tol, thr):
    """ Data binning of two variables """

    bins = np.arange(xmin, xmax + dx, dx)

    xb = np.arange(xmin, xmax, dx) + 0.5 * dx
    yb = np.ones(len(bins) - 1) * np.nan
    eb = np.ones(len(bins) - 1) * np.nan
    nb = np.ones(len(bins) - 1) * np.nan
    sb = np.ones(len(bins) - 1) * np.nan

    for i in xrange(len(bins) - 1):

        idx = (x >= bins[i]) & (x <= bins[i + 1])

        if len(y[idx]) == 0:
            continue

        ybv = y[idx]

        io = ~np.isnan(iterfilt(ybv.copy(), -9999, 9999, tol, thr))

        yb[i] = np.nanmedian(ybv[io])
        eb[i] = mad_std(ybv[io])
        nb[i] = len(ybv[io])
        sb[i] = np.sum(ybv[io])

    return xb, yb, eb, nb, sb

def binfilter(t, h, m, dt, a):
    """ Outlier filtering using bins """
    
    # Unique missions
    mi = np.unique(m)
    
    # Copy output vector
    hi = h.copy()
    
    # Loop trough missions
    for kx in xrange(len(mi)):
        
        # Determine alpha
        if mi[kx] > 3:
            
            # Set new alpha
            alpha = 2.0
        
        else:
            
            # Else keep old
            alpha = a
    
        # Get indexes of missions
        im = m == mi[kx]
        
        # Create monthly bins
        bins = np.arange(t[im].min(), t[im].max() + dt, dt)
        
        # Get data from mission
        tm, hm = t[im], h[im]
        
        # Loop trough bins
        for ky in xrange(len(bins) - 1):
            
            # Get index of data inside each bin
            idx = (tm >= bins[ky]) & (tm <= bins[ky + 1])
            
            # Get data from bin
            hb = hm[idx]
            
            # Check for empty bins
            if len(hb) == 0: continue
            
            # Compute difference
            dh = hb - np.nanmedian(hb)
            
            # Identify outliers
            io = np.abs(dh) > alpha * mad_std(hb)
            
            # Set data in bin to nan
            hb[io] = np.nan
            
            # Set data
            hm[idx] = hb
        
        # Set array!
        hi[im] = hm
            
    return hi

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


def cross_calibrate(ti, hi, dh, mi, a):
    """ Residual cross-calibration """

    # Create bias vector
    hb = np.zeros(hi.shape)

    # Set flag
    flag = 0

    # Satellite overlap periods
    to = np.array([[1995 +  5 / 12., 1996 + 5 / 12.],   # ERS-1 and ERS-2 (0)
                   [2002 + 10 / 12., 2003 + 6 / 12.],   # ERS-2 and RAA-2 (1)
                   [2010 +  6 / 12., 2011 + 0 / 12.]])  # RAA-2 and CRS-2 (3)

    # Satellite index vector
    mo = np.array([[1, 0],  # ERS-2 and ERS-1 (5,6)
                   [2, 1],  # ERS-2 and RAA-2 (3,5)
                   [3, 2]]) # RAA-2 and ICE-1 (3,0)

    # Initiate reference bias
    b_ref = 0

    # Loop trough overlaps
    for i in xrange(len(to)):

        # Get index of overlapping data
        im = (ti >= to[i, 0]) & (ti <= to[i, 1])

        # Compute the inter-mission bias
        b0 = np.nanmedian(dh[im][mi[im] == mo[i, 0]])
        b1 = np.nanmedian(dh[im][mi[im] == mo[i, 1]])

        # Compute standard deviation
        s0 = np.nanstd(dh[im][mi[im] == mo[i, 0]])
        s1 = np.nanstd(dh[im][mi[im] == mo[i, 1]])

        # Data points for each mission in each overlap
        n0 = len(dh[im][mi[im] == mo[i, 0]])
        n1 = len(dh[im][mi[im] == mo[i, 1]])

        # Standard error
        s0 /= np.sqrt(n0)
        s1 /= np.sqrt(n1)

        # Compute interval
        i0_min, i0_max, i1_min, i1_max = b0 - a * s0, b0 + a * s0, b1 - a * s1, b1 + a * s1

        # Test criterion
        if (n0 <= 50) or (n1 <= 50):
            # Set to zero
            b0, b1 = 0, 0
        elif np.isnan(b0) or np.isnan(b1):
            # Set to zero
            b0, b1 = 0, 0
        elif (i0_max > i1_min) and (i0_min < i1_max):
            # Set to zero
            b0, b1 = 0, 0
        else:
            pass

        # Cross-calibration bias
        hb[mi == mo[i, 0]] = b_ref + (b0 - b1)

        # Update bias
        b_ref = b_ref + (b0 - b1)

        # Set correction flag
        if (b0 != 0) and (b1 != 0):
            flag += 1

    return hb, flag


def sigma_filter(x):

    i = int(np.ceil(0.5*len(x)))
    xi = x[i].copy()
    x[i] = np.nan
    mean = np.nanmedian(x)
    rmse = mad_std(x)
    if np.abs(xi - mean) > 3.0 * rmse:
        xo = np.nanmean(x)
    else:
        xo = xi

    return xo

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


# Start timing of script
startTime = datetime.now()

# Output description of solution
description = ('Program for adaptive least-squares adjustment and optimal \
               merging of multi-mission altimetry data.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m), optional'),
        default=False,)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or m)'),
        default=[1,1],)

parser.add_argument(
        '-r', metavar=('r_min','r_max'), dest='radius', type=float, nargs=2,
        help=('min and max search radius (km)'),
        default=[1,10],)

parser.add_argument(
        '-i', metavar='niter', dest='niter', type=int, nargs=1,
        help=('number of iterations for least-squares adj.'),
        default=[50],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[100],)

parser.add_argument(
        '-t', metavar=('t_min','t_max'), dest='tspan', type=float, nargs=2,
        help=('min and max time for solution (yr), optional'),
        default=[-9999,9999],)

parser.add_argument(
        '-e', metavar=('ref_time'), dest='tref', type=float, nargs=1,
        help=('time to reference the solution to (yr), optional'),
        default=None,)

parser.add_argument(
        '-l', metavar=('dhdt_lim'), dest='dhdtlim', type=float, nargs=1,
        help=('discard estimate if |dh/dt| > dhdt_lim (m/yr)'),
        default=[9999],)

parser.add_argument(
        '-q', metavar=('dt_lim'), dest='dtlim', type=float, nargs=1,
        help=('discard estiamte if data-span < dt_lim (yr)'),
        default=[9999],)

parser.add_argument(
        '-k', metavar=('n_missions'), dest='nmissions', type=int, nargs=1,
        help=('min number of missions in solution'),
        default=[1],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(3031)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','s','i','b'), dest='vnames', type=str, nargs=7,
        help=('name of variables in the HDF5-file'),
        default=['lon','lat','t_year','h_res','m_rms','m_id','h_bs'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help='for parallel processing of multiple files, optional',
        default=[1],)

parser.add_argument(
        '-s', metavar=('tstep'), dest='tstep', type=float, nargs=1,
        help='time step of solution (yr)',
        default=[1.0],)

# Populate arguments
args = parser.parse_args()

# Pass arguments to internal variables
files = args.files
bbox  = args.bbox
dx    = args.dxy[0]*1e3
dy    = args.dxy[1]*1e3
dmin  = args.radius[0]*1e3
dmax  = args.radius[1]*1e3
nlim  = args.minobs[0]
t1lim = args.tspan[0]
t2lim = args.tspan[1]
tref  = args.tref[0]
dtlim = args.dtlim[0]
nmlim = args.nmissions[0]
proj  = args.proj[0]
icol  = args.vnames[:]
tstep_= args.tstep[0]
niter = args.niter[0]
njobs = args.njobs[0]

print 'parameters:'
for p in vars(args).iteritems(): print p

# Main program
def main(ifile, n=''):

    # Message to terminal
    print 'processing file:', ifile, '...'

    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print 'input file is empty!'
        return

    print 'loading data ...'

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print "input file must be in hdf5-format"
        return

    # Input variables names
    xvar, yvar, tvar, zvar, svar, ivar, ovar = icol

    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        # Read in needed variables
        lon   = fi[xvar][:]                                         # Longitude (deg)
        lat   = fi[yvar][:]                                         # Latitude  (deg)
        time  = fi[tvar][:]                                         # Time      (yrs)
        elev  = fi[zvar][:]                                         # Height    (meters)
        sigma = fi[svar][:]                                         # RMSE      (meters)
        mode  = fi[ivar][:]                                         # Mission   (int)
        dh_bs = fi[ovar][:] if ovar in fi else np.zeros(lon.shape)  # Scattering correction (meters)

    # Check for NaN-values
    inan = ~np.isnan(elev) & ~np.isnan(dh_bs)

    # Remove NaN values from arrays
    lon, lat, time, elev, sigma, mode, dh_bs = lon[inan], lat[inan], time[inan], \
                                        elev[inan], sigma[inan], mode[inan], dh_bs[inan]

    # Select only observations inside time span
    itime = (time > t1lim) & (time < t2lim)

    # Select wanted time span
    lon, lat, time, elev, sigma, mode, dh_bs = lon[itime], lat[itime], time[itime], \
                                        elev[itime], sigma[itime], mode[itime], dh_bs[itime]

    # Apply scattering correction if available
    elev -= dh_bs

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print 'converting lon/lat to x/y ...'

    # Get geographic boundaries + max search radius
    if bbox:

        # Extract bounding box
        (xmin, xmax, ymin, ymax) = bbox

        # Transform coordinates
        (x, y) = transform_coord(projGeo, projGrd, lon, lat)

        # Select data inside bounding box
        ig = (x >= xmin - dmax) & (x <= xmax + dmax) & \
             (y >= ymin - dmax) & (y <= ymax + dmax)

        # Check bbox for obs.
        if len(x[ig]) == 0:
            print 'no data points inside bounding box!'
            return

        # Cut data to bounding box limits
        lon, lat, time, elev, sigma, mode = lon[ig], lat[ig], time[ig], \
                                            elev[g], sigma[ig], mode[ig]

    else:

        # Convert into stereographic coordinates
        (x, y) = transform_coord(projGeo, projGrd, lon, lat)

        # Get bbox from data
        (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Construct solution grid - add border to grid
    (Xi, Yi) = make_grid(xmin-10e3, xmax+10e3, ymin-10e3, ymax+10e3, dx, dy)

    # Convert to geographical coordinates
    (LON, LAT) = transform_coord(projGrd, projGeo, Xi, Yi)

    # Flatten prediction grid
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Zip data to vector
    coord = zip(x.ravel(), y.ravel())

    print 'building the k-d tree ...'

    # Construct KD-Tree
    Tree = cKDTree(coord)
    
    print 'k-d tree built!'
    
    # Convert to years
    tstep = tstep_ / 12.0

    # Set up search cap
    dr = np.arange(dmin, dmax + 2e3, 2e3)

    # Create empty lists
    lats = list()
    lons = list()
    lat0 = list()
    lon0 = list()
    dxy0 = list()
    h_ts = list()
    e_ts = list()
    m_id = list()
    h_ct = list()
    h_cf = list()
    h_cr = list()
    f_cr = list()
    tobs = list()

    # Enter prediction loop
    for i in xrange(len(xi)):

        # Number of observations
        nobs = 0

        # Time difference
        dt = 0

        # Temporal sampling
        npct = 1

        # Number of sensors
        nsen = 0

        # Meet data constraints
        for ii in xrange(len(dr)):

            # Query the Tree with data coordinates
            idx = Tree.query_ball_point((xi[i], yi[i]), dr[ii])

            # Check for empty arrays
            if len(time[idx]) == 0:
                continue

            # Constraints parameters
            dt   = np.max(time[idx]) - np.min(time[idx])
            nobs = len(time[idx])
            nsen = len(np.unique(mode[idx]))

            # Bin time vector
            t_sample = binning(time[idx], time[idx], t1lim, t2lim, 1.0/12., 5, 5)[1]

            # Test for null vector
            if len(t_sample) == 0: continue

            # Sampling fraction
            npct = np.float(len(t_sample[~np.isnan(t_sample)])) / len(t_sample)

            # Constraints
            if nobs > nlim:
                if dt > dtlim:
                    if nsen >= nmlim:
                        if npct > 0.70:
                            break

        # Final test of data coverage
        if (nobs < nlim) or (dt < dtlim): continue

        # Parameters for model-solution
        xcap = x[idx]
        ycap = y[idx]
        tcap = time[idx]
        hcap = elev[idx]
        scap = sigma[idx]
        mcap = mode[idx]

        # Make copy of output variable
        horg = hcap.copy()
        
        # Centroid of all data
        xc = np.median(xcap)
        yc = np.median(ycap)

        # Compute distance from center
        dxy = np.sqrt((xcap - xc) ** 2 + (ycap - yc) ** 2)

        #
        # Least-Squares Adjustment
        # ---------------------------------
        #
        # h =  x_t + x_j + x_s
        # x = (A' A)^(-1) A' y
        # r = y - Ax
        #
        # ---------------------------------
        #

        # Threshold for outliers in each bin
        alpha = 3.0

        # Apply outlier filter to the data
        hcap = binfilter(tcap.copy(), hcap.copy(), mcap.copy(), tstep, alpha)
        
        # Compute number of NaN's
        n_nan = len(hcap[np.isnan(hcap)])

        # Make sure we have enough data for computation
        if (nobs - n_nan) < nlim: continue
        
        # Trend component
        dti = tcap - tref

        # Four-term fourier series for seasonality
        cos1 = np.cos(2 * np.pi * dti)
        sin1 = np.sin(2 * np.pi * dti)
        cos2 = np.cos(4 * np.pi * dti)
        sin2 = np.sin(4 * np.pi * dti)

        # Construct bias parameters
        b_ice1 = np.zeros(hcap.shape)
        b_csin = np.zeros(hcap.shape)
        b_clrm = np.zeros(hcap.shape)
        b_ra21 = np.zeros(hcap.shape)
        b_ers1 = np.zeros(hcap.shape)
        b_ers2 = np.zeros(hcap.shape)

        # Set unit-step functions (0/1)
        b_ers1[mcap == 6] = 1.
        b_ers2[mcap == 5] = 1.
        b_ice1[mcap == 0] = 1.
        b_ra21[mcap == 3] = 1.
        b_csin[mcap == 1] = 1.
        b_clrm[mcap == 2] = 1.

        # Design matrix for adjustment procedure
        Acap = np.vstack((dti, 0.5*dti**2, cos1, sin1, cos2, sin2, b_ice1, \
                          b_csin, b_clrm, b_ra21, b_ers2, b_ers1)).T

        # Try to solve least-squares system
        try:

            # Least-squares bias adjustment
            linear_model = sm.RLM(hcap, Acap, missing='drop')

            # Fit the model to the data
            linear_model_fit = linear_model.fit(maxiter=niter)

        # If not possible continue
        except:

            print "Solution invalid!"
            continue

        # Coefficients and standard errors
        Cm = linear_model_fit.params
        Ce = linear_model_fit.bse

        # Compute model residuals
        dh = hcap - np.dot(Acap, Cm)

        # Identify outliers
        inan = np.isnan(iterfilt(dh.copy(), -9999, 9999, 5, 5.0))

        # Set outliers to NaN
        hcap[inan] = np.nan

        # Compute RMSE of corrected residuals
        rms = mad_std(dh)

        # Bias correction
        h_cal_fit = np.dot(Acap[:,[-6,-5,-4,-3,-2,-1]], Cm[[-6,-5,-4,-3,-2,-1]])

        # Remove inter satellite biases
        hcap -= h_cal_fit
        
        """
           
           BE AWARE OF THE EXTRA CROSS-CALIBRATION YOU MIGHT WANNA TURN THAT OFF!
           
        """
        
        # Initiate residual cross-calibration flag
        flag = 0

        # Create residual cross-calibration index vector
        msat = np.ones(mcap.shape) * np.nan

        # Set overlap indexes
        msat[mcap == 6] = 0
        msat[mcap == 5] = 1
        msat[(mcap == 3) | (mcap == 0)] = 2
        msat[(mcap == 1) | (mcap == 2)] = 3

        # Apply post-fit residual cross-calibration in overlapping areas
        h_cal_res, flag = cross_calibrate(tcap.copy(), hcap.copy(), dh.copy(), msat.copy(), 2.0)

        # Correct for second bias
        hcap -= h_cal_res
        
        # Compute total correction
        h_cal_tot = h_cal_fit #+ h_cal_res

        # Transform coordinates
        (lon_i, lat_i) = transform_coord(projGrd, projGeo, xcap, ycap)
        (lon_0, lat_0) = transform_coord(projGrd, projGeo, xi[i], yi[i])

        # Save output variables to list for each solution
        lats.append(lat_i)
        lons.append(lon_i)
        lat0.append(lat_0)
        lon0.append(lon_0)
        dxy0.append(dxy)
        h_ts.append(horg)
        e_ts.append(scap)
        m_id.append(mcap)
        h_ct.append(h_cal_tot)
        h_cf.append(h_cal_fit)
        h_cr.append(h_cal_res)
        f_cr.append(flag)
        tobs.append(tcap)
        
        # Print meta data to terminal
        if (i % 10) == 0:
            print 'Progress:',str(i),'/',str(len(xi)),'Rate:', np.around(Cm[0],2), 'Acceleration:', np.around(Cm[1],2)

    # Saveing the data to file
    print 'Saving data to file ...'

    # Save data to specific file
    ofile = ifile.replace('.h5', '_mrg.dic')

    # Save using deepdish to hdf5
    dd.io.save(ofile, {'lat': lats, 'lon': lons, 'lat0': lat0, 'lon0': lon0, 'dh_ts': h_ts, 'de_ts': e_ts,\
                           'm_idx': m_id, 'h_cal_tot': h_ct, 'h_cal_fit': h_cf, 'h_cal_res': h_cr,\
                           'h_cal_flg': f_cr, 'dxy0': dxy0,'t_year': tobs}, compression='lzf')

# Run main program!
if njobs == 1:

    # Single core
    print 'running sequential code ...'
    [main(f) for f in files]

else:

    # Multiple cores
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

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

"""

Program for adaptive least-squares adjustment, cross-calibration and optimal merging of multi-mission altimetry data

"""
def fillnans(A):

    inds = np.arange(A.shape[0])
    
    good = np.where(np.isfinite(A))
    
    f = interp1d(inds[good], A[good],bounds_error=False)
    
    B = np.where(np.isfinite(A),A,f(inds))
    
    return B


def binning(x, y, xmin, xmax, dx, tol, thr):
    """ Data binning of two variables """

    bins = np.arange(xmin,xmax+dx,dx)

    xb = np.arange(xmin,xmax,dx) + 0.5 * dx
    yb = np.ones(len(bins)-1)*np.nan
    eb = np.ones(len(bins)-1)*np.nan
    nb = np.ones(len(bins)-1)*np.nan
    sb = np.ones(len(bins)-1)*np.nan
    
    for i in range(len(bins)-1):
        
        idx = (x >= bins[i]) & (x <= bins[i+1])
        
        if len(y[idx]) == 0:
            continue
    
        ybv = y[idx]
                
        io = ~np.isnan(iterfilt(ybv.copy(), -9999, 9999, tol, thr))
        
        yb[i] = np.nanmedian(ybv[io])
        eb[i] = mad_std(ybv[io])
        nb[i] = len(ybv[io])
        sb[i] = np.sum(ybv[io])
    
    return xb, yb, eb, nb, sb


def binfilt(x, y, xmin, xmax, alpha, dx):
    """ Data binning of two variables """

    bins = np.arange(xmin, xmax + dx, dx)
  
    for i in range(len(bins)-1):
        
        idx = (x >= bins[i]) & (x <= bins[i+1])
        
        if len(y[idx]) == 0: continue
    
        ybv = y[idx]
                
        io = np.abs(ybv - np.nanmedian(ybv)) > alpha * mad_std(ybv)
        
        ybv[io] = np.nan
        
        y[idx] = ybv
    
    return y


def bin_mission(ti, hi, mi, ei, tstart, tstop, tstep, tol, alpha):
    """ Binning of multi-mission data """

    # Get number of unique missions 
    mu  = np.unique(mi)
    
    # Get the size of the final vector
    tb = np.arange(tstart, tstop, tstep)

    # Create empty vectors
    hbi = np.ones((len(mu), len(tb))) * np.nan
    ebi = np.ones((len(mu), len(tb))) * np.nan
    mbi = np.ones((len(mu), len(tb))) * np.nan
    tbi = np.ones((len(mu), len(tb))) * np.nan
    nbi = np.ones((len(mu), len(tb))) * np.nan
    
    # Bin mission residuals to equal time steps
    for i in range(len(mu)):
        
        # Get indices for missions
        im = mi == mu[i]

        # Get the mission specific error
        m_rms = ei[im].mean()
        
        # Bin the residuals according to time using the median value
        (tb, hb, eb, nb) = binning(ti[im], hi[im], tstart, tstop, tstep, tol, alpha)[0:4]
        
        # Copy variable
        es = eb.copy()

        # Set systemtic error
        es[~np.isnan(es)] = m_rms
                
        # Stack output data
        hbi[i, :] = hb                      # Time series
        ebi[i, :] = np.sqrt(es**2 + eb**2)  # RSS combined systematic and random error
        mbi[i, :] = mu[i]                   # Mission index
        tbi[i, :] = tb                      # Time vector
        nbi[i, :] = nb                      # Number of observations in bin

    return tbi, hbi, ebi, nbi, mbi


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
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def fill(Hi,Hw):
    """ Fill data array with provide values. """
    
    for i in range(len(Hw)):
        
        Hi[np.isnan(Hi[:,i]),i] = Hw[i]
        
    return Hi


def cross_calibrate(t, h, dh, m, a):
    """Cross-calibration of missons"""

    # Set flag
    flag = 0

    # Satellite overlap periods
    to = np.array([[1995 + 5 / 12., 1996 + 5 / 12.],  # ERS-1 and ERS-2
                   [2002 + 10 / 12., 2003 + 6 / 12.],  # ERS-2 and RAA-2
                   [2003 + 2 / 12., 2009 + 9 / 12.],  # RAA-2 and ICE-1
                   [2010 + 5 / 12., 2011 + 3 / 12.]])  # RAA-2 and CRS-2

    # Satellite index vector
    mo = np.array([[5, 6],  # ERS-2 and ERS-1 (5,6)
                   [3, 5],  # ERS-2 and RAA-2 (3,5)
                   [0, 3],  # RAA-2 and ICE-1 (3,0)
                   [1, 3]])  # RAA-2 and CRS-2 (1,3)

    # Initiate reference bias
    b_ref = 0

    # Loop trough overlaps
    for i in range(len(to)):

        # Get index of overlapping data
        im = (t >= to[i, 0]) & (t <= to[i, 1])

        # Compute the inter-mission bias
        b0 = np.nanmean(dh[im][m[im] == mo[i, 0]])
        b1 = np.nanmean(dh[im][m[im] == mo[i, 1]])

        # Compute standard deviation
        s0 = np.nanstd(dh[im][m[im] == mo[i, 0]])
        s1 = np.nanstd(dh[im][m[im] == mo[i, 1]])

        # Data points for each mission in each overlap
        n0 = len(dh[im][m[im] == mo[i, 0]])
        n1 = len(dh[im][m[im] == mo[i, 1]])

        # Standard error
        s0 /= np.sqrt(n0 - 1)
        s1 /= np.sqrt(n1 - 1)

        # Compute interval
        i0_min, i0_max, i1_min, i1_max = b0 - a * s0, b0 + a * s0, b1 - a * s1, b1 + a * s1

        # Test criterions
        if (n0 <= 1) or (n1 <= 1):
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

        # Test for specific case
        if (mo[i, 0] == 0 and mo[i, 1] == 3):

            # Cross-calibrate
            h[m == mo[i, 0]] -= b_ref + (b0 - b1)

            # Set correction flag
            if (b0 != 0) and (b1 != 0):
                flag += 1

        else:

            # Cross-calibrate
            h[m == mo[i, 0]] -= b_ref + (b0 - b1)

            # Update bias
            b_ref = b_ref + (b0 - b1)

            # Set correction flag
            if (b0 != 0) and (b1 != 0):
                flag += 1

    return h, flag


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
        '-m', metavar=('id_mission'), dest='idmission', type=int, nargs=1,
        help=('index of reference missions'),
        default=[0],)

parser.add_argument(
        '-u', metavar=('resid_lim'), dest='residlim', type=float, nargs=1,
        help=('remove residuals if |residual| > resid_lim (m)'),
        default=[100],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(3031)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','s','i','b'), dest='vnames', type=str, nargs=7,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','t_year','h_cor','m_rms','m_id','h_bs'],)

parser.add_argument(
        '-x', metavar=('expr'), dest='expr',  type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[None],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files, optional",
        default=[1],)

parser.add_argument(
        '-s', metavar=('tstep'), dest='tstep', type=float, nargs=1,
        help="time step of solution (yr)",
        default=[1.0/12.0],)

# Populate arguments
args = parser.parse_args()

# Pass arguments to internal variables
files = args.files[:]
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
dhlim = args.dhdtlim[0]
nmlim = args.nmissions[0]
mref_ = args.idmission[0]
slim  = args.residlim[0]
proj  = args.proj[0]
icol  = args.vnames[:]
tstep = args.tstep[0]
niter = args.niter[0]
njobs = args.njobs[0]

print('parameters:')
for p in list(vars(args).items()): print(p)

# Main program
def main(ifile, n=''):
    
    # Message to terminal
    print(('processing file:', ifile, '...'))

    # Check for empty file
    if os.stat(ifile).st_size == 0:
        print('input file is empty!')
        return

    print('loading data ...')

    # Determine input file type
    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print("input file must be in hdf5-format")
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
        oind  = fi[ovar][:] if ovar in fi else np.ones(lon.shape)   # Outliers  (int)

    # Check for NaN-values
    inan = ~np.isnan(elev) & ~np.isnan(oind)

    # Remove NaN values from arrays
    lon, lat, time, elev, sigma, mode = lon[inan], lat[inan], time[inan], \
                                        elev[inan], sigma[inan], mode[inan]

    # Select only observations inside time interval
    itime = (time > t1lim) & (time < t2lim)

    # Select wanted time span
    lon, lat, time, elev, sigma, mode = lon[itime], lat[itime], time[itime], \
                                        elev[itime], sigma[itime], mode[itime]
                                        
    # Select only wanted missions - not mission 4
    imode = (mode != 4)                              
                                        
    # Select wanted modes
    lon, lat, time, elev, sigma, mode = lon[imode], lat[imode], time[imode], \
                                        elev[imode], sigma[imode], mode[imode]
    
    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print('converting lon/lat to x/y ...')

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
            print('no data points inside bounding box!')
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
    coord = list(zip(x.ravel(), y.ravel()))

    print('building the k-d tree ...')

    # Construct KD-Tree
    Tree = cKDTree(coord)

    # Number of months of time series
    months = len(np.arange(t1lim, t2lim + tstep, tstep))

    # Total number of columns
    ntot = months + 4

    # Create output array
    OFILE0 = np.ones((len(xi), 23))   * 9999
    OFILE1 = np.ones((len(xi), ntot)) * 9999
    OFILE2 = np.ones((len(xi), ntot)) * 9999
    OFILE3 = np.ones((len(xi), ntot)) * 9999
    OFILE4 = np.ones((len(xi), ntot)) * 9999

    # Save corrected rate
    b_rate = np.ones((len(xi), 1)) * np.nan
    # Set up search cap
    dr = np.arange(dmin, dmax + 2e3, 2e3)

    # Enter prediction loop
    for i in range(len(xi)):
        
        # Number of observations
        nobs = 0

        # Time difference
        dt = 0

        # Temporal sampling
        npct = 1

        # Number of sensors
        nsen = 0

        # Meet data constraints
        for ii in range(len(dr)):

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

        # Centroid of all data
        xc = np.median(xcap)
        yc = np.median(ycap)

        # Get reference
        mref = mref_
        
        # Reference to specific mission
        if len(hcap[mcap == mref]) > 0:

            # Tie to laser surface
            hcap -= np.median(hcap[mcap == mref])

        elif len(hcap[mcap == (mref + 1)]) > 0:

            # Tie to SARin surface
            hcap -= np.median(hcap[mcap == (mref + 1)])

            # Change mission tie index
            mref += 1

        else:

            # Tie to mean surface
            hcap -= np.median(hcap)
        
        #
        # Least-Squares Adjustment
        # ---------------------------------
        #
        # h =  x_t + x_j + x_s
        # x = (A' W A)^(-1) A' W y
        # r = y - Ax
        #
        # ---------------------------------
        #

        # Threshold for outliers in each bin
        alpha = 5.0

        # Convergence tolerance (%)
        tol = 3.0

        # Times series binning of each mission
        (tcap, hcap, scap, ncap, mcap) = bin_mission(tcap, hcap, mcap, scap, t1lim, t2lim, tstep, tol, alpha)

        # Size of original observational matrix
        (n, m) = hcap.T.shape

        # Unravel array to vectors
        tcap = tcap.T.ravel()
        hcap = hcap.T.ravel()
        scap = scap.T.ravel()
        mcap = mcap.T.ravel()

        # Additional outlier editing
        inan = np.isnan(binfilt(tcap.copy(), hcap.copy(), tcap.min(), tcap.max(), 3.0, 3./12.))

        # Set outliers to NaN
        hcap[inan] = np.nan
        scap[inan] = np.nan
        mcap[inan] = np.nan

        # Trend component
        dti = tcap - tref

        # Compute new statistics
        (nobs, tspan) = len(hcap[~np.isnan(hcap)]), tcap.max() - tcap.min()

        # Reject grid node if true
        if (nobs < nlim) & (tspan < dtlim): continue

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
        b_ra22 = np.zeros(hcap.shape)
        b_ers1 = np.zeros(hcap.shape)
        b_ers2 = np.zeros(hcap.shape)

        # Set unit-step functions (0/1)
        b_ers1[mcap == 6] = 1.
        b_ers2[mcap == 5] = 1.
        b_ice1[mcap == 0] = 1.
        b_ra21[mcap == 3] = 1.
        b_ra22[mcap == 4] = 1.
        b_csin[mcap == 1] = 1.
        b_clrm[mcap == 2] = 1.

        # Design matrix for adjustment procedure
        Acap = np.vstack((dti, 0.5*dti**2, cos1, sin1, cos2, sin2, b_ice1, \
                          b_csin, b_clrm, b_ra21, b_ra22, b_ers2, b_ers1)).T

        # Try to solve least-squares system
        try:
            
            # Least-squares bias adjustment
            linear_model = sm.RLM(hcap, Acap, missing='drop')

            # Fit the model to the data
            linear_model_fit = linear_model.fit(maxiter=10)

        # If not possible continue
        except:

            continue

        # Length post editing
        nsol = len(hcap)

        # Coefficients and standard errors
        Cm = linear_model_fit.params
        Ce = linear_model_fit.bse

        # Amplitude of annual seasoanl signal
        amp = np.sqrt(Cm[2]**2 + Cm[3]**2)

        # Phase of annual seasoanl signal
        phi = np.arctan2(Cm[3], Cm[2]) / (2.0 * np.pi)

        # Compute model residuals
        dh = hcap - np.dot(Acap, Cm)

        # Identify outliers
        inan = np.isnan(iterfilt(dh.copy(), -slim, slim, 5, 3.0))

        # Set outliers to NaN
        hcap[inan] = np.nan
        scap[inan] = np.nan
        mcap[inan] = np.nan
        
        # Compute RMSE of corrected residuals
        rmse = mad_std(dh[~inan])

        # Bias correction
        h_bias = np.dot(Acap[:,[-7,-6,-5,-4,-3,-2,-1]], Cm[[-7,-6,-5,-4,-3,-2,-1]])

        # Save original uncorrected time series
        horg = hcap.copy()

        # Remove inter mission biases
        hcap -= h_bias

        # Initiate residual cross-calibration flag
        flag = 0

        # Apply post-fit cross-calibration in overlapping areas
        hcap, flag = cross_calibrate(tcap.copy(), hcap.copy(), dh.copy(), mcap.copy(), 1.0)

        # Binned time for plotting
        tbin = np.arange(t1lim, t2lim, tstep) + 0.5 * tstep

        # Re-format back to arrays
        hbo = horg.reshape(n,m).T
        hbi = hcap.reshape(n,m).T
        tbi = tcap.reshape(n,m).T
        ebi = scap.reshape(n,m).T
        mbi = mcap.reshape(n,m).T
                
        # Copy original vector
        hcor = np.copy(hbi)

        # Take the weighted average of all mission in each bin
        (hbi_w, ebi_w) = np.ma.average(np.ma.array(hbi, mask=np.isnan(hbi)), \
                         weights=np.ma.array(ebi, mask=np.isnan(ebi)), \
                         axis=0, returned=True)
        
        # Convert back to original array, with nan's
        hbi_w = np.ma.filled(hbi_w, np.nan)
        ebi_w = np.ma.filled(ebi_w, np.nan)

        # Number of rows to add
        n_add = 6 - len(hbi)

        # Construct binary mask
        binary = hbi_w.copy()

        # Set to zeros (data) and ones (nan)
        binary[~np.isnan(binary)] = 0
        binary[np.isnan(binary)] = 1

        # Apply distance transform
        bwd = distance_transform_edt(binary)

        # Set these values to nan's
        inoip = bwd >= 12

        # Pad by adding rows
        for kx in range(n_add):

            # Add rows to obs. matrix
            hbo = np.vstack((hbo, np.ones(hbi_w.shape) * np.nan))
            hbi = np.vstack((hbi, np.ones(hbi_w.shape) * np.nan))
            ebi = np.vstack((ebi, np.ones(hbi_w.shape) * np.nan))
            mbi = np.vstack((mbi, np.ones(hbi_w.shape) * np.nan))
            tbi = np.vstack((tbi, tbin))

        # Padd mission arrays using weighted averages
        hbi = fill(hbi, hbi_w)
        ebi = fill(ebi, ebi_w)

        # Reject grid node if true
        if len(hbi_w[~np.isnan(hbi_w)]) <= 2: continue

        #
        # Kalman state-space model
        # ------------------------
        #
        # z_t = H * z_t + v_t
        # x_t = A * x_t-1 + w_t-1
        #
        # ------------------------
        #

        # Create observational matrix
        Ht = np.eye(4)

        # Determine the number of rows to add
        n_add = len(hbi) - 4

        # Rows to observational matrix
        for ky in range(n_add):

            # Add rows to obs. matrix
            Ht = np.vstack((Ht, [0, 0, 0, 0]))

        # Populate observational matrix
        Ht[:, [0, 2]] = 1

        # Seasonal signal
        ck = np.cos(np.pi/6)
        sk = np.sin(np.pi/6)

        # Transition matrix
        At = [[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, +ck, +sk],
              [0.0, 0.0, -sk, +ck]]

        # Observational noise
        Rt = np.diag(np.nanmean(ebi**2,1))

        # Initial start value of filter
        y0 = np.median(hbi_w[~np.isnan(hbi_w)][0:3])

        # Constrain only transition covariance
        params = ['transition_covariance']

        # Estimating transition covaraiance from individual missions
        if len(hcap[(mcap <= 3) & (~np.isnan(mcap))]) > 1:

            # Only good missions
            Ct = KalmanFilter(em_vars=params). \
                em(hcap[mcap <= 3], n_iter=2).transition_covariance

        else:

            # All missions
            Ct = KalmanFilter(em_vars=params). \
                em(hcap[~np.isnan(hcap)], n_iter=2).transition_covariance

        # Transition covariance
        Qt = np.diag([0.0, 1.0, 0.5, 0.5]) * tstep * Ct

        # Initial state vector
        m0 = [y0, Cm[0], Cm[2], Cm[3]]

        # Initial state covariance
        q0 = np.diag([0, Ce[0], Ce[2], Ce[3]]) ** 2

        # Create kalman filter
        kf = KalmanFilter(initial_state_mean       = m0,
                          initial_state_covariance = q0,
                          transition_matrices      = At,
                          observation_matrices     = Ht,
                          observation_covariance   = Rt,
                          transition_covariance    = Qt)

        # Estimate number percentage of interpolated data
        n_per = 100 * float(len(hbi_w[np.isnan(hbi_w)])) / len(hbi_w)

        # Mask and transpose array
        hbi_masked =  ma.masked_array(hbi,mask=np.isnan(hbi)).T

        # Apply Kalman filter with parameter learning on residuals
        (dh_ts, dh_es) = kf.smooth(hbi_masked)

        # Compute the total RSS of all model parameters
        dh_es = [dh_es[k, 0, 0] for k in range(len(dh_es))]

        # Sum all parameters for time series
        dh_ts = dh_ts[:, 0]

        # Compute standard deviation
        dh_es = np.sqrt(dh_es)

        # Mask output array
        dh_ts[inoip] = np.nan
        dh_es[inoip] = np.nan

        # Rename weighted solution
        dh_ws = hbi_w
        dh_ew = ebi_w
        
        # Converte back to georaphical coordinates
        (lon_c, lat_c) = transform_coord(projGrd, projGeo, xc, yc)

        # Final search radius
        radius = dr[ii]

        # Compute new elevation change rate after post-fit residuals
        b_rate = np.polyfit(tbin[~np.isnan(dh_ws)] - tbin[~np.isnan(dh_ws)].mean(),
                               dh_ws[~np.isnan(dh_ws)], 2, w=1.0/dh_ew[[~np.isnan(dh_ws)]] ** 2)[1]

        # Save data to output files
        OFILE0[i, :] = np.hstack((lat_c, lon_c, Cm[0], Ce[0], Cm[1], Ce[1], rmse, dt, amp, phi, n_per,Cm[[-7, -6, -5, -4, -3, -2, -1]], nobs, nsol, radius, flag, b_rate))
        OFILE1[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_ts))
        OFILE2[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_es))
        OFILE3[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_ws))
        OFILE4[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_es))

        # Print progress
        print((str(i) + "/" + str(len(xi))+" Radius: "+ str(np.around(dr[ii], 0)) +" Rate: "+str(np.around(Cm[0]*100,2))+\
              " (cm/yr)"+' Interp: '+str(np.around(n_per,0))+' Rate_adj: '+str(np.around(b_rate*100,2))+" (cm/yr)"))

    # Identify unwanted data
    I0 = OFILE0[:, 0] != 9999
    I1 = OFILE1[:, 0] != 9999
    I2 = OFILE2[:, 0] != 9999
    I3 = OFILE3[:, 0] != 9999
    I4 = OFILE4[:, 0] != 9999

    # Remove unwnated data
    OFILE0 = OFILE0[I0, :]
    OFILE1 = OFILE1[I1, :]
    OFILE2 = OFILE2[I2, :]
    OFILE3 = OFILE3[I3, :]
    OFILE4 = OFILE4[I4, :]

    # Check if we have any data
    if len(OFILE0[:, 0]) == 0:
        # Print message
        print(" No data to save! ")
        return

    # Save solution to disk
    with h5py.File(ifile.replace('.h5', '_sf.h5'), 'w') as f0:

        # Save meta data
        f0['sf'] = OFILE0

    with h5py.File(ifile.replace('.h5', '_ts.h5'), 'w') as f1:

        # Save adjusted and merged time series
        f1['ts'] = OFILE1

    with h5py.File(ifile.replace('.h5', '_es.h5'), 'w') as f2:

        # Save error estimate for time series
        f2['es'] = OFILE2

    with h5py.File(ifile.replace('.h5', '_tw.h5'), 'w') as f3:

        # Save error estimate for time series
        f3['tw'] = OFILE3

    with h5py.File(ifile.replace('.h5', '_ew.h5'), 'w') as f4:

        # Save error estimate for time series
        f4['ew'] = OFILE4


# Run main program!
if njobs == 1:
    
    # Single core
    print('running sequential code ...')
    [main(f) for f in files]

else:
    
    # Multiple cores
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

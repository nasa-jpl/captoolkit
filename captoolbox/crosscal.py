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
import pyproj
import h5py
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import deepdish as dd
from scipy.spatial import cKDTree

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


def cross_calibrate_old(ti, hi, dh, mi, a):
    """ Residual cross-calibration """

    # Create bias vector
    hb = np.zeros(hi.shape)

    # Set flag
    flag = 0

    # Satellite overlap periods
    to = np.array([[1995 +  5 / 12. - 1.0, 1996 + 5 / 12. + 1.0],   # ERS-1 and ERS-2 (0)
                   [2002 + 10 / 12. - 1.0, 2003 + 6 / 12. + 1.0],   # ERS-2 and RAA-2 (1)
                   [2010 +  6 / 12. - 1.0, 2011 + 0 / 12. + 1.0]])  # RAA-2 and CRS-2 (3)

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
            #print "nobs"
        elif np.isnan(b0) or np.isnan(b1):
            # Set to zero
            b0, b1 = 0, 0
            #print "nans"
        elif (i0_max > i1_min) and (i0_min < i1_max):
            # Set to zero
            b0, b1 = 0, 0
            #print "rmse"
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


def design_matrix(t, m):
    """Design matrix padded with dummy variables"""

    # Four-term fourier series for seasonality
    cos0 = np.cos(2 * np.pi * t)
    sin0 = np.sin(2 * np.pi * t)
    cos1 = np.cos(4 * np.pi * t)
    sin1 = np.sin(4 * np.pi * t)

    # Standard design matrix
    A = np.vstack((np.ones(t.shape), t, 0.5 * t ** 2,\
                   cos0, sin0, cos1, sin1)).T

    # Unique indices
    mi = np.unique(m)

    # Make column list
    cols = []

    # Add biases to design matrix
    for i in xrange(len(mi)):

        # Create offset array
        b = np.zeros((len(m),1))

        # Set values
        b[m == mi[i]] = 1.0

        # Add bias to array
        A = np.hstack((A, b))

        # Index column
        i_col = 7 + i

        # Save to list
        cols.append(i_col)

    return A, cols


def rlsq(x, y, n=1):
    """ Fit a robust polynomial of n:th deg."""
    
    # Test solution
    if len(x[~np.isnan(y)]) <= (n + 1):

        if n == 0:
            p = np.nan
            s = np.nan
        else:
            p = np.zeros((1,n)) * np.nan
            s = np.nan
        
        return p, s

    # Empty array
    A = np.empty((0,len(x)))

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
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=3, tol=0.001)

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
            p = np.zeros((1,n)) * np.nan
            s = np.nan

    return p, s


def cross_calibrate(ti, hi, dh, mi, a):
    """ Residual cross-calibration """

    # Create bias vector
    hb = np.zeros(hi.shape)

    # Set flag
    flag = 0
    
    # Satellite overlap periods
    to = np.array([[1995 + 05. / 12. - .5, 1996 + 05. / 12. + .5],   # ERS-1 and ERS-2 (0)
                   [2002 + 10. / 12. - .5, 2003 + 06. / 12. + .5],   # ERS-2 and RAA-2 (1)
                   [2010 + 06. / 12. - .5, 2010 + 10. / 12. + .5]])  # RAA-2 and CRS-2 (3)
    
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

        # Get mission data for fit
        t0, t1 = ti[im][mi[im] == mo[i, 0]], ti[im][mi[im] == mo[i, 1]]
        h0, h1 = dh[im][mi[im] == mo[i, 0]], dh[im][mi[im] == mo[i, 1]]
        
        # Fit zeroth order polynomial - mean value
        p0, s0 = rlsq(t0, h0, n=0)
        p1, s1 = rlsq(t1, h1, n=0)

        # Estimate bias at given overlap time
        b0 = np.nan if np.isnan(p0) else p0
        b1 = np.nan if np.isnan(p1) else p1
        
        # Data points for each mission in each overlap
        n0 = len(dh[im][mi[im] == mo[i, 0]])
        n1 = len(dh[im][mi[im] == mo[i, 1]])

        # Standard error
        s0 /= np.sqrt(n0)
        s1 /= np.sqrt(n1)

        # Compute interval overlap
        i0_min, i0_max, i1_min, i1_max = b0 - a * s0, b0 + a * s0, b1 - a * s1, b1 + a * s1

        # Test criterion
        if np.isnan(b0) or np.isnan(b1):
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


# Output description of solution
description = ('Program for adaptive least-squares adjustment and optimal \
               merging of multi-mission altimetry data.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or m)'),
        default=[1,1],)

parser.add_argument(
        '-r', metavar=('r_min','r_max'), dest='radius', type=float, nargs=2,
        help=('min and max search radius (km)'),
        default=[5,5],)

parser.add_argument(
        '-i', metavar='niter', dest='niter', type=int, nargs=1,
        help=('number of iterations for least-squares adj.'),
        default=[50],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[100],)

parser.add_argument(
        '-t', metavar=('ref_time'), dest='tref', type=float, nargs=1,
        help=('time to reference the solution to (yr), optional'),
        default=None,)

parser.add_argument(
        '-l', metavar=('dhdt_lim'), dest='dhdtlim', type=float, nargs=1,
        help=('discard estimate if |dh/dt| > dhdt_lim (m/yr)'),
        default=[9999],)

parser.add_argument(
        '-q', metavar=('dt_lim'), dest='dtlim', type=float, nargs=1,
        help=('discard estiamte if data-span < dt_lim (yr)'),
        default=[0],)

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

parser.add_argument(
        '-b', dest='rcali', action='store_true',
        help=('apply residual cross-calibration'),
        default=False)

parser.add_argument(
        '-a', dest='apply', action='store_true',
        help=('apply cross-calibration to elevation residuals'),
        default=False)

parser.add_argument(
        '-o', dest='serie', action='store_true',
        help=('save point data as time series'),
        default=False)


# Populate arguments
args = parser.parse_args()

# Pass arguments to internal variables
files = args.files
dx    = args.dxy[0]*1e3
dy    = args.dxy[1]*1e3
dmin  = args.radius[0]*1e3
dmax  = args.radius[1]*1e3
nlim  = args.minobs[0]
tref  = args.tref[0]
dtlim = args.dtlim[0]
nmlim = args.nmissions[0]
proj  = args.proj[0]
icol  = args.vnames[:]
tstep_= args.tstep[0]
niter = args.niter[0]
njobs = args.njobs[0]
rcali = args.rcali
apply = args.apply
serie = args.serie

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
        lon   = fi[xvar][:]                                                 # Longitude (deg)
        lat   = fi[yvar][:]                                                 # Latitude  (deg)
        time  = fi[tvar][:]                                                 # Time      (yrs)
        elev  = fi[zvar][:]                                                 # Height    (meters)
        sigma = fi[svar][:] if svar in fi else np.zeros(lon.shape) * np.nan # RMSE      (meters)
        mode  = fi[ivar][:]                                                 # Mission   (int)
        dh_bs = fi[ovar][:] if ovar in fi else np.zeros(lon.shape)          # Scattering correction (meters)

    # Apply scattering correction if available
    elev -= dh_bs

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = proj

    print 'converting lon/lat to x/y ...'

    # Convert into stereographic coordinates
    (x, y) = transform_coord(projGeo, projGrd, lon, lat)

    # Get bbox from data
    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Construct solution grid - add border to grid
    (Xi, Yi) = make_grid(xmin - dx, xmax + dx, ymin - dy, ymax + dy, dx, dy)

    # Flatten prediction grid
    xi = Xi.ravel()
    yi = Yi.ravel()

    # Zip data to vector
    coord = zip(x.ravel(), y.ravel())

    print 'building the k-d tree ...'

    # Construct KD-Tree
    tree = cKDTree(coord)
    
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

    # Cross-calibration container
    h_cal = np.zeros(elev.shape)

    # Temporal coverage
    t_pct = np.zeros(elev.shape)

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
            idx = tree.query_ball_point((xi[i], yi[i]), dr[ii])

            # Check for empty arrays
            if len(time[idx]) == 0:
                continue

            # Constraints parameters
            dt   = np.max(time[idx]) - np.min(time[idx])
            nobs = len(time[idx])
            nsen = len(np.unique(mode[idx]))

            # Bin time vector
            t_sample = binning(time[idx], time[idx], time[idx].min(), time[idx].max(), 1.0/12., 5, 5)[1]

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
    
        # Local calibration vector
        h_cal_cap = np.zeros(hcap.shape)

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
        dt = tcap - tref

        # Create design matrix for alignment
        Acap, cols = design_matrix(dt, mcap)
        
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
        # NOT USED RIGHT NOW! COULD BE USED FOR CHECKING CAL. IMPROVEMENTS FOR FIT-CAL AND RES-CAL.
        rms = mad_std(dh)

        # Bias correction
        h_cal_fit = np.dot(Acap[:,cols], Cm[cols])

        # Remove inter satellite biases
        hcap -= h_cal_fit
        
        # Initiate residual cross-calibration flag
        flag = 0

        # Apply residual cross-calibration
        if rcali:

            # Create residual cross-calibration index vector
            msat = np.ones(mcap.shape) * np.nan

            # Set overlap indexes
            msat[(mcap == 7) | (mcap == 5)] = 0 # ERS-1 ocean, ERS-1 ice
            msat[(mcap == 6) | (mcap == 4)] = 1 # ERS-2 ocean, ERS-2 ice
            msat[(mcap == 3) | (mcap == 0)] = 2 # ENV-1, ICE-1
            msat[(mcap == 1) | (mcap == 2)] = 3 # LRM, SIN

            # Apply post-fit residual cross-calibration in overlapping areas
            h_cal_res, flag = cross_calibrate(tcap.copy(), hcap.copy(), dh.copy(), msat.copy(), 2.0)

            # Correct for second bias
            hcap -= h_cal_res

            # Compute total correction
            h_cal_tot = h_cal_fit + h_cal_res

        # Only apply correction from fit
        else:
            
            # Set residual crosscal vector to zero
            h_cal_res = np.zeros(h_cal_fit.shape)
            
            # Only provide overall least-squares adjustment
            h_cal_tot = h_cal_fit + h_cal_res
        
        # Save as independent time series
        if serie:

            # Transform coordinates
            (lon_i, lat_i) = transform_coord(projGrd, projGeo, xcap, ycap)
            (lon_0, lat_0) = transform_coord(projGrd, projGeo, xi[i], yi[i])

            # Apply calibration if true
            if apply:

                # Apply calibration to original vector
                horg -= h_cal_tot

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

            # Save data to specific file
            ofile = ifile.replace('.h5', '.bin')

            # Save using deepdish to hdf5
            dd.io.save(ofile, {'lat': lats, 'lon': lons, 'lat0': lat0, 'lon0': lon0, 'dh_ts': h_ts, 'de_ts': e_ts, \
                               'm_idx': m_id, 'h_cal_tot': h_ct, 'h_cal_fit': h_cf, 'h_cal_res': h_cr, \
                               'h_cal_flg': f_cr, 'dxy0': dxy0, 't_year': tobs}, compression='lzf')

        # Save as point correction
        else:

            """ This part here needs to be fully checked!!! """

            # Find out if we need to update cell
            i_update, = np.where(t_pct[idx] <= npct)

            # Only keep the indices/values that need update
            idx_new = [idx[ki] for ki in i_update]

            # Set and update values
            h_cal_tot_new = h_cal_tot[i_update]

            # Populate calibration vector
            h_cal[idx_new] = h_cal_tot_new
            t_pct[idx_new] = npct

            # Save bs params as external file
            with h5py.File(ifile, 'a') as fi:

                # Delete calibration variable
                try:
                    del fi['h_cal']
                except:
                    pass

                # Save calibration
                fi['h_cal'] = h_cal

                # Correct elevations if true
                if apply:

                    # Try to create variable
                    try:
                        # Save
                        fi[zvar] = elev - h_cal
                    except:
                        # Update
                        fi[zvar][:] = elev - h_cal

        # Print meta data to terminal
        if (i % 10) == 0:
            print 'Progress:',str(i),'/',str(len(xi)),'Rate:', np.around(Cm[0],2), 'Acceleration:', np.around(Cm[1],2)

    # Saveing the data to file
    print 'Saving data to file ...'

    """ Section for testing cross calibration by selecting random points """

    if 0:

        # Search radius
        r_search = 7.5e3

        # Number of points to draw
        n_rnd = 100

        # Initialize counter
        q = 0

        # Select random points
        ir = np.random.choice(np.arange(len(x)), n_rnd, replace=False)

        # Plot random plots for testing
        while q < n_rnd:

            # Get obs. around ROI
            idx_rand = tree.query_ball_point((x[ir][q], y[ir][q]), r_search)

            # Apply correction
            h_corr = elev[idx_rand] - h_cal[idx_rand]

            # Set larges values to NaN for easier vizulization
            h_corr[np.abs(h_corr) > 25] = np.nan

            # Time vector of data
            t_rnd = time[idx_rand]
            
            # Select missions
            mission = mode[idx_rand]
            
            # Increase counter
            q += 1

            # Plot the time series
            plt.figure()
            plt.scatter(t_rnd, h_corr, s=10, c=mission, alpha=0.75)
            plt.show()

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

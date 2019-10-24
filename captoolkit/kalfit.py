#!/usr/bin/env python
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import sys
import h5py
import glob
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import deepdish as dd
from datetime import datetime
from pykalman import KalmanFilter
from pykalman.sqrt import CholeskyKalmanFilter
from numpy import ma
import matplotlib.pyplot as plt

def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

def bin_mission(ti, hi, mi, ei, tstart, tstop, tstep, p0, tref, ci):
    """ Binning of multi-mission data """

    # Get number of unique missions
    mu = np.unique(mi)

    # Get the size of the final vector
    tb = np.arange(tstart, tstop+tstep, tstep)

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

        # Get the mission specific error - single measurement error
        m_rms = ei[im].mean() / np.sqrt(2.0)

        # If Geosat
        if mu[i] == 8:
            win = 6/12.
        else:
            win = 1/12.

        # Bin the residuals according to time using the median value
        (tb, hb, eb, nb) = binning2(ti[im], hi[im], tstart, tstop, tstep, window=win, median=True)[0:4]

        # Set to mission rms
        eb[eb < 0.01] = m_rms

        # Effective value for standard error
        if len(hb[~np.isnan(hb)]) != 0:

            # Number of data points
            n_dat = len(hb[~np.isnan(hb)])

            # De-correlation scale of 2 months
            n_eff = (n_dat * (1./12)) / (2. * (3./12))

        else:

            # Set to one otherwise
            n_eff = 1.0

        # Time difference
        dt = tb - tref

        # Least squares model
        A0 = np.vstack((np.ones(tb.shape), dt, 0.5*dt**2,\
                      np.cos(2*np.pi*dt), np.sin(2*np.pi*dt))).T

        # Model values
        h_model = np.dot(A0, p0)

        # Difference from overall model
        em = hb - h_model

        # Compute standard binning error
        eb /= np.sqrt(n_eff)

        # Copy variable
        es = eb.copy()

        # Set systematic error
        es[~np.isnan(es)] = m_rms

        # Total error
        et = np.sqrt(es ** 2 + eb ** 2 + em ** 2)

        # Stack output data
        hbi[i, :] = hb      # Time series
        ebi[i, :] = et      # RSS combined systematic, random and model error
        mbi[i, :] = mu[i]   # Mission index
        tbi[i, :] = tb      # Time vector
        nbi[i, :] = nb      # Number of observations in bin

    return tbi, hbi, ebi, nbi, mbi

def binfilter(t, h, m, dt, alpha, mode=0):
    """ Outlier filtering using bins """
    
    # Unique missions
    mi = np.unique(m)

    # Copy output vector
    hi = h.copy()

    # Loop trough missions
    for kx in range(len(mi)):
                    
        # Get indexes of missions
        im = m == mi[kx]
        
        # Create monthly bins
        bins = np.arange(t[im].min(), t[im].max() + dt, dt)

        # Get data from mission
        tm, hm = t[im], h[im]

        # Loop trough bins
        for ky in range(len(bins) - 1):

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
            
            # Determine filtering process
            if mode == 0:
                
                # Set data in bin to nan
                hb[io] = np.nan
            
            else:
                
                # Set to mean value
                hb[io] = np.nanmean(hb[~io])
            
            # Set data
            hm[idx] = hb

        # Set array!
        hi[im] = hm

    return hi


def binning2(x, y, xmin=None, xmax=None, dx=1 / 12.,
             window=3 / 12., interp=False, median=False):
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

    steps = np.arange(xmin, xmax, dx)  # time steps
    bins = [(ti, ti + window) for ti in steps]  # bin limits

    N = len(bins)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in range(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            xb[i] = 0.5 * (t1 + t2)
            continue

        ybv = y[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)
        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        yb = np.interp(x, xb, yb)
        eb = np.interp(x, xb, eb)
        sb = np.interp(x, xb, sb)
        xb = x

    return xb, yb, eb, nb, sb


def runfilter(t, h, m, alpha, win):
    """ Running median time series filter """
    # Unique missions
    mi = np.unique(m)

    # Copy output vector
    hi = h.copy()

    # Loop trough missions
    for kx in range(len(mi)):

        # Get indexes of missions
        im = m == mi[kx]

        # Get data from mission
        tm, hm = t[im], h[im]

        # Smooth time series for each mission
        hs = binning2(tm.copy(), hm.copy(), window=win, \
                      interp=True, median=True)[1]

        # Compute difference
        dh = hm - hs

        # Identify outliers
        io = np.abs(dh) > alpha * mad_std(dh)

        # Set data in bin to nan
        hm[io] = np.nan

        # Add back data
        hi[im] = hm.copy()

    return hi

def binning(x, y, xmin, xmax, dx, tol, thr):
    """ Data binning of two variables """

    bins = np.arange(xmin, xmax + dx, dx)

    xb = np.arange(xmin, xmax, dx) + 0.5 * dx
    yb = np.ones(len(bins) - 1) * np.nan
    eb = np.ones(len(bins) - 1) * np.nan
    nb = np.ones(len(bins) - 1) * np.nan
    sb = np.ones(len(bins) - 1) * np.nan

    for i in range(len(bins) - 1):

        idx = (x >= bins[i]-0.02) & (x <= bins[i + 1]+0.02)

        if len(y[idx]) == 0:
            continue

        ybv = y[idx]

        yb[i] = np.nanmedian(ybv)
        eb[i] = mad_std(ybv)
        nb[i] = len(ybv)
        sb[i] = np.nansum(ybv)

    return xb, yb, eb, nb, sb


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


def fcheck(f):
    """ Checks if file has already been processed """
    
    # Get path
    path = f[0][:f[0].rfind('/')]
    
    # Get processed and original files
    fo = glob.glob(path+'/'+'*.bin')
    fp = glob.glob(path+'/'+'*kal_ts.h5')
    
    # Initiate process flag
    flag = np.zeros(len(fo))
    
    # New file list
    fout = []

    # Loop through files
    for kx in range(len(fo)):
        for ky in range(len(fp)):
            if fo[kx].replace('.bin','') == fp[ky].replace('_kal_ts.h5',''):
                flag[kx] = 1
                break
    
    # Save list of files not processed
    for i in range(len(fo)):
        if flag[i] == 0:
            fout.append(fo[i])

    return fout


def group_model(hbi, ebi, rt, order):
    """ Create variable matrices """

    # Create observational matrix
    H = np.zeros((len(hbi), order+1))
        
    # Populate observational matrix
    if order > 2:
        H[:,[0,2]] = 1
    else:
        H[:, 0] = 1

    # Mean error
    E = np.diag(rt)
    
    # Initiate output
    Ht = []
    Et = []
    
    # Loop through time series
    for i in range(len(hbi.T)):
        
        # Get current epoch
        hb = hbi[:,i]
        eb = ebi[:,i]
        
        # No data available
        if np.all(np.isnan(hb)):
            
            # All obs. -> interpolate
            Ht.append(H * 0.0)
            Et.append(E)
    
        # Only use obs. entries
        else:
            
            # Copy array to make changes
            Hi = H.copy()

            # Set observability to zero for specific entries
            Hi[:,0][np.isnan(hb)] = 0.0

            # Set and pad errors
            eb[np.isnan(eb)] = rt[np.isnan(eb)]
            
            # Append edited observability
            Ht.append(Hi)
            Et.append(np.diag(eb))
    
    return Ht, Et


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
            p = np.zeros((1,n)) * np.nan
            s = np.nan

    return p[::-1], s


def mask(x):
    """ Mask the array accordingly """

    # Set values to one
    x[~np.isnan(x)] = 1
    
    # Length of vector
    n = len(x)
    
    y = np.zeros(x.shape)
    
    # Loop trough values
    for i in range(n-1):
        
        if i == 0:
            if x[i] == 1:
                #y[i+3] = 1
                #y[i+2] = 1
                y[i+1] = 1
                y[i]   = 1

        if i == n-1:
            if x[i] == 1:
                y[i]   = 1
                y[i+1] = 1
                #y[i-2] = 1
                #y[i-3] = 1
        else:
            if x[i] == 1:
                #y[i+3] = 1
                #y[i+2] = 1
                y[i+1] = 1
                y[i]   = 1
                y[i-1] = 1
                #y[i-2] = 1
                #y[i-3] = 1
    return y


def mrms(dh,mi):
    """ Estimate rms value per mission """
    
    # Unique values
    mu = np.unique(mi)
    
    # Get myself a vector to save data
    rms = np.zeros(mu.shape) * np.nan
    mis = np.zeros(mu.shape) * np.nan
    nis = np.zeros(mu.shape) * np.nan

    # Loop trough missions
    for i in range(len(mu)):
        
        # Get index from vector
        idx = mi == mu[i]
        
        # Get data
        dhm = dh[idx]
        
        # Number of samples
        n = len(dhm[np.isnan(dhm)])
        
        if (n == 0) or np.all(np.isnan(dhm)):
            continue
        
        # Compute rms value
        rms[i] = mad_std(dhm)
        
        # Save index
        mis[i] = mu[i]

    return rms, mis, nis

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
        '-v', metavar=('x','y','t','h','s','i','b'), dest='vnames', type=str, nargs=7,
        help=('name of varibales in the HDF5-file'),
        default=['lon','lat','t_year','h_cor','m_rms','m_id','h_bs'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files, optional",
        default=[1],)

parser.add_argument(
        '-s', metavar=('tstep'), dest='tstep', type=float, nargs=1,
        help="time step of solution (months)",
        default=[1.0],)

parser.add_argument(
        '-f', dest='proc', action='store_true',
        help=('check for already processed files'),
        default=False)

# Populate arguments
args = parser.parse_args()

# Pass arguments to internal variables
files  = args.files[:]
nlim   = args.minobs[0]
t1lim  = args.tspan[0]
t2lim  = args.tspan[1]
tref   = args.tref[0]
icol   = args.vnames[:]
tstep_ = args.tstep[0]
niter  = args.niter[0]
njobs  = args.njobs[0]
proc   = args.proc

print('parameters:')
for p in vars(args).items(): print(p)

# Main program
def main(ifile, n=''):
    
    # Input variables names
    xvar, yvar, tvar, zvar, svar, ivar, ovar = icol

    # Load variables 
    lon_i = dd.io.load(ifile, '/'+xvar)
    lat_i = dd.io.load(ifile, '/'+yvar)
    time  = dd.io.load(ifile, '/'+tvar)
    elev  = dd.io.load(ifile, '/'+zvar)
    sigma = dd.io.load(ifile, '/'+svar)
    mode  = dd.io.load(ifile, '/'+ivar)
    mdims = dd.io.load(ifile, '/'+ovar)

    # Get months
    tstep = tstep_ / 12.0
    
    # Number of months of time series
    months = len(np.arange(t1lim, t2lim + tstep, tstep))

    # Total number of columns
    ntot = months + 4

    # Create output array
    OFILE1 = np.ones((len(lon_i), ntot)) * 9999
    OFILE2 = np.ones((len(lon_i), ntot)) * 9999

    # Container for trend
    trend = []

    # Enter prediction loop
    for i in range(len(lon_i)):
        
        # Selected variables
        tcap = time[i]
        hcap = elev[i]
        scap = sigma[i]
        mcap = mode[i]
        mdim  = mdims[i]

        # Original time series
        horg = hcap.copy()

        # Set all ERS-2 Ocean to NaN ######################### REMEMBER TO FIX LATER
        #hcap[mcap == 5] = np.nan

        # Don't use if we gave to little points
        if len(hcap[~np.isnan(hcap)]) < nlim: continue

        # Time difference
        dt = tcap - tref

        # Design matrix for adjustment procedure and cleaning
        X = np.vstack((np.ones(dt.shape), dt, 0.5 * dt ** 2, \
                       np.cos(2 * np.pi * dt), np.sin(2 * np.pi * dt))).T

        # Try to solve model
        try:
            
            # Solve system and get coeff
            model = sm.RLM(hcap[tcap > 1990], X[tcap > 1990,:], missing='drop').fit(maxiter=niter)

        except:
            
            # Solution failed!
            print("Solution invalid!")
            continue
        
        # Get parameters 
        Cm = model.params
        Ce = model.bse
        
        # Compute residuals
        dh = hcap - np.dot(X, Cm)
        #di = horg - np.dot(X, Cm)

        # Only operate on the residuals
        #hcap = dh + np.dot(X[:,[2,3,4]], Cm[[2,3,4]])
        #horg = di + np.dot(X[:,[2,3,4]], Cm[[2,3,4]])

        # Save rates for each location
        trend.append([lon_i[i], lat_i[i], Cm[1], Cm[2], tref])

        # RMSE value of model
        rmse = mad_std(dh)

        # Keep good time series only
        if rmse > 20 or Cm[1] > 5: continue

        # Recreate the arrays
        tbi = tcap.reshape(mdim)
        hbi = hcap.reshape(mdim)
        ebi = scap.reshape(mdim)
        mbi = mcap.reshape(mdim)
        horg = horg.reshape(mdim)

        # Identify all nan rows
        i_m = ~np.all(np.isnan(hbi), axis=1)

        # Remove nan rows in arrays
        tbi, hbi, ebi, mbi, horg = tbi[i_m, :], hbi[i_m, :], ebi[i_m, :], mbi[i_m, :], horg[i_m, :]

        # Compute mean RMS for each mission
        rms = np.nanmean(ebi, 1)
        
        # Get the unique values
        mis = np.unique(mbi)

        # Don't use time series with less than two months
        if len(hbi[~np.isnan(hbi)]) < 2: continue
        
        # Size of original observational matrix
        (n, m) = hbi.shape

        # Take the weighted average of all mission in each bin
        (hbi_w, ebi_w) = np.ma.average(np.ma.array(hbi, mask=np.isnan(hbi)), \
                         weights=np.ma.array((1.0/ebi**2), mask=np.isnan(ebi)), \
                         axis=0, returned=True)

        # Convert back to original array, with nan's
        hbi_w = np.ma.filled(hbi_w, np.nan)
        ebi_w = np.ma.filled(ebi_w, np.nan)
        
        # Convert back to standard deviation
        ebi_w = np.sqrt(1.0 / ebi_w)

        # Save original error
        eorg = ebi.copy()

        # Mission errors
        rt = np.nanmean(eorg ** 2, 1)

        # Create zeroth order dynamical model
        Ht, Rt = group_model(hbi.copy(), ebi.copy(), rt, 0)

        """
        # Seasonal signal
        ck = np.cos(np.pi / 6)
        sk = np.sin(np.pi / 6)

        # Transition matrix
        At = [[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, +ck, +sk],
              [0.0, 0.0, -sk, +ck]]

        
        # Transition matrix
        At = [[1.0, 1.0],
              [0.0, 1.0]]
        
        # Make to numpy array
        At = np.asarray(At)
        """

        # Set new parameters
        At, m0, q0 = 1.0, 0.0, 1e6
        
        # Safety switch
        if np.all(np.isnan(hbi)): continue
        
        # Zero out all empty entries
        hbi[np.isnan(hbi)] = 0
        
        # Estimated process noise
        Qt = 0.1

        # Initial start value of filter
        y0 = np.median(hbi_w[~np.isnan(hbi_w)][0:3])

        # Initial state vector
        #m0 = [y0, 0]
        #m0 = [y0, Cm[1], Cm[3], Cm[4]]

        # Initial state covariance
        #q0 = np.diag([1e6, 1e6])
        #q0 = np.diag([0, stdev]) ** 2
        #q0 = np.diag([0, Ce[1], Ce[3], Ce[4]]) ** 2
        #q0 = np.diag([1e6, 1e6, 1e6, 1e6])

        # Convert to array
        #Qt = np.diag([0.0, 1.0]) * 0.1
        #Qt = np.diag([0.0, .5, .25, .25])**2

        # Create Kalman filter - time series filtering
        kf_short = KalmanFilter(initial_state_mean       = m0,
                                initial_state_covariance = q0,
                                transition_matrices      = At,
                                observation_matrices     = Ht,
                                observation_covariance   = Rt,
                                transition_covariance    = Qt)

        # Apply kalman smoother a second time to observations
        (dh_ts_short, dh_es_short) = kf_short.smooth(hbi.T)

        # Get time
        tbin = tbi[0, :]

        # Sum components
        dh_ts_short = np.sum(dh_ts_short, 1)

        # Recover signal by adding back trend
        dh_ts = dh_ts_short

        # Errors for short and long term processes
        dh_es = np.sqrt(0.25*np.asarray([np.diag(dh_es_short[k]).sum() for k in range(len(dh_es_short))]))

        # Convert back to NaN
        hbi[hbi == 0] = np.nan

        # Identify data outside the confidence interval
        i_ = np.abs(hbi) > np.abs(dh_ts) + 3 * dh_es

        # Set to NaN
        hbi[i_] = np.nan

        # Create zeroth order dynamical model
        Ht, Rt = group_model(hbi.copy(), ebi.copy(), rt, 0)

        # Zero out all empty entries
        hbi[np.isnan(hbi)] = 0

        # Reset the transition covariance
        Qt = 1.0

        # Create Kalman filter - time series filtering
        kf_short = KalmanFilter(initial_state_mean       = m0,
                                initial_state_covariance = q0,
                                transition_matrices      = At,
                                observation_matrices     = Ht,
                                observation_covariance   = Rt,
                                transition_covariance    = Qt)

        # Apply kalman smoother a second time to observations
        (dh_ts_short, dh_es_short) = kf_short.smooth(hbi.T)

        # Sum components
        dh_ts_short = np.sum(dh_ts_short, 1)

        # Recover signal by adding back trend
        dh_ts = dh_ts_short

        # Errors for short and long term processes
        dh_es = np.sqrt(0.25 * np.asarray([np.diag(dh_es_short[k]).sum() for k in range(len(dh_es_short))]))

        # Extract bin center
        (lon_c, lat_c) = lon_i[i], lat_i[i]
        
        # Find values to mask out
        i_mask = mask(hbi_w)
        
        # Set these values to nan's
        inoip = i_mask == 0

        # Mask output array 
        dh_ts[inoip] = np.nan
        dh_es[inoip] = np.nan

        # Make sure we center the data
        try:

            # This needs to be fixed up better to make sure we get to zero at tref!
            dh_ts -= np.interp(tref, tbin[~np.isnan(dh_ts)], dh_ts[~np.isnan(dh_ts)]) * 1

        except:

            print('Could not center!')
            pass

        # Print progress
        if (i % 1) == 0:
            
            # Get rate from least-squares solution
            rate = np.around(Cm[1]*100, 2)
            
            # Print statistics to terminal
            print(str(i) + "/" + str(len(lon_i))+" Rate: "+str(rate)+" (cm/yr)", " Sigma =",np.around(rmse,2))
        
        # Intermediate plots!
        if 0:

            # Set all hbi == 0 to NaN
            hbi[hbi == 0] = np.nan
            
            #plt.subplot(2,1,1)
            #plt.scatter(tb_i, hb_i, s=20, c=mb_i, cmap='tab10', edgecolors='k')
            #plt.subplot(2,1,2)
            plt.figure()
            plt.scatter(tbi, hbi, s=20, c=mbi, cmap='tab10', edgecolors='k')
            plt.plot(tbi[0, :], dh_ts, 'k', linewidth=2)
            plt.fill_between(tbi[0, :], dh_ts - 2 * dh_es, dh_ts + 2 * dh_es, color='gray', alpha=0.7)
            # Set all hbi == 0 to NaN
            #hbi[hbi == 0] = np.nan

            # Plot solution
            #plt.figure(figsize=(12,3))
            #plt.errorbar(tbi[0,:],dh_ts, yerr=dh_es, fmt='o', alpha=0.70, capsize=3, mfc='gray', \
            #             mec='gray', ecolor='gray')
            #plt.scatter(tcap, hcap, s=5, c=mcap, cmap='tab10')
            #plt.scatter(tbi, hbi, s=20, c=mbi, cmap='tab10', edgecolors='k')
            #plt.plot(tbi[0, :], dh_ts, 'k', linewidth=2)
            #plt.axhline(y=0)
            #plt.axvline(x=2011.045)
            #plt.figure(figsize=(12,3))
            #plt.plot(tbi[0,:],dh_es,'o')
            plt.show()
            continue

        # Save data to output files
        OFILE1[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_ts))
        OFILE2[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(hbi.T), dh_es))

    # Identify unwanted data
    I1 = OFILE1[:, 0] != 9999
    I2 = OFILE2[:, 0] != 9999

    # Remove unwnated data
    OFILE1 = OFILE1[I1, :]
    OFILE2 = OFILE2[I2, :]

    # Save our poor trend file
    np.savetxt(ifile.replace('.bin', '_trend.txt'), np.asarray(trend, dtype=np.float32), delimiter=' ', fmt="%10.4f")

    # Check if we have any data
    if len(OFILE1[:, 0]) == 0:
    
        # Print message
        print(" No data to save! ")
        return
    
    # Save solution to disk
    with h5py.File(ifile.replace('.bin', '_kal_ts.h5'), 'w') as f1:
    
        # Save adjusted and merged time series
        f1['ts'] = OFILE1

    with h5py.File(ifile.replace('.bin', '_kal_es.h5'), 'w') as f2:
    
        # Save error estimate for time series
        f2['es'] = OFILE2


# Run main program!
if njobs == 1:
    
    # Unprocessed files in folder
    if proc:
        
        # Get unprocessed files
        files = fcheck(files)
    
    # Single core
    print('running sequential code ...')
    [main(f) for f in files]

else:
    
    # Unprocessed files in folder
    if proc:
        # Get unprocessed files
        files = fcheck(files)

    # Multiple cores
    print('running parallel code (%d jobs) ...' % njobs)
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

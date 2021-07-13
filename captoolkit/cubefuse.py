#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-calibration and fusion of multi-mission altimetry data.

Compute offsets between individual data sets through
adaptive least-squares adjustment and fuse calibrated
data into a continous time series. 

"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import medfilt


# Output description of solution
description = ('Program for adaptive least-squares adjustment and optimal \
               merging of multi-mission altimetry data.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='file(s) to process (HDF5)')
parser.add_argument(
        '-t', metavar=('ref_time'), dest='tref', type=float, nargs=1,
        help=('time to reference the solution to (yr), optional'),
        default=[2010],)
parser.add_argument(
        '-i', metavar='niter', dest='niter', type=int, nargs=1,
        help=('number of iterations for least-squares adj.'),
        default=[50],)
parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[25],)
parser.add_argument(
        '-v', metavar=('x','y','t','h','e','i'), dest='vnames', type=str, nargs=6,
        help=('name of variables in the HDF5-file'),
        default=['lon','lat','t_year','h_res','m_rms','m_id'],)
parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help='for parallel processing of multiple files, optional',
        default=[1],)


def binfilter(t, h, m, window=3, n_abs=5, interp=True):
    mi = np.unique(m) 
    # Loop trough missions
    for kx in xrange(len(mi)):
        i_m = (m == mi[kx])
        ti, hi = t[i_m], h[i_m]
        hi = medfilt(hi, window)
        hi[np.abs(hi-np.nanmean(hi))>n_abs] = np.nan
        idx = ~np.isnan(hi)
        if interp and sum(idx) > 2:
            hi = np.interp(ti, ti[idx], hi[idx]) 
        h[i_m] = hi
    return h


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


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

    mi = np.unique(m)  # Unique indices
    
    cols = []

    # Add biases to design matrix
    for i in xrange(len(mi)):

        # Create offset array
        b = np.zeros((len(m),1))
            
        b[m == mi[i]] = 1.0

        # Add bias to array
        A = np.hstack((A, b))

        # Index column
        i_col = 7 + i

        # Save to list
        cols.append(i_col)

    return A, cols


def rlsq(x, y, n=1, o=5):
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
        fit = sm.RLM(y, A.T, missing='drop').fit(maxiter=o)

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


def cross_calibrate(ti, hi, dh, mi, a):
    """ Residual cross-calibration """

    # Create bias vector
    hb = np.zeros(hi.shape)
    
    # Set flag
    flag = 0
    
    # Satellite overlap periods
    to = np.array([[1995 + 05. / 12. - .5, 1996 + 05. / 12. + .5],  # ERS-1 and ERS-2 (1)
                   [2002 + 10. / 12. - .5, 2003 + 06. / 12. + .5],  # ERS-2 and RAA-2 (2)
                   [2010 + 06. / 12. - .5, 2010 + 10. / 12. + .5]]) # RAA-2 and CRS-2 (3)
                 
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
        i0_min, i0_max, i1_min, i1_max = \
                b0 - a * s0, b0 + a * s0, b1 - a * s1, b1 + a * s1
        
        # Limit of number of obs.
        if i == 0:
            nlim = 1
            i0_min, i0_max, i1_min, i1_max = 0,0,0,0
        else:
            nlim = 50

        # Test criterion
        if np.isnan(b0) or np.isnan(b1):
            # Set to zero
            b0, b1 = 0, 0
        # Test criterion
        if (n0 < nlim) or (n1 < nlim):
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

    return hb,flag


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

    for i in xrange(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        xb[i] = 0.5 * (t1+t2)

        if len(idx) == 0: continue

        ybv = y[idx]
        xbv = x[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        yb = np.interp(x, xb, yb)
        eb = np.interp(x, xb, eb)
        nb = np.interp(x, xb, nb)
        sb = np.interp(x, xb, sb)
        xb = x

    return xb, yb, eb, nb, sb


def find_nearest(arr, val):
    """Find index for "nearest" value.
    
    Parameters
    ----------
    arr : array_like, shape nd
        The array to search in (nd). No need to be sorted.
    val : scalar or array_like
        Value(s) to find.

    Returns
    -------
    out : tuple
        The index (or tuple if nd array) of nearest entry found. If `val` is a
        list of values then a tuple of ndarray with the indices of each value
        is return.

    See also
    --------
    find_nearest2

    """
    idx = []
    if np.ndim(val) == 0: val = np.array([val]) 
    for v in val: idx.append((np.abs(arr-v)).argmin())
    idx = np.unravel_index(idx, arr.shape)
    return idx


# Main program
def main(files, n=''):

    # Input variables names
    xvar, yvar, tvar, zvar, evar, ivar = icol

    # If cubes for each mission are in separate files,
    # concatenate them and generate a single cube.
    # Each mission (on individual file) will be given a unique identifier.
    for nf, ifile in enumerate(files):
        print 'processing file:', ifile, '...'

        if nf == 0:
            with h5py.File(ifile, 'r') as fi:
                x = fi[xvar][:]     # 1d
                y = fi[yvar][:]     # 1d  
                time = fi[tvar][:]  # 1d
                elev = fi[zvar][:]  # 3d
                mode = fi[ivar][:] if ivar in fi \
                        else np.full_like(time, nf)  # 1d
                sigma = fi[evar][:] if evar in fi \
                        else np.full_like(elev, np.nan)  # 3d
        else:
            with h5py.File(ifile, 'r') as fi:
                time = np.hstack((time, fi[tvar][:]))  # 1d
                elev = np.dstack((elev, fi[zvar][:]))  # 3d
                mode = np.hstack((mode, fi[ivar][:] if ivar in fi \
                        else np.full_like(fi[tvar][:], nf)))  # 1d
                sigma = np.dstack((sigma, fi[evar][:] if evar in fi \
                        else np.full_like(fi[zvar][:], np.nan)))  # 3d

    if len(np.unique(mode)) < 2:
        print 'it seems there is only one mission!'
        return

    t1, t2 = np.nanmin(time), np.nanmax(time)  ##TODO: Rethink this

    # Output containers
    zi = np.full_like(elev, np.nan)
    ei = np.full_like(elev, np.nan)
    ni = np.full_like(elev, np.nan)

    # Temporal coverage
    t_pct = np.zeros(elev.shape)

    # Minimum sampling for all mission < 81.5 deg
    nsam = 0.60
    
    # Enter prediction loop
    for i in xrange(elev.shape[0]):
        for j in xrange(elev.shape[1]):

            # Number of observations
            nobs = 0

            # Time difference
            dt = 0

            # Temporal sampling
            npct = 1

            # Number of sensors
            nsen = 0
            
            # Final test of data coverage
            #if (nobs < nlim) or (npct < 0.70): continue
            
            # Parameters for model-solution
            tcap = time[:]
            mcap = mode[:]
            hcap = elev[i,j,:]
            scap = sigma[i,j,:]

            torg = tcap.copy()
            morg = mcap.copy()
            horg = hcap.copy()
            sorg = scap.copy()

            # Least-Squares Adjustment
            # ---------------------------------
            #
            # h =  x_t + x_j + x_s
            # x = (A' A)^(-1) A' y
            # r = y - Ax
            #
            # ---------------------------------


            # Need to think of a smarth way to filter out outliears.
            # In particular those at the end of each mission-record!!!
            # Also, need to plot and see how the model fit compares to the data.
            ##FIXME ############################################################

            # compute median series
            ##NOTE: Not needed for calibrating cube series (they are clean)
            if 0:
                hcap = binfilter(tcap, hcap, mcap, window=3, n_abs=5, interp=False)

            ##FIXME ############################################################


            if sum(~np.isnan(hcap)) < nlim: continue

            #plt.figure()
            ii = mcap == np.unique(mcap)[0]
            jj = mcap == np.unique(mcap)[1]

            plt.plot(tcap[ii], hcap[ii])
            plt.plot(tcap[jj], hcap[jj])

            dt = tcap - tref  # trend component

            # Create design matrix for alignment
            Acap, cols = design_matrix(dt, mcap)
            
            try:
                # Least-squares bias adjustment
                linear_model = sm.RLM(hcap, Acap, missing='drop')
                linear_model_fit = linear_model.fit(maxiter=niter)
            except:
                print "Solution invalid!"
                continue
            
            # Coefficients and standard errors
            Cm = linear_model_fit.params
            Ce = linear_model_fit.bse
        
            # Compute model residuals
            dh = hcap - np.dot(Acap, Cm)

            # Compute RMSE of corrected residuals (fit)
            rms_fit = mad_std(dh)

            # Bias correction (mission offsets)
            h_cal_fit = np.dot(Acap[:,cols], Cm[cols])
            
            # Remove inter satellite biases
            horg -= h_cal_fit
           
            # Plot
            if 1:
                plt.figure()
                plt.plot(torg[ii], horg[ii])
                plt.plot(torg[jj], horg[jj])
                plt.show()


            ##FIXME: This doesn't work. Think of a better strategy!!!!!!!!!!!!
            ##TODO: How/Where to do this??? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # Bin full calibrated record
            if 0:
                tmed, hmed, emed, nmed = binning(torg, horg, xmin=t1, xmax=t2,
                                                 dx=1/12., window=3/12.,
                                                 median=True, interp=False)[:4]

            # Interpolate 
            '''
            try:
                i_valid = ~np.isnan(hmed)
                i_inval = np.isnan(hmed)
                hmed[i_inval] = np.interp(tmed[i_inval], tmed[i_valid], hmed[i_valid])
            except:
                continue
            '''

            # Reference final solution
            '''
            if 1:
                # To original discrete time step
                idx = find_nearest(tmed, tref)
                hmed -= hmed[idx]
            else:
                # To exact given time epoch 
                href = np.interp(tref, tmed[~np.isnan(hmed)], hmed[~np.isnan(hmed)])
            '''

            """
            zi[i,j,:] = hmed
            ei[i,j,:] = emed
            ni[i,j,:] = nmed
            """

            # Plot crosscal time series
            if 1:
                    horg[np.abs(horg)>mad_std(horg)*5] = np.nan

                    plt.figure(figsize=(12,4))
                    plt.scatter(tcap, horg, s=10, c=mcap, alpha=0.7, cmap='tab10')
                    plt.scatter(tcap, hcap, s=10, c=mcap, cmap='gray')

                    try:
                        plt.figure(figsize=(12,3.5))
                        plt.plot(tmed, hmed, '-', linewidth=2)
                        plt.ylim(np.nanmin(hmed), np.nanmax(hmed))
                        plt.xlim(t1, t2)
                    except:
                        pass

                    plt.show()
                    continue

            '''
            # Transform coordinates
            (lon_i, lat_i) = transform_coord(projGrd, projGeo, xcap, ycap)
            (lon_0, lat_0) = transform_coord(projGrd, projGeo, xi[i], yi[i])
            
            # ********************** #
            
            # Apply calibration to original data points
            horg -= h_cal_fit
                
            # Save output variables to list for each solution
            lats.append(lat_i)
            lons.append(lon_i)
            lat0.append(lat_0)
            lon0.append(lon_0)
            dxy0.append(dxy)
            h_ts.append(horg)
            e_ts.append(sorg)
            m_id.append(morg)
            h_cf.append(h_cal_fit)
            f_cr.append(flag)
            tobs.append(torg)
            rmse.append(rms_fit)
            '''
            # Transform coordinates

            # Print meta data to terminal
            if (i % 1) == 0:
                print 'Progress:',str(i),'/',str(len(xi)), \
                      'Rate:', np.around(Cm[1],2), \
                      'Acceleration:', np.around(Cm[2],2)
                        
    # Saveing the data to file
    print 'Saving data to file ...'
    
    '''
    ofile = ifile.replace('.h5', '_XCAL_FUSED.h5')
    with h5py.File(ofile, 'w') as f:
        f['h_res'] = zi.reshape(Xi.shape[0], Xi.shape[1], ti.shape[0])
        f['h_err'] = ei.reshape(Xi.shape[0], Xi.shape[1], ti.shape[0])
        f['n_obs'] = ni.reshape(Xi.shape[0], Xi.shape[1], ti.shape[0])
        f['x'] = Xi[0,:]
        f['y'] = Yi[:,0]
        f['t'] = tmed

    print 'out ->', ofile
    '''
    return


# Populate arguments
args = parser.parse_args()

# Pass arguments to internal variables
files = args.files
nlim = args.minobs[0]
tref = args.tref[0]
icol = args.vnames[:]
niter = args.niter[0]
njobs = args.njobs[0]

print 'parameters:'
for p in vars(args).iteritems(): print p

main(files)

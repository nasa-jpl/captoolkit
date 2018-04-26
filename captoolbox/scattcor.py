#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Corrects radar altimetry height to correlation with waveform parameters.

Example:
    scattcor.py -v lon lat h_res t_year -w bs lew tes -d 1 -r 4 -q 2 -p det -f /path/to/*files.h5
    scattcor.py -v lon lat h_res t_year -w bs lew tes -d 1 -r 5 -q 1 -p det -f /path/to/*files.h5

Notes:
    The (back)scattering correction is applied as:

        hc_cor = h - h_bs

"""

import os
import sys
import h5py
import pyproj
import warnings
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial import cKDTree
import timeit

# This uses random cells, plot results, and do not save data
TEST_MODE = False
USE_SEED = True
N_CELLS = 200

# If True, uses given locations instead of random nodes (for TEST_MODE)
USE_NODES = False

# Specific locations for testing: Ross, Getz, PIG
NODES = [(-158.71, -78.7584), (-124.427, -74.4377), (-100.97, -75.1478)]

# True = uses LOWESS for detrending, False = uses Robust line. Always use LOWESS!!!
LOWESS = True

# Minimum correlation for each waveform param
R_MIN = 0.1

# Mininum r-squared for multivariate fit
R2_MIN = 0.1

# Max std percentage increase after correction
P_MAX = 0.1

# Minimum points per cell to compute solution
MIN_PTS = 50

# Minimum number of months to compute solution 
MIN_MONTHS = 3

# Supress anoying warnings
warnings.filterwarnings('ignore')


def get_args():
    """ Get command-line arguments. """

    msg = 'Correct height data for surface scattering variation.'
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument(
            '-f', metavar='file', dest='files', type=str, nargs='+',
            help='HDF5 file(s) to process',
            required=True)

    parser.add_argument(
            '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
            help=('block size of grid cell (km)'),
            default=[None], required=True)

    parser.add_argument(
            '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
            help=('search radius (w/relocation) (km)'),
            default=[0],)

    parser.add_argument(
            '-q', metavar=('n_reloc'), dest='nreloc', type=int, nargs=1,
            help=('number of relocations for search radius'),
            default=[0],)

    parser.add_argument(
            '-p', metavar=None, dest='proc', type=str, nargs=1,
            help=('pre-process series for sensitivity estimation'),
            choices=('det', 'dif'), default=[None],)

    parser.add_argument(
            '-v', metavar=('lon','lat', 'h', 't'), dest='vnames',
            type=str, nargs=4,
            help=('name of x/y/z/t variables in the HDF5'),
            default=[None], required=True)

    parser.add_argument(
            '-w', metavar=('bs', 'lew', 'tes'), dest='wnames',
            type=str, nargs=3,
            help=('name of sig0/LeW/TeS parameters in HDF5'),
            default=[None], required=True)

    parser.add_argument(
            '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

    parser.add_argument(
            '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
            help="number of jobs for parallel processing",
            default=[1],)

    parser.add_argument(
            '-a', dest='apply', action='store_true',
            help=('apply correction to height in addition to saving'),
            default=False)

    return parser.parse_args()


""" Generic functions """


def binning(x, y, xmin=None, xmax=None, dx=1/12., window=3/12.,
        interp=False, median=False):
    """
    Time-series binning (w/overlapping windows).

    Args:
        x,y: time and value of time series.
        xmin,xmax: time span of returned binned series.
        dx: time step of binning.
        window: size of binning window.
        interp: interpolate binned values to original x points.
    """

    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin-dx, xmax+dx, dx)
    bins = [(ti, ti+window) for ti in steps]

    N = len(bins)

    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in xrange(N):

        t1, t2 = bins[i]
        idx, = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            continue

        ybv = y[idx]
        xbv = x[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
            xb[i] = np.nanmedian(xbv)
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


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def mad_se(x, axis=None):
    """ Robust standard error (using MAD). """
    return mad_std(x, axis=axis) / np.sqrt(np.sum(~np.isnan(x, axis=axis)))


def _sigma_filter2(x, n_sigma=3):
    """ Remove values greater than n * MAD_Std. """
    i_outlier, = np.where(np.abs(x) > n_sigma * mad_std(x)) #[1]
    #i_outlier, = np.where(np.abs(x) > n_sigma * np.nanstd(x)) #[1]
    x[i_outlier] = np.nan
    return len(i_outlier)


def _sigma_filter(x, y, n_sigma=3, frac=1/3.):
    """
    Remove values greater than n * std from the LOWESS trend.
    
    See sigma_filter()
    """
    y2 = y.copy()
    idx, = np.where(~np.isnan(y))

    # Detrend
    trend = detrend(x[idx], y[idx], lowess=True)[1]
    y2[idx] = y[idx] - trend

    # Filter
    i_outlier, = np.where(np.abs(y2) > n_sigma * mad_std(y2)) #[1]
    y[i_outlier] = np.nan

    # [1] NaNs are not included!
    return len(i_outlier)


def sigma_filter(x, y, n_sigma=3, iterative=True, lowess=False, frac=1/3./2, maxiter=5):
    """
    Robust iterative sigma filter.

    Remove values greater than n * MAD-Std [from the LOWESS trend].
    """
    n_iter = 0
    n_outliers = 1
    while n_outliers != 0 and not np.isnan(y).all():

        if lowess:
            n_outliers = _sigma_filter(x, y, n_sigma=n_sigma, frac=frac)
        else:
            n_outliers = _sigma_filter2(x, n_sigma=n_sigma)

        if not iterative or n_iter == maxiter:
            break
        n_iter += 1

    return y


def mode_filter(x, min_count=10, maxiter=3):
    """ 
    Iterative mode filter. 

    Remove values repeated 'min_count' times.
    """
    n_iter = 0
    while n_iter < maxiter:
        mod, count = mode(x[~np.isnan(x)])
        if len(count) < 1:
            break
        if count[0] > min_count:
            x[x==mod[0]] = np.nan
            n_iter += 1
        else:
            n_iter = maxiter
    return x


def median_filter(x, n_median=3):
    """ Remove values greater than n * MAD-Std. """ 
    x[np.abs(x) > n_median * mad_std(x)] = np.nan
    return x


def detrend(x, y, lowess=False, frac=1/6., poly=0):
    """
    Remove trend from time series data.

    Detrend using a Robust line fit (lowess=False), a nonparametric
    LOWESS (lowess=True), or an OLS polynomial of degree 'poly'.

    Return:
        y_resid, y_trend: residuals and trend.

    Notes:
        Use frac=1/6 (half of the standard 1/3) due to LOWESS
        being applied to the binned data (much shorter time series).
    """
    # Set flag
    flag = 0
    
    if lowess:
        # Detrend using parametric fit (bin every month)
        x_b, y_b = binning(x, y, dx=1/12., window=1/12.)[:2]
        y_trend = sm.nonparametric.lowess(y_b, x_b, frac=frac, it=2)[:,1]
        flag = 1
    elif poly != 0:
        # Detrend using OLS polynomial fit
        x_mean = np.nanmean(x)
        p = np.polyfit(x - x_mean, y, poly)
        y_trend = np.polyval(p, x - x_mean)
    else:
        # Detrend using Robust straight line
        y_trend = linefit(x, y)[1]
 
    if np.isnan(y_trend).all():
        y_trend = np.zeros_like(x)
    elif flag > 0:
        y_trend = np.interp(x, x_b[~np.isnan(y_b)], y_trend)
    else:
        pass

    return y-y_trend, y_trend


def center(*arrs):
    """ Remove mean from array(s). """
    return [a - np.nanmean(a) for a in arrs]


def normalize(*arrs):
    """ Normalize array(s) by std. """
    #return [a / np.nanstd(a, ddof=1) for a in arrs]
    return [a / mad_std(a) for a in arrs]


def corr_coef(h, bs, lew, tes):
    """ Get corr coef between h and w/f params. """ 
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) & ~np.isnan(lew) & ~np.isnan(tes))
    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]
    r_bs = np.corrcoef(bs_, h_)[0,1]
    r_lew = np.corrcoef(lew_, h_)[0,1]
    r_tes = np.corrcoef(tes_, h_)[0,1]
    return r_bs, r_lew, r_tes


def corr_grad(h, bs, lew, tes, normalize=False, robust=False):
    """ Get corr gradient (slope) between h and w/f params. """ 
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) & ~np.isnan(lew) & ~np.isnan(tes))
    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]

    if robust:
        # Robust line fit
        s_bs = linefit(bs_, h_, return_coef=True)[0]
        s_lew = linefit(lew_, h_, return_coef=True)[0]
        s_tes = linefit(tes_, h_, return_coef=True)[0]
    else:
        # OLS line fit
        s_bs = np.polyfit(bs_, h_, 1)[0]
        s_lew = np.polyfit(lew_, h_, 1)[0]
        s_tes = np.polyfit(tes_, h_, 1)[0]

    if normalize:
        s_bs /= mad_std(bs_)
        s_lew /= mad_std(lew_)
        s_tes /= mad_std(tes_)

    return s_bs, s_lew, s_tes
        

def linefit(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate: Huber, 1981).

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    assert sum(~np.isnan(y)) > 1

    X = sm.add_constant(x, prepend=False)
    y_fit = sm.RLM(y, X, M=sm.robust.norms.HuberT(), missing="drop").fit(maxiter=1,tol=0.001)
    
    if return_coef:
        if len(y_fit.params) < 2: 
            return y_fit.params[0], 0.
        else: 
            return y_fit.params[:]
    else:
        return x, y_fit.fittedvalues


""" Helper functions """


def filter_data(t, h, bs, lew, tes, n_sigma=5):
    """
    Use various filters to remove outliers.

    Replaces outliers with NaNs.
    """

    # Iterative mode filter
    h = mode_filter(h, min_count=10, maxiter=3)
    bs = mode_filter(bs, min_count=10, maxiter=3)
    lew = mode_filter(lew, min_count=10, maxiter=3)
    tes = mode_filter(tes, min_count=10, maxiter=3)

    if 1:
        # Iterative 5-sigma filter (USE LOWESS!)
        h = sigma_filter(t, h, n_sigma=n_sigma, maxiter=3, lowess=True)
        bs = sigma_filter(t, bs, n_sigma=n_sigma, maxiter=3, lowess=True)
        lew = sigma_filter(t, lew, n_sigma=n_sigma, maxiter=3, lowess=True)
        tes = sigma_filter(t, tes, n_sigma=n_sigma, maxiter=3, lowess=True)
    else:
        # Running median filter - three months
        h = box_filter(t, h, n_sigma=n_sigma)
        bs = box_filter(t, bs, n_sigma=n_sigma)
        lew = box_filter(t, lew, n_sigma=n_sigma)
        tes = box_filter(t, tes, n_sigma=n_sigma)
    
    # Non-iterative 5-median filter
    h = median_filter(h, n_median=5)

    return t, h, bs, lew, tes


def interp_params(t, h, bs, lew, tes):
    """
    Interpolate waveform parameters based on height series valid entries.

    See also:
        interp_params2()
    """

    params = [bs, lew, tes]

    # Find the number of valid entries
    npts = [sum(~np.isnan(x)) for x in params]

    # Determine all the entries that should have valid data
    isvalid = ~np.isnan(h) 
    n_valid = sum(isvalid)

    # Do nothing if params are empty or have the same valid entries
    if np.all(npts == n_valid):
        return params 

    # Sort indices for interpolation
    i_sort = np.argsort(t)

    for k, (n_p, p) in enumerate(zip(npts, [bs, lew, tes])):
        
        if n_p == n_valid:
            continue

        # Get the points that should be interpolated (if any)
        i_interp, = np.where(np.isnan(p) & isvalid)

        '''
        print 'full length:      ', len(p)
        print 'max valid entries:', n_valid
        print 'valid entries:    ', sum(~np.isnan(p))
        print 'interp entries:   ', len(i_interp)
        print t[i_interp]
        '''

        # Get sorted and clean (w/o NaNs) series to interpolate
        tt, pp = t[i_sort], p[i_sort]
        i_valid, = np.where(~np.isnan(pp))
        tt, pp = tt[i_valid], pp[i_valid]

        p_interp = np.interp(t[i_interp], tt, pp)

        p[i_interp] = p_interp

        params[k] = p

    return params



def interp_params2(t, bs, lew, tes):
    """
    Interpolate waveform parameters based on series w/largest valid entries.

    See also:
        interp_params()
    """

    params = [bs, lew, tes]

    # Find the variable with the largest amount of valid entries
    npts = [len(x[~np.isnan(x)]) for x in params]

    # Do nothing if params are empty or have the same valid entries
    if np.all(npts == npts[0]):
        return params 

    i_sort = np.argsort(t)
    i_max = np.argmax(npts)

    # Determine all the entries that should have valid data
    isvalid = ~np.isnan(params[i_max]) 

    for k,p in enumerate(params):
        
        if k == i_max:
            continue

        # Get the points that should be interpolated (if any)
        jj, = np.where(np.isnan(p) & isvalid)

        '''
        print 'full length:      ', len(p)
        print 'max valid entries:', npts[i_max]
        print 'valid entries:    ', len(p[~np.isnan(p)])
        print 'invalid entries:  ', len(jj)
        print t[jj]
        '''

        tt, pp = t[i_sort], p[i_sort]
        i_valid, = np.where(~np.isnan(pp))
        tt, pp = tt[i_valid], pp[i_valid]

        p_interp = np.interp(t[jj], tt, pp)

        p[jj] = p_interp

        params[k] = p

    return params


def get_bboxs(lon, lat, dxy, proj='3031'):
    """ Define cells (bbox) for estimating corrections. """

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Number of tile edges on each dimension 
    Nns = int(np.abs(np.nanmax(y) - np.nanmin(y)) / dxy) + 1
    New = int(np.abs(np.nanmax(x) - np.nanmin(x)) / dxy) + 1

    # Coord of tile edges for each dimension
    xg = np.linspace(x.min(), x.max(), New)
    yg = np.linspace(y.min(), y.max(), Nns)

    # Vector of bbox for each cell
    bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg

    #print 'total grid cells:', len(bboxs)
    return bboxs


def get_grid_nodes(x, y, dxy, proj='3031'):
    """ Returns the nodes of each grid cell => (x0, y0). """

    # Number of tile edges on each dimension 
    Nns = int(np.abs(np.nanmax(y) - np.nanmin(y)) / dxy) + 1
    New = int(np.abs(np.nanmax(x) - np.nanmin(x)) / dxy) + 1

    # Coord of tile edges for each dimension
    xg = np.linspace(x.min(), x.max(), New)
    yg = np.linspace(y.min(), y.max(), Nns)

    # Make grid
    xx, yy = np.meshgrid(xg, yg)

    return xx.ravel(), yy.ravel()


def get_cell_idx(lon, lat, bbox, proj=3031):
    """ Get indexes of all data points inside cell. """

    # Bounding box of grid cell
    xmin, xmax, ymin, ymax = bbox
    
    # Convert lon/lat to sterographic coordinates
    x, y = transform_coord(4326, proj, lon, lat)

    # Get the sub-tile (grid-cell) indices
    i_cell, = np.where( (x >= xmin) & (x <= xmax) & 
                        (y >= ymin) & (y <= ymax) )

    return i_cell


def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0):            #NOTE: Add min reloc dist?????
    """ Get indices of all data points inside radius. """

    # Query the Tree from the node
    idx = Tree.query_ball_point((x0, y0), r)

    ###print 'query #: 1 ( first search )'

    # Either no relocation or not enough points to do relocation
    if n_reloc < 1 or len(idx) < 2:
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

        ###print 'query #:', k+2, '( reloc #:', k+1, ')'
        ###print 'relocation dist:', reloc_dist

        # Query from the new location
        idx = Tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

    return idx


def get_scatt_cor(t, h, bs, lew, tes, proc='dif'):
    """
    Calculate scattering correction for height time series.

    The correction is given as a linear combination (multivariate fit) of
    waveform parameters. The correction time series is obtain in two steps:

        1) Fit the coefficients to the differenced/detrended series:

        det[h](t) = a det[Bs](t) + b det[LeW](t) + c det[TeS](t)

        2) Linear combination of original series using fitted coeffs:

        h_cor(t) = a Bs(t) + b LeW(t) + c TeS(t)

    where a, b, c are the "sensitivities" (or weights, or scaling factors)
    for each waveform-parameter time series.

    Args:
        t: time
        h: height change (residuals from mean topo)
        bs: backscatter coefficient
        lew: leading-edge width
        tes: trailing-edge slope
        proc: pre-process time series: 'dif'|'det'|None

    """ 
    if proc == 'dif':

        # Difference time series
        h_ = np.gradient(h)
        bs_ = np.gradient(bs)
        lew_ = np.gradient(lew)
        tes_ = np.gradient(tes)

    elif proc == 'det':

        # Detrend time series
        h_ = detrend(t, h, lowess=LOWESS)[0]
        bs_ = detrend(t, bs, lowess=LOWESS)[0]
        lew_ = detrend(t, lew, lowess=LOWESS)[0]
        tes_ = detrend(t, tes, lowess=LOWESS)[0]

    else:

        h_ = h.copy()
        bs_ = bs.copy()
        lew_ = lew.copy()
        tes_ = tes.copy()

    # Ensure zero mean of processed series
    h_, bs_, lew_, tes_ = center(h_, bs_, lew_, tes_)

    # Construct design matrix: First-order model
    A = np.vstack((bs, lew, tes)).T
    A_ = np.vstack((bs_, lew_, tes_)).T

    # Check for division by zero
    try:
        
        # Fit robust linear model on differenced series (w/o NaNs)
        #model = sm.RLM(h_, A_, M=sm.robust.norms.HuberT(), missing="drop").fit(maxiter=3)
        #model = sm.WLS(h_, A_, weights=e_, missing="drop").fit(maxiter=3)
        model = sm.OLS(h_, A_, missing="drop").fit(method='qr')

        #print model.summary()
        
        # Get multivar coefficients for Bs, LeW, TeS
        a, b, c = model.params[:3]

        # Get adjusted r-squared -> model performance metric
        # (adjusted for the model degrees of freedom; 3 in this case)
        r2 = model.rsquared_adj

        # Get p-value of F-statistics -> significance of overall fit
        # (F-test assesses multiple coefficients simultaneously)
        pval = model.f_pvalue

        # Get p-value of t-statistics -> significance of each coef
        # (t-test assesses each model coefficient individually)
        pvals = model.pvalues

        # Get linear combination of original series => h_bs = a Bs + b LeW + c TeS
        h_bs = np.dot(A, [a, b, c])

        #NOTE 1: Use np.dot instead of .fittedvalues to keep NaNs from A
        #NOTE 2: The correction is generated using the original w/f series
    
    # Set all params to zero if exception detected
    except:

        print 'MULTIVARIATE FIT FAILED, setting h_bs -> zeros'
        print 'VALID DATA POINTS in h:', sum(~np.isnan(h_))
    
        h_bs = np.zeros_like(h)
        a, b, c = np.nan, np.nan, np.nan
        r2, pval, pvals = np.nan, np.nan, [np.nan, np.nan, np.nan]
    
    return [h_bs, a, b, c, r2, pval, pvals, h_, bs_, lew_, tes_]


def std_change(t, x1, x2, detrend_=False):
    """
    Compute variance change from x1 to x2 (magnitude and percentage).

    If detrend_=True, detrend using a quadratic fit by OLS.

    """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
    if detrend_:
        x1_ = detrend(t_, x1_, poly=2)[0]  # use OLS poly fit
        x2_ = detrend(t_, x2_, poly=2)[0]
    s1 = mad_std(x1_)
    s2 = mad_std(x2_)
    delta_s = s2 - s1
    return delta_s, delta_s/s1 


def trend_change(t, x1, x2):
    """
    Compute linear-trend change from x1 to x2 (magnitude and percentage).

    """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
    x1_ -= x1_.mean()
    x2_ -= x2_.mean()
    a1 = np.polyfit(t_, x1_, 1)[0]  # use OLS poly fit
    a2 = np.polyfit(t_, x2_, 1)[0]
    delta_a = a2 - a1
    return delta_a, delta_a/np.abs(a1)


def plot(x, y, xc, yc, tc, hc, bc, wc, sc,
         hc_, bc_, wc_, sc_, hc_cor, h_bs,
         r_bc, r_wc, r_sc, s_bc, s_wc, s_sc,
         d_std, p_std, d_trend, p_trend, r2, pval, pvals):

    tc_ = tc.copy()

    # Bin variables
    hc_b = binning(tc, hc_cor, median=True, window=1/12.)[1]
    bc_b = binning(tc, bc, median=True, window=1/12.)[1]
    wc_b = binning(tc, wc, median=True, window=1/12.)[1]
    tc_b, sc_b = binning(tc, sc, median=True, window=1/12.)[:2]

    # Compute trends for plot
    ii, = np.where(~np.isnan(tc) & ~np.isnan(hc) & ~np.isnan(hc_cor))
    t_, h1_, h2_ = tc[ii], hc[ii], hc_cor[ii]
    coefs1 = np.polyfit(t_, h1_, 1)
    coefs2 = np.polyfit(t_, h2_, 1)
    trend1 = np.polyval(coefs1, tc)
    trend2 = np.polyval(coefs2, tc)

    # Mask NaNs for plotting
    mask = np.isfinite(hc_b)
    
    # Default color cycle
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(6,8))

    plt.subplot(4,1,1)
    plt.plot(tc, hc, '.')
    plt.plot(tc, hc_cor, '.')
    plt.plot(tc_b[mask], hc_b[mask], '-', color=cmap(3), linewidth=2)
    plt.plot(tc, trend1, '-', color=cmap(0), linewidth=1.5)
    plt.plot(tc, trend2, '-', color=cmap(3), linewidth=1.5)
    plt.ylabel('Height (m)')
    plt.title('Original time series')

    plt.subplot(4,1,2)
    plt.plot(tc, bc, '.')
    plt.plot(tc_b[mask], bc_b[mask], '-', linewidth=2)
    plt.ylabel('Bs (s.d.)')
    
    plt.subplot(4,1,3)
    plt.plot(tc, wc, '.')
    plt.plot(tc_b[mask], wc_b[mask], '-', linewidth=2)
    plt.ylabel('LeW (s.d.)')
    
    plt.subplot(4,1,4)
    plt.plot(tc, sc, '.')
    plt.plot(tc_b[mask], sc_b[mask], '-', linewidth=2)
    plt.ylabel('TeS (s.d.)')

    # Bin variables
    hc_b = binning(tc_, hc_, median=False, window=1/12.)[1]
    bc_b = binning(tc_, bc_, median=False, window=1/12.)[1]
    wc_b = binning(tc_, wc_, median=False, window=1/12.)[1]
    tc_b, sc_b = binning(tc_, sc_, median=True, window=1/12.)[:2]

    # mask NaNs for plotting
    mask = np.isfinite(hc_b)

    plt.figure(figsize=(6,8))

    plt.subplot(4,1,1)
    plt.plot(tc_, hc_, '.')
    plt.plot(tc_b[mask], hc_b[mask], '-')
    plt.ylabel('Height (m)')
    plt.title('Processed time series')

    plt.subplot(4,1,2)
    plt.plot(tc_, bc_, '.')
    plt.plot(tc_b[mask], bc_b[mask], '-')
    plt.ylabel('Bs (s.d.)')
    
    plt.subplot(4,1,3)
    plt.plot(tc_, wc_, '.')
    plt.plot(tc_b[mask], wc_b[mask], '-')
    plt.ylabel('LeW (s.d.)')
    
    plt.subplot(4,1,4)
    plt.plot(tc_, sc_, '.')
    plt.plot(tc_b[mask], sc_b[mask], '-')
    plt.ylabel('TeS (s.d.)')

    plt.figure()
    plt.plot(x, y, '.', color='0.6', zorder=1)
    plt.scatter(xc, yc, c=hc_cor, s=5, vmin=-1, vmax=1, zorder=2)
    plt.plot(np.nanmedian(xc), np.nanmedian(yc), 'o', color='red', zorder=3)
    plt.title('Tracks')

    plt.figure(figsize=(3,9))

    plt.subplot(311)
    plt.plot(bc_, hc_, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('Bs (s.d.)')
    plt.ylabel('h (m)')

    plt.subplot(312)
    plt.plot(wc_, hc_, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('LeW (s.d.)')
    plt.ylabel('h (m)')

    plt.subplot(313)
    plt.plot(sc_, hc_, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('TeS (s.d.)')
    plt.ylabel('h (m)')

    print 'Summary:'
    print '--------'
    print 'cor applied: ', (h_bs[~np.isnan(h_bs)] != 0).any()
    print 'std change:   %.3f m (%.1f %%)' % (round(d_std, 3), round(p_std*100, 1))
    print 'trend change: %.3f m/yr (%.1f %%)' % (round(d_trend, 3), round(p_trend*100, 1))
    print ''
    print 'r-squared: ', round(r2, 3)
    print 'p-value:   ', round(pval, 3)
    print 'p-values:  ', [round(p, 3) for p in pvals]
    print ''
    print 'r_bs:      ', round(r_bc, 3)
    print 'r_lew:     ', round(r_wc, 3)
    print 'r_tes:     ', round(r_sc, 3)
    print ''                            
    print 's_bs:      ', round(s_bc, 3)
    print 's_lew:     ', round(s_wc, 3)
    print 's_tes:     ', round(s_sc, 3)

    plt.show()


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, proc=None, apply_=False):

    if TEST_MODE:
        print '*********************************************************'
        print '* RUNNING IN TEST MODE (PLOTTING ONLY, NOT SAVING DATA) *'
        print '*********************************************************'

    print 'processing file:', ifile, '...'
    
    # Test if parameter file exists
    if '_scatgrd' in ifile.lower():
        return

    xvar, yvar, zvar, tvar = vnames
    bpar, wpar, spar = wnames

    #TIME
    #print 'loading data ...',
    #start_time = timeit.default_timer()

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:

        t = fi[tvar][:]
        h = fi[zvar][:]
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        bs = fi[bpar][:]
        lew = fi[wpar][:]
        tes = fi[spar][:]

    #TIME
    #elapsed = timeit.default_timer() - start_time
    #print elapsed, 'sec'
    
    #TIME
    #print 'transforming coord/gen output containers ...',
    #start_time = timeit.default_timer()

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Filter time
    #FIXME: Always check this!!!
    if 1:
        h[t<1992] = np.nan
        bs[t<1992] = np.nan
        lew[t<1992] = np.nan
        tes[t<1992] = np.nan

    # Get nodes of solution grid
    x_nodes, y_nodes = get_grid_nodes(x, y, dxy, proj=proj)

    """ Create output containers """

    N_data = len(x)
    N_nodes = len(x_nodes)

    # Values for each data point
    r2fit = np.full(N_data, 0.0)      # r2 of the multivar fit 
    pval = np.full(N_data, np.nan)    # r2 of the multivar fit 
    dstd = np.full(N_data, np.nan)    # magnitude std change after cor 
    dtrend = np.full(N_data, np.nan)  # magnitude trend change after cor 
    pstd = np.full(N_data, np.nan)    # perc std change after cor 
    ptrend = np.full(N_data, np.nan)  # perc trend change after cor 
    hbs = np.full(N_data, np.nan)     # scatt cor from multivar fit
    rbs = np.full(N_data, np.nan)     # corr coef h x Bs
    rlew = np.full(N_data, np.nan)    # corr coef h x LeW
    rtes = np.full(N_data, np.nan)    # corr coef h x TeS
    sbs = np.full(N_data, np.nan)     # sensit h x Bs
    slew = np.full(N_data, np.nan)    # sensit h x LeW
    stes = np.full(N_data, np.nan)    # sensit h x TeS
    bbs = np.full(N_data, np.nan)     # multivar fit coef a.Bs
    blew = np.full(N_data, np.nan)    # multivar fit coef b.LeW
    btes = np.full(N_data, np.nan)    # multivar fit coef c.TeS

    # Values for each node
    r2fitc = np.full(N_nodes, 0.0)
    pvalc = np.full(N_nodes, np.nan)
    dstdc = np.full(N_nodes, np.nan)
    dtrendc = np.full(N_nodes, np.nan)
    pstdc = np.full(N_nodes, np.nan)
    ptrendc = np.full(N_nodes, np.nan)
    rbsc = np.full(N_nodes, np.nan)
    rlewc = np.full(N_nodes, np.nan) 
    rtesc = np.full(N_nodes, np.nan) 
    sbsc = np.full(N_nodes, np.nan) 
    slewc = np.full(N_nodes, np.nan) 
    stesc = np.full(N_nodes, np.nan) 
    bbsc = np.full(N_nodes, np.nan) 
    blewc = np.full(N_nodes, np.nan) 
    btesc = np.full(N_nodes, np.nan) 
    lonc = np.full(N_nodes, np.nan) 
    latc = np.full(N_nodes, np.nan) 

    #TIME
    #elapsed = timeit.default_timer() - start_time
    #print elapsed, 'sec'

    # Select cells at random (for testing)
    if TEST_MODE:

        if USE_NODES:
            # Convert into sterographic coordinates
            x_nodes = [transform_coord('4326', '3031', xp, yp)[0] for xp, yp in NODES]
            y_nodes = [transform_coord('4326', '3031', xp, yp)[1] for xp, yp in NODES]

        else:
            if USE_SEED:
                np.random.seed(999)  # not so random!

            # Select a few random nodes
            ii = np.random.randint(0, N_nodes, N_CELLS)
            x_nodes, y_nodes = x_nodes[ii], y_nodes[ii] 

        N_nodes = len(x_nodes)

    #TIME
    #print 'building KD-Tree ...',
    #start_time = timeit.default_timer()

    # Build KD-Tree with polar stereo coords
    x, y = transform_coord(4326, proj, lon, lat)
    Tree = cKDTree(zip(x, y))

    #TIME
    #elapsed = timeit.default_timer() - start_time
    #print elapsed, 'sec'

    # Loop through nodes
    for k in xrange(N_nodes):

        if (k%1000) == 0:
            print 'Calculating correction for node', k, 'of', N_nodes, '...'

        xi, yi = x_nodes[k], y_nodes[k]

        #TIME
        #print 'querying KD-Tree w/relocations ...',
        #start_time = timeit.default_timer()
        
        # Get indices of data within search radius
        i_cell = get_radius_idx(x, y, xi, yi, radius, Tree, n_reloc=n_reloc)

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        # If cell empty or not enough data go to next node
        if len(i_cell) < MIN_PTS:
            continue

        #TIME
        #print 'extracting data for selected cell ...',
        #start_time = timeit.default_timer()
        
        # Get all data within the grid search radius
        tc = t[i_cell]
        hc = h[i_cell]
        xc = x[i_cell]
        yc = y[i_cell]
        bc = bs[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        #NOTE: Check if this is really needed!
        # Ensure all data points are sorted
        if 0:
            i_sort = np.argsort(tc)
            tc = tc[i_sort]
            hc = hc[i_sort]
            xc = xc[i_sort]
            yc = yc[i_sort]
            bc = bc[i_sort]
            wc = wc[i_sort]
            sc = sc[i_sort]

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        # Keep original (unfiltered) data
        tc_orig, hc_orig = tc.copy(), hc.copy()

        #bc0, wc0, sc0 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        #TIME
        #print 'filtering data ...',
        #start_time = timeit.default_timer()

        # Filter invalid points
        tc, hc, bc, wc, sc = filter_data(tc, hc, bc, wc, sc, n_sigma=5)

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        #bc1, wc1, sc1 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        # Test minimum number of obs in all params
        nobs = min([len(v[~np.isnan(v)]) for v in [hc, bc, wc, sc]])

        # Test for enough points
        if (nobs < MIN_PTS):
            continue

        # Bin at monthly intervals to check temporal sampling
        h_binned = binning(tc, hc, dx=1/12., window=1/12.)[1]
        n_months = sum(~np.isnan(h_binned))

        if n_months < MIN_MONTHS:
            continue

        #TIME
        #print 'interpolating time series ...',
        #start_time = timeit.default_timer()

        # Interpolate missing w/f params based on h series
        bc, wc, sc = interp_params(tc, hc, bc, wc, sc)

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        #bc2, wc2, sc2 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        # Plot interpolated time-series points
        if 0:

            bc2[bc2==bc1] = np.nan
            wc2[wc2==wc1] = np.nan
            sc2[sc2==sc1] = np.nan

            ax0 = plt.subplot(411)
            plt.plot(tc, hc_orig, '.')
            plt.plot(tc, hc, '.')
            ax1 = plt.subplot(412)
            plt.plot(tc, bc0, '.')
            plt.plot(tc, bc1, '.')
            plt.plot(tc, bc2, 'x')
            ax2 = plt.subplot(413)
            plt.plot(tc, wc0, '.')
            plt.plot(tc, wc1, '.')
            plt.plot(tc, wc2, 'x')
            ax3 = plt.subplot(414)
            plt.plot(tc, sc0, '.')
            plt.plot(tc, sc1, '.')
            plt.plot(tc, sc2, 'x')

            plt.figure()
            plt.plot(x, y, '.', color='.5', rasterized=True)
            plt.plot(xc, yc, '.')
            plt.plot(xc[~np.isnan(hc)], yc[~np.isnan(hc)], '.')

            plt.show()
            continue

        #TIME
        #print 'calculating scattering correction ...',
        #start_time = timeit.default_timer()
        
        # Ensure zero mean on all variables
        hc, bc, wc, sc = center(hc, bc, wc, sc)

        # Normalize the w/f params to std = 1
        bc, wc, sc = normalize(bc, wc, sc)

        # Calculate correction for data in search radius
        hc_bs, b_bc, b_wc, b_sc, r2, pval, pvals, hc_, bc_, wc_, sc_ = \
                get_scatt_cor(tc, hc, bc, wc, sc, proc=proc)
        
        # Calculate variance change (magnitude and perc)
        d_std, p_std = std_change(tc, hc, hc_cor, detrend_=True)
        
        # Change method if dif and p_std is true
        if p_std > 0.05 and proc == 'dif':

            # Calculate correction for data in search radius
            hc_bs, b_bc, b_wc, b_sc, r2, pval, pvals, hc_, bc_, wc_, sc_ = \
                    get_scatt_cor(tc, hc, bc, wc, sc, proc='det')

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        # If no correction could be generated, skip
        if (hc_bs == 0).all():
            continue

        # Apply correction to height
        hc_cor = hc - hc_bs

        #TIME
        #print 'calculating corr/sens/changes ...',
        #start_time = timeit.default_timer()
        
        # Calculate correlation between h and waveform params
        r_bc, r_wc, r_sc = corr_coef(hc_, bc_, wc_, sc_)

        # Calculate sensitivity values (corr grad)
        s_bc, s_wc, s_sc = corr_grad(hc_, bc_, wc_, sc_, normalize=False)

        # Calculate variance change (magnitude and perc)
        d_std, p_std = std_change(tc, hc, hc_cor, detrend_=True)

        # Calculate trend change (magnitude and perc)
        d_trend, p_trend = trend_change(tc, hc, hc_cor)
        
        # Test if at least one correlation is significant
        #r_cond = (np.abs(r_bc) < R_MIN and np.abs(r_wc) < R_MIN and np.abs(r_sc) < R_MIN)

        # Do not apply correction if:
        # - r-squared is not significant 
        # - std increases by more than 5%
        if pval > 0.05 or p_std > 0.05:

            # Cor is set to zero
            hc_cor = hc.copy()
            hc_bs[:] = 0. 

            # All params are set to zero/one
            b_bc, b_wc, b_sc = 0., 0., 0.
            r_bc, r_wc, r_sc = 0., 0., 0.
            s_bc, s_wc, s_sc = 0., 0., 0.
            r2, pval, pvals = 0., 1., (1., 1., 1.)
            d_std, p_std, d_trend, p_trend = 0., 0., 0., 0.
        
        # Set filtered out values (not used in the calculation) to NaN
        hc_bs[np.isnan(hc)] = np.nan

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'

        # Plot individual grid cells for testing
        if TEST_MODE:

            plt.figure(figsize=(6,2))
            plt.plot(tc_orig, hc_orig, '.', color='0.3')

            plot(x, y, xc, yc, tc, hc, bc, wc, sc,
                 hc_, bc_, wc_, sc_, hc_cor, hc_bs,
                 r_bc, r_wc, r_sc, s_bc, s_wc, s_sc,
                 d_std, p_std, d_trend, p_trend,
                 r2, pval, pvals)


        """ Store results (while checking previously stored estimates) """


        #TIME
        #print 'testing if update needed and saving ...',
        #start_time = timeit.default_timer()

        # Check where/if previously stored values need update (r2_prev < r2_new)
        #i_update, = np.where(pstd[i_cell] <= p_std)  # use '<=' !!!
        i_update, = np.where(r2fit[i_cell] <= r2)  # use '<=' !!!

        # Only keep the indices/values that need update
        i_cell_new = [i_cell[i] for i in i_update]  # => a list!
        hc_bs_new = hc_bs[i_update]

        # Store correction for cell (only improved values)
        hbs[i_cell_new] = hc_bs_new    # set of values
        r2fit[i_cell_new] = r2         # one value (same for all)
        dstd[i_cell_new] = d_std      
        pstd[i_cell_new] = p_std      
        dtrend[i_cell_new] = d_trend 
        ptrend[i_cell_new] = p_trend 
        rbs[i_cell_new] = r_bc         # corr coef
        rlew[i_cell_new] = r_wc        
        rtes[i_cell_new] = r_sc
        sbs[i_cell_new] = s_bc         # sensitivity
        slew[i_cell_new] = s_wc
        stes[i_cell_new] = s_sc
        bbs[i_cell_new] = b_bc         # multivar fit coef
        blew[i_cell_new] = b_wc
        btes[i_cell_new] = b_sc

        # Compute centroid of cell 
        xc_ = np.nanmedian(xc)
        yc_ = np.nanmedian(yc)

        # Convert x/y -> lon/lat 
        lon_c, lat_c = transform_coord(proj, 4326, xc_, yc_)

        # Store one s and r value per cell
        lonc[k] = lon_c
        latc[k] = lat_c
        r2fitc[k] = r2
        dstdc[k] = d_std
        pstdc[k] = p_std
        dtrendc[k] = d_trend
        ptrendc[k] = p_trend
        rbsc[k] = r_bc
        rlewc[k] = r_wc
        rtesc[k] = r_sc
        sbsc[k] = s_bc
        slewc[k] = s_wc
        stesc[k] = s_sc
        bbsc[k] = b_bc
        blewc[k] = b_wc
        btesc[k] = b_sc

        #TIME
        #elapsed = timeit.default_timer() - start_time
        #print elapsed, 'sec'


    """ Correct h (full dataset) with best values """


    if apply_:
        h[~np.isnan(hbs)] -= hbs[~np.isnan(hbs)]


    """ Save data """


    if not TEST_MODE:
        print 'saving data ...'

        with h5py.File(ifile, 'a') as fi:

            # Update h in the file and save correction (all cells at once)
            fi[zvar][:] = h
            
            # Try to create varibales
            try:
                
                # Save params for each point
                fi['h_bs'] = hbs
                fi['r2'] = r2fit
                fi['d_std'] = dstd
                fi['p_std'] = pstd
                fi['d_trend'] = dtrend
                fi['p_trend'] = ptrend
                fi['r_bs'] = rbs
                fi['r_lew'] = rlew
                fi['r_tes'] = rtes
                fi['s_bs'] = sbs
                fi['s_lew'] = slew
                fi['s_tes'] = stes
                fi['b_bs'] = bbs
                fi['b_lew'] = blew
                fi['b_tes'] = btes
            
            #FIXME: Check if this is a good idea. Content of input file is being deleted!!! 
            # Update variabels instead
            except:

                # Save params for each point
                fi['h_bs'][:] = hbs
                fi['r2'][:] = r2fit
                fi['d_std'][:] = dstd
                fi['p_std'][:] = pstd
                fi['d_trend'][:] = dtrend
                fi['p_trend'][:] = ptrend
                fi['r_bs'][:] = rbs
                fi['r_lew'][:] = rlew
                fi['r_tes'][:] = rtes
                fi['s_bs'][:] = sbs
                fi['s_lew'][:] = slew
                fi['s_tes'][:] = stes
                fi['b_bs'][:] = bbs
                fi['b_lew'][:] = blew
                fi['b_tes'][:] = btes

        # Only rename file if _SCAT has not been added
        if ifile.find('_SCAT.h5') < 0:
            
            # Rename file
            os.rename(ifile, ifile.replace('.h5', '_SCAT.h5'))
        
        # Save bs params as external file 
        with h5py.File(ifile.replace('.h5', '_SCATGRD.h5'), 'w') as fo:
            
            # Try to svave variables
            try:
                
                # Save varibales
                fo['lon'] = lonc
                fo['lat'] = latc
                fo['r2'] = r2fitc
                fo['d_std'] = dstdc
                fo['p_std'] = pstdc
                fo['d_trend'] = dtrendc
                fo['p_trend'] = ptrendc
                fo['r_bs'] = rbsc
                fo['r_lew'] = rlewc
                fo['r_tes'] = rtesc
                fo['s_bs'] = sbsc
                fo['s_lew'] = slewc
                fo['s_tes'] = stesc
                fo['b_bs'] = bbsc
                fo['b_lew'] = blewc
                fo['b_tes'] = btesc
            
            # Catch any exceptions 
            except:
                
                # Exit program
                print 'COUND NOT SAVE PARAMETERS FOR EACH CELL'
                return


if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.files[:]         # input files
    vnames = args.vnames[:]        # lon/lat/h/time variable names
    wnames = args.wnames[:]        # variables to use for correction
    dxy = args.dxy[0] * 1e3        # grid-cell length (km -> m)
    radius = args.radius[0] * 1e3  # search radius (km -> m) 
    nreloc = args.nreloc[0]        # number of relocations
    proc = args.proc[0]            # det, dif, bin or None series
    proj = args.proj[0]            # EPSG proj number
    apply_ = args.apply            # Apply cor in addition to saving
    njobs = args.njobs[0]          # parallel writing

    print 'parameters:'
    for arg in vars(args).iteritems():
        print arg

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc, apply_) \
                for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc, apply_) \
                        for ifile in ifiles)

    print 'done!'

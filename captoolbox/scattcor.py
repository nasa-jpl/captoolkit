#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Corrects radar altimetry height to correlation with waveform parameters.

Example:
    scattcor.py -d 5 -v lon lat h_res t_year -w bs lew tes \
            -n 8 -f ~/data/envisat/all/bak/*.h5

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

# This uses random cells, plot results, and do not save data
TEST_MODE = False
USE_SEED = True
N_CELLS = 100

# True = uses LOWESS regression for detrending, False = uses Robust line
LOWESS = True

# Minimum correlation for each waveform param
R_MIN = 0.1

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
            choices=('det', 'dif', 'bin'), default=[None],)

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
    trend = sm.nonparametric.lowess(y[idx], x[idx], frac=frac, it=2)[:,1]
    y2[idx] = y[idx] - trend

    # Filter
    i_outlier, = np.where(np.abs(y2) > n_sigma * mad_std(y2)) #[1]
    y[i_outlier] = np.nan

    # [1] NaNs are not included!
    return len(i_outlier)


def sigma_filter(x, y, n_sigma=3, iterative=True, lowess=False, frac=1/3., maxiter=5):
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


def detrend(x, y, lowess=False, frac=1/3.):
    """
    Detrend using Robust line (lowess=False) or nonlinear LOWESS (lowess=True).

    Return:
        y_res, y_trend: residuals and trend.
    """
    if lowess:
        y_trend = sm.nonparametric.lowess(y, x, frac=frac)[:,1]

    else:
        y_trend = linefit(x, y)[1]

    if np.isnan(y_trend).all():
        y_trend = np.zeros_like(x)

    elif np.isnan(y).any():
        y_trend = np.interp(x, x[~np.isnan(y)], y_trend)

    return y-y_trend, y_trend


def center(*arrs):
    """ Remove mean from array(s). """
    return [a - np.nanmean(a) for a in arrs]


def corr_coef(arrs, proc=None, time=None):
    """ Get corr coef between arrs[0] and arrs[1:]. """ 

    if proc == 'det':
        arrs = [detrend(time, a, lowess=LOWESS, frac=1/3.)[0] for a in arrs]

    elif proc == 'dif':
        arrs = [np.gradient(a) for a in arrs]

    else:
        pass

    x = arrs[0]
    return [np.corrcoef(x[(~np.isnan(x))&(~np.isnan(y))],
                        y[(~np.isnan(x))&(~np.isnan(y))])[0,1] \
                                for y in arrs[1:]]


def corr_grad(arrs, proc=None, time=None, normalize=False):
    """ Get corr gradient (slope) between arrs[0] and arrs[1:]. """ 

    if proc == 'det':
        arrs = [detrend(time, a, lowess=LOWESS, frac=1/3.)[0] for a in arrs]

    elif proc == 'dif':
        arrs = [np.gradient(a) for a in arrs]

    else:
        pass

    x = arrs[0]
    '''
    # OLS line fit
    sens = [np.polyfit(x[(~np.isnan(x))&(~np.isnan(y))],
                       y[(~np.isnan(x))&(~np.isnan(y))], 1)[0] \
                                for y in arrs[1:]]
    '''
    # Robust line fit
    sens = [linefit(x[(~np.isnan(x))&(~np.isnan(y))],
                    y[(~np.isnan(x))&(~np.isnan(y))], return_coef=True)[0] \
                                for y in arrs[1:]]

    if normalize:
        sens = [s/np.nanstd(v, ddof=1) for s,v in zip(sens, arrs[1:])]

    return sens
        

def linefit(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate: Huber, 1981).

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    assert sum(~np.isnan(y)) > 1

    X = sm.add_constant(x, prepend=False)
    y_fit = sm.RLM(y, X, M=sm.robust.norms.HuberT(), missing="drop").fit(maxiter=3)

    if return_coef:
        if len(y_fit.params) < 2: 
            return y_fit.params[0], 0.
        else: 
            return y_fit.params[:]
    else:
        return x, y_fit.fittedvalues


""" Helper functions """

def filter_data(t, h, bs, lew, tes):
    """
    Use various filters to remove outliers.

    It adds NaNs in place of filtered outliers.
    """

    # Iterative mode filter
    h = mode_filter(h, min_count=10, maxiter=3)
    bs = mode_filter(bs, min_count=10, maxiter=3)
    lew = mode_filter(lew, min_count=10, maxiter=3)
    tes = mode_filter(tes, min_count=10, maxiter=3)

    # Iterative 5-sigma filter (USE LOWESS!)
    h = sigma_filter(t, h, n_sigma=5, frac=1/3., maxiter=3, lowess=True)
    bs = sigma_filter(t, bs, n_sigma=5,frac=1/3., maxiter=3, lowess=True)
    lew = sigma_filter(t, lew, n_sigma=5,frac=1/3., maxiter=3, lowess=True)
    tes = sigma_filter(t, tes, n_sigma=5,frac=1/3., maxiter=3, lowess=True)

    # Non-iterative median filter
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


def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0,
        min_months=24, max_reloc=3, time=None, height=None):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the center of cell 
    idx = Tree.query_ball_point((x0, y0), r)

    print 'query #: 1 ( first search )'

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

        print 'query #:', k+2, '( reloc #:', k+1, ')'
        print 'relocation dist:', reloc_dist

        idx = Tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

        # If time provided, keep relocating until time-coverage is sufficient 
        if time is not None:

            t_b, x_b = binning(time[idx], height[idx], dx=1/12., window=1/12.)[:2]

            print 'months #:', np.sum(~np.isnan(x_b))

            # If sufficient coverage, exit
            if np.sum(~np.isnan(x_b)) >= min_months:
                break

    return idx


def get_scatt_cor(t, h, bs, lew, tes, proc=None):
    """
    Calculate scattering correction for height time series.

    The correction is given as a linear combination (multivariate fit)
    of waveform parameters as:

        h_cor(t) = a Bs(t) + b LeW(t) + c TeS(t)

    where a, b, and c are the "sensitivity" of h to each waveform param,
    and they are derived separately by fitting the above model using 
    differenced/detrended time series of waveform params.

    Args:
        t: time
        h: height change (residuals from mean topo)
        bs: backscatter coefficient
        lew: leading-edge width
        tes: trailing-edge slope
        proc: pre-process time series: 'bin'|'det'|'dif'
    """ 

    # Construct design matrix: First-order model
    A_orig = np.vstack((bs, lew, tes)).T

    # Bin time series
    if proc == 'bin':

        # Need enough data for binning (at least 1 year)
        if t.max() - t.min() < 1.0:

            print 'BINNED'
            h = binning(t, h, dx=1/12., window=3/12., interp=True)[1]
            bs = binning(t, bs, dx=1/12., window=3/12., interp=True)[1]
            lew = binning(t, lew, dx=1/12., window=3/12., interp=True)[1]
            tes = binning(t, tes, dx=1/12., window=3/12., interp=True)[1]

    # Detrend time series
    elif proc == 'det':

        print 'DETRENDED'
        h = detrend(t, h, lowess=LOWESS)[0]
        bs = detrend(t, bs, lowess=LOWESS)[0]
        lew = detrend(t, lew, lowess=LOWESS)[0]
        tes = detrend(t, tes, lowess=LOWESS)[0]

    # Difference time series
    elif proc == 'dif':

        print 'DIFFERENCED'
        h = np.gradient(h)
        bs = np.gradient(bs)
        lew = np.gradient(lew)
        tes = np.gradient(tes)

    else:

        A_proc = A_orig

    # If data has been preprocessed
    if proc: 

        h, bs, lew, tes = center(h, bs, lew, tes)
        A_proc = np.vstack((bs, lew, tes)).T
    
    # Temporal sampling
    t_s = binning(t, h, dx=1/12., window=1/12.)[1]
    
    # Criteria for fitting procedure
    n_mth = np.sum(~np.isnan(t_s))
    n_obs = len(h[~np.isnan(h)])
             
    # Check sampling for fit.
    if (n_mth >= 6) and (n_obs > 10):
        
        # Check for division by zero
        try:
            
            # Fit robust linear model on clean data
            model = sm.RLM(h, A_proc, M=sm.robust.norms.HuberT(), missing="drop").fit(maxiter=3)
            #model = sm.OLS(h, A_proc, missing="drop").fit(maxiter=3)

            # Get multivar coefficients for Bs, LeW, TeS
            a_bs, b_lew, c_tes = model.params[:3]

            # Get multivariate fit => h_bs = a Bs + b LeW + c TeS
            h_bs = np.dot(A_orig, [a_bs, b_lew, c_tes])

            #NOTE 1: Use np.dot instead of .fittedvalues to keep NaNs from A
            #NOTE 2: Correction is generated using the original parameters
        
        # Set all params to zero if exception detected
        except:
            
            # Not enough data!
            print 'COULD NOT DO MULTIVARIATE FIT. Bs_cor -> zeros'
            h_bs = np.zeros_like(h)
            a_bs, b_lew, c_tes = 0., 0., 0.
    else:
        
        # Not enough data!
        print 'COULD NOT DO MULTIVARIATE FIT. Bs_cor -> zeros'
        h_bs = np.zeros_like(h)
        a_bs, b_lew, c_tes = 0., 0., 0.
    
    return [h_bs, a_bs, b_lew, c_tes]


def apply_scatt_cor(t, h, h_bs, filt=False, test_std=False):
    """ Apply correction (if decreases std of residuals). """

    h_cor = h - h_bs 

    if filt:
        h_cor = sigma_filter(t, h_cor,  n_sigma=3,  frac=1/3., lowess=True, maxiter=1)

    if test_std:
        p_std = std_change(t, h, h_cor, detrend_=True, lowess=LOWESS)

        # Do not apply cor if std of corrected resid increases
        if p_std > 0:
            h_cor = h.copy()
            h_bs[:] = 0.  # cor is set to zero

    return h_cor, h_bs


def std_change(t, x1, x2, detrend_=False, lowess=False):
    """ Compute the perc. variance change from x1 to x2 @ valid y. """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
    if detrend_:
        x1_ = detrend(t_, x1_, lowess=lowess)[0]
        x2_ = detrend(t_, x2_, lowess=lowess)[0]
    s1 = x1_.std(ddof=1)
    s2 = x2_.std(ddof=1)
    return (s2 - s1) / s1 


def trend_change(t, x1, x2):
    """ Compute the perc. trend change from x1 to x2 @ valid y. """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
    x1_ -= x1_.mean()
    x2_ -= x2_.mean()
    a1 = linefit(t_, x1_, return_coef=True)[0]
    a2 = linefit(t_, x2_, return_coef=True)[0]
    return (a2 - a1) / np.abs(a1)


def plot(xc, yc, tc, hc, bc, wc, sc, hc_cor, h_bs,
        x_full, y_full, proc=None):

    # Plot only corrected points
    idx = ~np.isnan(hc_cor) & ~np.isnan(h_bs)
    
    if len(idx) == 0:
        return
    
    if proc == 'det':
        hc_proc = detrend(tc, hc, lowess=LOWESS)[0]
        bc_proc = detrend(tc, bc, lowess=LOWESS)[0]
        wc_proc = detrend(tc, wc, lowess=LOWESS)[0]
        sc_proc = detrend(tc, sc, lowess=LOWESS)[0]
        tc_proc = tc.copy()

    elif proc == 'dif':
        hc_proc = np.gradient(hc)
        bc_proc = np.gradient(bc)
        wc_proc = np.gradient(wc)
        sc_proc = np.gradient(sc)
        tc_proc = tc.copy()

    else:
        hc_proc = hc.copy()
        bc_proc = bc.copy()
        wc_proc = wc.copy()
        sc_proc = sc.copy()
        tc_proc = tc.copy()

    # Correlate variables
    r_bc, r_wc, r_sc = corr_coef([hc, bc, wc, sc], proc=proc, time=tc)
    r_bc2, r_wc2, r_sc2 = corr_coef([hc_cor, bc, wc, sc], proc=proc, time=tc)

    # Sensitivity values
    s_bc, s_wc, s_sc = corr_grad([hc, bc, wc, sc], proc=proc, time=tc)

    # Normalized sensitivity values
    s_bc2, s_wc2, s_sc2 = corr_grad([hc, bc, wc, sc], proc=proc, time=tc, normalize=True)

    # Detrend both time series for estimating std(res)
    hc_r = detrend(tc, hc, lowess=LOWESS)[0]
    hc_cor_r = detrend(tc, hc_cor, lowess=LOWESS)[0]

    # Std before and after correction
    idx, = np.where(~np.isnan(hc_r) & ~np.isnan(hc_cor_r))
    std1 = hc_r[idx].std(ddof=1)
    std2 = hc_cor_r[idx].std(ddof=1)

    # Percentage change
    p_std = std_change(tc, hc, hc_cor, detrend_=True, lowess=LOWESS)
    p_trend = trend_change(tc, hc, hc_cor)

    # Bin variables
    hc_b = binning(tc[idx], hc_cor[idx], median=True, window=1/12.)[1]
    bc_b = binning(tc[idx], bc[idx], median=True, window=1/12.)[1]
    wc_b = binning(tc[idx], wc[idx], median=True, window=1/12.)[1]
    tc_b, sc_b = binning(tc[idx], sc[idx], median=True, window=1/12.)[:2]

    # mask NaNs for plotting
    mask = np.isfinite(hc_b)
    
    plt.figure(figsize=(6,8))

    plt.subplot(4,1,1)
    plt.plot(tc[idx], hc[idx], '.')
    plt.plot(tc[idx], hc_cor[idx], '.')
    plt.plot(tc_b[mask], hc_b[mask], '-', linewidth=2)
    plt.ylabel('Height (m)')
    plt.title('Original time series')

    plt.subplot(4,1,2)
    plt.plot(tc[idx], bc[idx], '.')
    plt.plot(tc_b[mask], bc_b[mask], '-', linewidth=2)
    plt.ylabel('Bs (dB)')
    
    plt.subplot(4,1,3)
    plt.plot(tc[idx], wc[idx], '.')
    plt.plot(tc_b[mask], wc_b[mask], '-', linewidth=2)
    plt.ylabel('LeW (m)')
    
    plt.subplot(4,1,4)
    plt.plot(tc[idx], sc[idx], '.')
    plt.plot(tc_b[mask], sc_b[mask], '-', linewidth=2)
    plt.ylabel('TeS (?)')

    # Bin variables
    hc_proc_b = binning(tc_proc, hc_proc, median=False, window=1/12.)[1]
    bc_proc_b = binning(tc_proc, bc_proc, median=False, window=1/12.)[1]
    wc_proc_b = binning(tc_proc, wc_proc, median=False, window=1/12.)[1]
    tc_proc_b, sc_proc_b = binning(tc_proc, sc_proc, median=True, window=1/12.)[:2]

    # mask NaNs for plotting
    mask = np.isfinite(hc_proc_b)

    plt.figure(figsize=(6,8))

    plt.subplot(4,1,1)
    plt.plot(tc_proc, hc_proc, '.')
    plt.plot(tc_proc_b[mask], hc_proc_b[mask], '-')
    plt.ylabel('Height (m)')
    plt.title('Processed time series')

    plt.subplot(4,1,2)
    plt.plot(tc_proc, bc_proc, '.')
    plt.plot(tc_proc_b[mask], bc_proc_b[mask], '-')
    plt.ylabel('Bs (dB)')
    
    plt.subplot(4,1,3)
    plt.plot(tc_proc, wc_proc, '.')
    plt.plot(tc_proc_b[mask], wc_proc_b[mask], '-')
    plt.ylabel('LeW (m)')
    
    plt.subplot(4,1,4)
    plt.plot(tc_proc, sc_proc, '.')
    plt.plot(tc_proc_b[mask], sc_proc_b[mask], '-')
    plt.ylabel('TeS (?)')

    plt.figure()
    plt.plot(x_full, y_full, '.', color='0.6', zorder=1)
    plt.scatter(xc[idx], yc[idx], c=hc_cor[idx], s=5, vmin=-1, vmax=1, zorder=2)
    plt.plot(np.nanmedian(xc), np.nanmedian(yc), 'o', color='red', zorder=3)
    plt.title('Tracks')

    plt.figure(figsize=(3,9))

    plt.subplot(311)
    plt.plot(bc_proc, hc_proc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('Bs (dB)')
    plt.ylabel('h (m)')

    plt.subplot(312)
    plt.plot(wc_proc, hc_proc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('LeW (m)')
    plt.ylabel('h (m)')

    plt.subplot(313)
    plt.plot(sc_proc, hc_proc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('TeS (?)')
    plt.ylabel('h (m)')
    
    print 'Summary:'
    print 'std_unc:     ', std1
    print 'std_cor:     ', std2
    print 'change:      ', round(p_std*100, 1), '%'
    print ''
    print 'trend_unc:   ', linefit(tc, hc, return_coef=True)[0]
    print 'trend_cor:   ', linefit(tc, hc_cor, return_coef=True)[0]
    print 'change:      ', round(p_trend*100, 1), '%'
    print ''
    print 'r_hxbs_unc:  ', r_bc
    print 'r_hxlew_unc: ', r_wc
    print 'r_hxtes_unc: ', r_sc
    print ''
    print 'r_hxbs_cor:  ', r_bc2
    print 'r_hxlew_cor: ', r_wc2
    print 'r_hxtes_cor: ', r_sc2
    print ''
    print 's_hxbs_unc:  ', s_bc
    print 's_hxlew_unc: ', s_wc
    print 's_hxtes_unc: ', s_sc
    print ''
    print 's_hxbs/std:  ', s_bc2
    print 's_hxlew/std: ', s_wc2
    print 's_hxtes/std: ', s_sc2
    plt.show()


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, proc=None):

    if TEST_MODE:
        print '*********************************************************'
        print '* RUNNING IN TEST MODE (PLOTTING ONLY, NOT SAVING DATA) *'
        print '*********************************************************'

    print 'processing file:', ifile, '...'

    xvar, yvar, zvar, tvar = vnames
    bpar, wpar, spar = wnames

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:

        t = fi[tvar][:]
        h = fi[zvar][:]
        lon = fi[xvar][:]
        lat = fi[yvar][:]
        bs = fi[bpar][:]
        lew = fi[wpar][:]
        tes = fi[spar][:]

    # Filter time
    #FIXME: Always check this!!!
    if 1:
        h[t<1992] = np.nan
        bs[t<1992] = np.nan
        lew[t<1992] = np.nan
        tes[t<1992] = np.nan

    #TODO: Replace by get_grid?
    # Get bbox of all cells (the grid)
    bboxs = get_bboxs(lon, lat, dxy, proj=proj)

    """ Create output containers """

    N = len(bboxs)

    # Values for each point
    pstd = np.zeros_like(h)          # perc std change after cor 
    ptrend = np.zeros_like(h)        # perc trend change after cor 
    hbs = np.full_like(h, np.nan)    # scatt cor from multivar fit
    rbs = np.full_like(h, np.nan)    # corr coef h x Bs
    rlew = np.full_like(h, np.nan)   # corr coef h x LeW
    rtes = np.full_like(h, np.nan)   # corr coef h x TeS
    sbs = np.full_like(h, np.nan)    # sensit h x Bs
    slew = np.full_like(h, np.nan)   # sensit h x LeW
    stes = np.full_like(h, np.nan)   # sensit h x TeS
    sbs2 = np.full_like(h, np.nan)   # sensit h x Bs
    slew2 = np.full_like(h, np.nan)  # sensit h x LeW
    stes2 = np.full_like(h, np.nan)  # sensit h x TeS
    bbs = np.full_like(h, np.nan)    # multivar fit coef a.Bs
    blew = np.full_like(h, np.nan)   # multivar fit coef b.LeW
    btes = np.full_like(h, np.nan)   # multivar fit coef c.TeS

    # Values for each cell
    pstdc = np.full(N, 0.0)
    ptrendc = np.full(N, 0.0)
    rbsc = np.full(N, np.nan)
    rlewc = np.full(N, np.nan) 
    rtesc = np.full(N, np.nan) 
    sbsc = np.full(N, np.nan) 
    slewc = np.full(N, np.nan) 
    stesc = np.full(N, np.nan) 
    sbsc2 = np.full(N, np.nan) 
    slewc2 = np.full(N, np.nan) 
    stesc2 = np.full(N, np.nan) 
    bbsc = np.full(N, np.nan) 
    blewc = np.full(N, np.nan) 
    btesc = np.full(N, np.nan) 
    lonc = np.full(N, np.nan) 
    latc = np.full(N, np.nan) 
    hbsmnc = np.full(N, np.nan) 
    hbsmdc = np.full(N, np.nan) 
    hbssdc = np.full(N, np.nan) 

    # Select cells at random (for testing)
    if TEST_MODE:
        if USE_SEED:
            np.random.seed(999)  # not so random!
        ii = range(len(bboxs))
        bboxs = np.array(bboxs)[np.random.choice(ii, N_CELLS)]

    # Build KD-Tree with polar stereo coords (if a radius is provided)
    if radius > 0:
        x, y = transform_coord(4326, proj, lon, lat)
        Tree = cKDTree(zip(x, y))

    # Loop through cells
    for k,bbox in enumerate(bboxs):

        print 'Calculating correction for cell', k, 'of', len(bboxs), '...'

        # Get indexes of data within search radius or cell bbox
        if radius > 0:
            i_cell = get_radius_idx(
                    x, y, bbox[0], bbox[2], radius, Tree, n_reloc=n_reloc,
                    min_months=18, max_reloc=3, time=t, height=h)
        else:
            i_cell = get_cell_idx(lon, lat, bbox, proj=proj)

        # Get all data within the grid cell/search radius
        tc = t[i_cell]
        hc = h[i_cell]
        xc = lon[i_cell]
        yc = lat[i_cell]
        bc = bs[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        # Keep original (unfiltered) data
        tc_orig, hc_orig = tc.copy(), hc.copy()

        #bc0, wc0, sc0 = bc.copy(), wc.copy(), sc.copy()

        # Filter invalid points
        tc, hc, bc, wc, sc = filter_data(tc, hc, bc, wc, sc)

        #bc1, wc1, sc1 = bc.copy(), wc.copy(), sc.copy()

        # Test minimum number of obs
        nobs = min([len(v[~np.isnan(v)]) for v in [hc, bc, wc, sc]])

        # Test for enough points
        if (nobs < 25):
            continue

        #bc, wc, sc = interp_params(tc, bc, wc, sc)  # based on longset w/f series
        bc, wc, sc = interp_params(tc, hc, bc, wc, sc)  # based on h series

        '''
        bc2, wc2, sc2 = bc.copy(), wc.copy(), sc.copy()

        bc2[bc2==bc1] = np.nan
        wc2[wc2==wc1] = np.nan
        sc2[sc2==sc1] = np.nan

        ax1 = plt.subplot(311)
        plt.plot(tc, bc0, '.')
        plt.plot(tc, bc1, '.')
        plt.plot(tc, bc2, 'x')
        ax2 = plt.subplot(312)
        plt.plot(tc, wc0, '.')
        plt.plot(tc, wc1, '.')
        plt.plot(tc, wc2, 'x')
        ax3 = plt.subplot(313)
        plt.plot(tc, sc0, '.')
        plt.plot(tc, sc1, '.')
        plt.plot(tc, sc2, 'x')

        plt.show()
        continue
        '''

        # Ensure zero mean on all variables
        hc, bc, wc, sc = center(hc, bc, wc, sc)

        # Calculate correction for grid cell/search radius
        hc_bs, b_bc, b_wc, b_sc = get_scatt_cor(tc, hc, bc, wc, sc, proc=proc)

        if (hc_bs == 0).all():
            r_bc, r_wc, r_sc = 0., 0., 0.
            s_bc, s_wc, s_sc = 0., 0., 0.
            s_bc2, s_wc2, s_sc2 = 0., 0., 0.

        else:
            # Calculate correlation between h and waveform params
            r_bc, r_wc, r_sc = corr_coef([hc, bc, wc, sc], proc=proc, time=tc)

            # Calculate sensitivity between h and waveform params
            s_bc, s_wc, s_sc = corr_grad([hc, bc, wc, sc], proc=proc, time=tc)

            # Calculate normalized sensitivity values
            s_bc2, s_wc2, s_sc2 = corr_grad([hc, bc, wc, sc], proc=proc, time=tc, normalize=True)

        # Test if at least one correlation is significant
        cond = (np.abs(r_bc) > R_MIN or np.abs(r_wc) > R_MIN or np.abs(r_sc) > R_MIN)

        # Apply correction only if improves residuals, or corr is significant
        if not np.all(hc_bs == 0) and cond:

            #NOTE: filt = True or False ???
            hc_cor, hc_bs = apply_scatt_cor(tc, hc, hc_bs, filt=False, test_std=True)

        else:

            hc_cor = hc.copy()
            hc_bs[:] = 0.

        # Set filtered out (invalid) values
        hc_bs[np.isnan(hc)] = np.nan

        # Plot individual grid cells for testing
        if TEST_MODE:

            print 'total std (including trend):\n'
            print 'h_original: ', np.nanstd(hc_orig)
            print 'h_corrected:', np.nanstd(hc)
            print ''
 
            plt.figure(figsize=(6,2))

            plt.plot(tc_orig, hc_orig, '.', color='0.3')

            plot(xc, yc, tc, hc, bc, wc, sc, hc_cor, hc_bs, lon, lat, proc=proc)

        """ Store results (while checking previously stored estimates) """

        # Get percentange of variance change in cell
        p_new = std_change(tc, hc, hc_cor, detrend_=True, lowess=LOWESS)

        # Get percentange of trend change in cell
        a_new = trend_change(tc, hc, hc_cor)

        # Check where/if previously stored values need update (p_old < p_new)
        i_update, = np.where(pstd[i_cell] <= p_new)  # use '<=' !!!

        # Keep only improved values
        i_cell_new = [i_cell[i] for i in i_update]  # a list!
        hc_bs_new = hc_bs[i_update]

        # Store correction for cell (only improved values)
        hbs[i_cell_new] = hc_bs_new    # set of values
        pstd[i_cell_new] = p_new       # one value (same for all)
        ptrend[i_cell_new] = a_new 
        rbs[i_cell_new] = r_bc         # corr coef
        rlew[i_cell_new] = r_wc        
        rtes[i_cell_new] = r_sc
        sbs[i_cell_new] = s_bc         # sensitivity
        slew[i_cell_new] = s_wc
        stes[i_cell_new] = s_sc
        sbs2[i_cell_new] = s_bc2       # sensitivity normalized
        slew2[i_cell_new] = s_wc2
        stes2[i_cell_new] = s_sc2
        bbs[i_cell_new] = b_bc         # multivar fit coef
        blew[i_cell_new] = b_wc
        btes[i_cell_new] = b_sc
        
        # Compute centroid of cell 
        lon_c = np.nanmedian(xc)
        lat_c = np.nanmedian(yc)

        # Statistics of scatt cor for each cell
        h_bs_mnc = np.nanmean(hc_bs_new)
        h_bs_mdc = np.nanmedian(hc_bs_new)
        h_bs_sdc = np.nanstd(hc_bs_new, ddof=1)

        # Store one s and r value per cell
        lonc[k] = lon_c
        latc[k] = lat_c
        pstdc[k] = p_new
        ptrendc[k] = a_new
        hbsmnc[k] = h_bs_mnc
        hbsmdc[k] = h_bs_mdc
        hbssdc[k] = h_bs_sdc
        rbsc[k] = r_bc
        rlewc[k] = r_wc
        rtesc[k] = r_sc
        sbsc[k] = s_bc
        slewc[k] = s_wc
        stesc[k] = s_sc
        sbsc2[k] = s_bc2
        slewc2[k] = s_wc2
        stesc2[k] = s_sc2
        bbsc[k] = b_bc
        blewc[k] = b_wc
        btesc[k] = b_sc

        print 'Correlation     (Bs, Lew, Tes): ', \
                np.around(r_bc,2),np.around(r_wc,2),np.around(r_sc,2)
        print 'Sensitivity     (Bs, Lew, Tes): ', \
                np.around(s_bc,2),np.around(s_wc,2),np.around(s_sc,2)
        print 'Sensitivity/std (Bs, Lew, Tes): ', \
                np.around(s_bc2,2),np.around(s_wc2,2),np.around(s_sc2,2)
        print 'Trend change (%): ', np.around(a_new,3)

    """ Correct h (full dataset) with best values """

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
                fi['p_std'] = pstd
                fi['p_trend'] = ptrend
                fi['r_bs'] = rbs
                fi['r_lew'] = rlew
                fi['r_tes'] = rtes
                fi['s_bs'] = sbs
                fi['s_lew'] = slew
                fi['s_tes'] = stes
                fi['s_bs2'] = sbs2
                fi['s_lew2'] = slew2
                fi['s_tes2'] = stes2
                fi['b_bs'] = bbs
                fi['b_lew'] = blew
                fi['b_tes'] = btes
            
            #FIXME: Check if this is a good idea. Content of input file is being deleted!!! 
            # Update variabels instead
            except:

                # Save params for each point
                fi['h_bs'][:] = hbs
                fi['p_std'][:] = pstd
                fi['p_trend'][:] = ptrend
                fi['r_bs'][:] = rbs
                fi['r_lew'][:] = rlew
                fi['r_tes'][:] = rtes
                fi['s_bs'][:] = sbs
                fi['s_lew'][:] = slew
                fi['s_tes'][:] = stes
                fi['s_bs2'][:] = sbs2
                fi['s_lew2'][:] = slew2
                fi['s_tes2'][:] = stes2
                fi['b_bs'][:] = bbs
                fi['b_lew'][:] = blew
                fi['b_tes'][:] = btes

        # Rename file
        os.rename(ifile, ifile.replace('.h5', '_SCAT.h5'))
        
        # Save bs params as external file 
        with h5py.File(ifile.replace('.h5', '_PARAMS.h5'), 'w') as fo:
            
            # Try to svave variables
            try:
                
                # Save varibales
                fo['lon'] = lonc
                fo['lat'] = latc
                fo['p_std'] = pstdc
                fo['p_trend'] = ptrendc
                fo['h_bs_mean'] = hbsmnc
                fo['h_bs_median'] = hbsmdc
                fo['h_bs_std'] = hbssdc
                fo['r_bs'] = rbsc
                fo['r_lew'] = rlewc
                fo['r_tes'] = rtesc
                fo['s_bs'] = sbsc
                fo['s_lew'] = slewc
                fo['s_tes'] = stesc
                fo['s_bs2'] = sbsc2
                fo['s_lew2'] = slewc2
                fo['s_tes2'] = stesc2
                fo['b_bs'] = bbsc
                fo['b_lew'] = blewc
                fo['b_tes'] = btesc
            
            # Catch any exceptions 
            except:
                
                # Exit program
                print 'COUND NOT SAVE PARAMETERS FOR EACH CELL'
                return

    """ Plot maps """

    if TEST_MODE:
        # Convert into sterographic coordinates
        xi, yi = transform_coord('4326', proj, lonc, latc)
        plt.scatter(xi, yi, c=rbsc, s=25, cmap=plt.cm.bwr)
        plt.colorbar()
        plt.show()


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
    njobs = args.njobs[0]          # parallel writing

    print 'parameters:'
    for arg in vars(args).iteritems():
        print arg

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc) \
                for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc) \
                        for ifile in ifiles)

    print 'done!'

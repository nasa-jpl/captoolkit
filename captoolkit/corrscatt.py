#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Corrects radar altimetry height to correlation with waveform parameters.

Example:
    scattcor.py -v lon lat h_res t_year -w bs lew tes -d 1 -r 4 -q 2 
        -p dif -f /path/to/*files.h5

Notes:
    The (back)scattering correction is applied as:

        hc_cor = h - h_bs

"""
import os
import sys
import h5py
import pyproj
import timeit
import warnings
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter

from sklearn.metrics import mean_squared_error
from math import sqrt

#--- Edit ------------------------------------------------------------

##NOTE: This uses random cells, plot results, and do not save data
TEST_MODE = False

# Use random locations
USE_SEED = True
N_CELLS = 20
SEED = 222

# If True, uses given locations instead of random nodes (specified below)
USE_NODES = False

# Specific locations for testing: Ross, Getz, PIG
NODES = [(-158.71, -78.7584),
        #(-124.427, -74.4377),  # Getz
        #(-100.97, -75.1478),   # PIG
         (-158.40, -78.80), 
         (-178.40, -78.80), 
         (-188.00, -77.95),
         (-160.00, -80.40), 
         (-178.40, -80.60), 
         (-190.40, -80.60),]

# Suffix for output file
SUFFIX1 = '_SCAT'
SUFFIX2 = '_SCATGRD'

# Nave of variable to save bs-correction
H_BS = 'h_bs'

# Apply 3-month running median to processed time series
BIN_SERIES = True

# Minimum correlation for each waveform param (NOT USED)
R_MIN = 0.1

# Minimum points per cell to compute solution
MIN_PTS = 50

# Minimum number of months to compute solution 
MIN_MONTHS = 24

# Default time range
TMIN, TMAX = -9999, 9999

# Savitzky-Golay params for numerical diff
WINDOW = 15
ORDER = 1
DERIV = 1

#----------------------------------------------------------------

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
            '-t', metavar=('tmin','tmax'), dest='tlim', type=float, nargs=2,
            help="time interval to compute corrections (dec years)",
            default=[TMIN,TMAX],)
    parser.add_argument(
            '-b', metavar=('e','w','s','n'), dest='bbox', type=float, nargs=4,
            help="full bbox in case of processing tiles for consistency",
            default=[None],)
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
    if xmin is None: xmin = np.nanmin(x)
    if xmax is None: xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax+dx, dx)
    bins = [(ti, ti+window) for ti in steps]

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


def detrend_binned(x, y, order=1, window=3/12.):
    """ Bin data (Med), compute trend (OLS) on binned, detrend original data. """
    x_b, y_b = binning(x, y, median=True, window=window, interp=False)[:2]
    i_valid = ~np.isnan(y_b)
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
        idx, = np.where(np.abs(y_res) > mad_std(y_res)*n_sigma)
        if len(idx) == 0: break  # if no data to filter, stop iterating
        y_res[idx] = np.nan
        if np.sum(~np.isnan(y_res)) < 10: break  ##NOTE: Arbitrary min obs
    y_filt[np.isnan(y_res)] = np.nan    
    return y_filt


def detrend_binned2(x, y, window=3/12.):
    """ Bin data (Med), detrend original data with binned data. """
    x_b, y_b = binning(x, y, median=True, window=window, interp=True)[:2]
    return y-y_b, y_b


def sigma_filter2(x, y, window=3/12., n_iter=3, n_sigma=3):
    """ Bin data, remove binned, remove 3 sigma from residual, repeat. """
    y_filt, y_res = y.copy(), y.copy()
    for _ in range(n_iter):
        y_res, y_trend = detrend_binned2(x, y_res, window=window)
        idx, = np.where(np.abs(y_res) > mad_std(y_res)*n_sigma)
        if len(idx) == 0: break  # if no data to filter, stop iterating
        y_res[idx] = np.nan
        if sum(~np.isnan(y_res)) < 10: break  ##NOTE: Arbitrary min obs
    y_filt[np.isnan(y_res)] = np.nan    
    return y_filt


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
    if len(idx) < 3: return np.nan, np.nan, np.nan
    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]

    try:
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
    except:
        return np.nan, np.nan, np.nan

    return s_bs, s_lew, s_tes
        

def linefit(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate: Huber, 1981).

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    assert sum(~np.isnan(y)) > 1

    X = sm.add_constant(x, prepend=False)
    y_fit = sm.RLM(y, X, M=sm.robust.norms.HuberT(), missing="drop") \
            .fit(maxiter=1, tol=0.001)
    
    x_fit = x[~np.isnan(y)]

    if return_coef:
        if len(y_fit.params) < 2: 
            return y_fit.params[0], 0.
        else: 
            return y_fit.params[:]
    else:
        return x_fit, y_fit.fittedvalues


def is_empty(ifile):
    """If file is empty/corruted, return True."""
    try:
        with h5py.File(ifile, 'r') as f: return not bool(list(f.keys()))
    except:
        return True


""" Helper functions """


def filter_data(t, h, bs, lew, tes, n_sigma=3, window=3/12.):
    """
    Use various filters to remove outliers.

    Replaces outliers with NaNs.
    """
    # Iterative mode filter (for repeated values)
    h = mode_filter(h, min_count=10, maxiter=3)
    bs = mode_filter(bs, min_count=10, maxiter=3)
    lew = mode_filter(lew, min_count=10, maxiter=3)
    tes = mode_filter(tes, min_count=10, maxiter=3)
    if 1:
        # Iterative n-sigma filter (w.r.t. the OLS trend)
        h = sigma_filter(t, h, order=2, n_sigma=n_sigma, n_iter=3)
        bs = sigma_filter(t, bs, order=2, n_sigma=n_sigma, n_iter=3)
        lew = sigma_filter(t, lew, order=2, n_sigma=n_sigma, n_iter=3)
        tes = sigma_filter(t, tes, order=2, n_sigma=n_sigma, n_iter=3)
    else:
        # Iterative n-sigma filter (w.r.t. the Med-binned)
        h = sigma_filter2(t, h, window=window, n_sigma=n_sigma, n_iter=3)
        bs = sigma_filter2(t, bs, window=window, n_sigma=n_sigma, n_iter=3)
        lew = sigma_filter2(t, lew, window=window, n_sigma=n_sigma, n_iter=3)
        tes = sigma_filter2(t, tes, window=window, n_sigma=n_sigma, n_iter=3)
    return t, h, bs, lew, tes


def interp_params(t, h, bs, lew, tes):
    """
    Interpolate waveform params based on height-series valid entries.

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


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Construct output grid-coordinates. """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1  # ny
    Ne = int((np.abs(xmax - xmin)) / dx) + 1  # nx
    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)
    return np.meshgrid(xi, yi)


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


def multi_fit_coef(t_, h_, bs_, lew_, tes_):
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

    """ 
    # Ensure zero mean of processed series
    h_, bs_, lew_, tes_ = center(h_, bs_, lew_, tes_)

    # Construct design matrix: First-order model
    A_ = np.vstack((bs_, lew_, tes_)).T

    # Check for division by zero
    try:
        
        # Fit robust linear model on differenced series (w/o NaNs)
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

    # Set all params to zero if exception detected
    except:

        print('MULTIVARIATE FIT FAILED, setting params -> 0')
        print('VALID DATA POINTS in h:', sum(~np.isnan(h_)))
    
        a, b, c, r2, pval, pvals = 0, 0, 0, 0, 1e3, [1e3, 1e3, 1e3]
    
    return [a, b, c, r2, pval, pvals]


def rmse(t, x1, x2, order=1):
    """ RMSE between (detrended) x1 and x2. """
    x1_res = detrend_binned(t, x1, order=order)[0]  # use OLS poly fit
    x2_res = detrend_binned(t, x2, order=order)[0]
    ii = ~np.isnan(x1_res) & ~np.isnan(x2_res)
    x1_res, x2_res = x1_res[ii], x2_res[ii]
    return sqrt(mean_squared_error(x1_res, x2_res))


def std_change(t, x1, x2, order=1):
    """ Variance change from (detrended) x1 to x2 (magnitude and percentage). """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    if sum(idx) < 3: return np.nan, np.nan
    x1_res = detrend_binned(t, x1, order=order)[0]  # use OLS poly fit
    x2_res = detrend_binned(t, x2, order=order)[0]
    s1 = mad_std(x1_res)
    s2 = mad_std(x2_res)
    delta_s = s2 - s1
    return delta_s, delta_s/s1 


def trend_change(t, x1, x2):
    """ Linear-trend change from x1 to x2 (magnitude and percentage). """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    if sum(idx) < 3: return np.nan, np.nan
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
    x1_ -= x1_.mean()
    x2_ -= x2_.mean()
    a1 = np.polyfit(t_, x1_, 1)[0]  # use OLS poly fit
    a2 = np.polyfit(t_, x2_, 1)[0]
    delta_a = a2 - a1
    return delta_a, delta_a/np.abs(a1)


def sgolay1d(h, window=3, order=1, deriv=0, dt=1.0, mode='nearest', time=None):
    """Savitztky-Golay filter with support for NaNs

    If time is given, interpolate NaNs otherwise pad w/zeros.

    dt is spacing between samples.
    """
    h2 = h.copy()
    ii, = np.where(np.isnan(h2))
    jj, = np.where(np.isfinite(h2))
    if len(ii) > 0 and time is not None:
        h2[ii] = np.interp(time[ii], time[jj], h2[jj])
    elif len(ii) > 0:
        h2[ii] = 0
    else:
        pass
    h2 = savgol_filter(h2, window, order, deriv, delta=dt, mode=mode)
    return h2 


def overlap(x1, x2, y1, y2):
    """ Return True if x1-x2/y1-y2 ranges overlap. """
    return (x2 >= y1) & (y2 >= x1)


def intersect(x1, x2, y1, y2, a1, a2, b1, b2):
    """ Return True if two (x1,x2,y1,y2) rectangles intersect. """
    return (overlap(x1, x2, a1, a2) & overlap(y1, y2, b1, b2))


def plot(x, y, xc, yc, tc, hc, bc, wc, sc,
         hc_, bc_, wc_, sc_, hc_cor, h_bs,
         r_bc, r_wc, r_sc, s_bc, s_wc, s_sc,
         d_std, p_std, d_trend, p_trend, r2, pval, pvals):

    tc_ = tc.copy()

    # Bin variables
    hc_b = binning(tc, hc_cor, median=True, window=3/12.)[1]    ##NOTE: If binning before, this is binning over a binning
    bc_b = binning(tc, bc, median=True, window=3/12.)[1]
    wc_b = binning(tc, wc, median=True, window=3/12.)[1]
    tc_b, sc_b = binning(tc, sc, median=True, window=3/12.)[:2]

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
    hc_b_ = binning(tc_, hc_, median=False, window=3/12.)[1]         ##NOTE: If original TS were binned,
    bc_b_ = binning(tc_, bc_, median=False, window=3/12.)[1]         ## this is binning over a binning
    wc_b_ = binning(tc_, wc_, median=False, window=3/12.)[1]
    tc_b_, sc_b_ = binning(tc_, sc_, median=True, window=3/12.)[:2]

    # Fit seasonality
    plot_season = True
    if plot_season:

        from scipy import optimize

        # Trend + Seasonal model
        def func(t, c, m, n, a, p):
            """ Seasonality with amplitude and phase. """
            #return a * np.sin(2*np.pi * t + p)
            return c + m * t + n * t**2 + a * np.sin(2*np.pi * t + p)

        # Fit seasonality
        ii = ~np.isnan(tc_) & ~np.isnan(hc_)
        t_season, h_season = tc_[ii], hc_[ii]
        params, params_covariance = optimize.curve_fit(func, t_season, h_season, p0=[0., 0., 0., 0.1, 0.1])
        a0, a1, a2, amp, pha = params
        t_season = np.linspace(t_season.min(), t_season.max(), 100)
        h_season = func(t_season, a0, a1, a2, amp, pha)


    # mask NaNs for plotting
    mask = np.isfinite(hc_b_)

    plt.figure(figsize=(6,8))

    plt.subplot(4,1,1)
    plt.plot(tc_, hc_, '.')
    plt.plot(tc_b_[mask], hc_b_[mask], '-')
    if plot_season: plt.plot(t_season, h_season, '-r')
    plt.ylabel('Height (m)')
    plt.title('Processed time series')
    if plot_season: plt.title('Amplitude = %.2f, Phase = %.2f' % (amp, pha))

    plt.subplot(4,1,2)
    plt.plot(tc_, bc_, '.')
    plt.plot(tc_b_[mask], bc_b_[mask], '-')
    plt.ylabel('Bs (s.d.)')
    
    plt.subplot(4,1,3)
    plt.plot(tc_, wc_, '.')
    plt.plot(tc_b_[mask], wc_b_[mask], '-')
    plt.ylabel('LeW (s.d.)')
    
    plt.subplot(4,1,4)
    plt.plot(tc_, sc_, '.')
    plt.plot(tc_b_[mask], sc_b_[mask], '-')
    plt.ylabel('TeS (s.d.)')

    plt.figure()
    plt.plot(x, y, '.', color='0.6', zorder=1)
    plt.scatter(xc, yc, c=hc_cor, s=5, vmin=-1, vmax=1, zorder=2)
    plt.plot(np.nanmedian(xc), np.nanmedian(yc), 'o', color='red', zorder=3)
    plt.title('Tracks')

    # Plot Spectrum
    if 1:

        tc, hc, bc, wc, sc = tc_, hc_, bc_, wc_, sc_  # processed TS

        from astropy.stats import LombScargle

        periods = np.arange(3/12., 1.5, 0.01)
        freq = 1/periods

        hc[np.isnan(hc)] = 0.
        bc[np.isnan(bc)] = 0.
        wc[np.isnan(wc)] = 0.
        sc[np.isnan(sc)] = 0.

        ls = LombScargle(tc, hc, nterms=1)
        power_h = ls.power(freq)
        ls = LombScargle(tc, bc, nterms=1)
        power_b = ls.power(freq)
        ls = LombScargle(tc, wc, nterms=1)
        power_w = ls.power(freq)
        ls = LombScargle(tc, sc, nterms=1)
        power_s = ls.power(freq)

        plt.figure(figsize=(6,8))

        plt.subplot(4,1,1)
        plt.plot(freq, power_h, linewidth=2) 
        plt.xlabel('Frequency (cycles/year)')
        plt.ylabel('Power (1/RMSE)')        
        plt.title('Spectrum of processed time series')

        plt.subplot(4,1,2)
        plt.plot(freq, power_b, linewidth=2) 
        plt.xlabel('Frequency (cycles/year)')
        plt.ylabel('Power (1/RMSE)')        

        plt.subplot(4,1,3)
        plt.plot(freq, power_w, linewidth=2) 
        plt.xlabel('Frequency (cycles/year)')
        plt.ylabel('Power (1/RMSE)')        

        plt.subplot(4,1,4)
        plt.plot(freq, power_s, linewidth=2) 
        plt.xlabel('Frequency (cycles/year)')
        plt.ylabel('Power (1/RMSE)')        

    # Plot Crosscorrelation
    if 1:

        tc, hc, bc, wc, sc = tc_, hc_, bc_, wc_, sc_  # processed TS

        plt.figure(figsize=(6,8))

        plt.subplot(311)
        plt.xcorr(hc, bc)
        plt.ylabel('h x bs')
        plt.title('Crosscorrelation')

        plt.subplot(312)
        plt.xcorr(hc, wc)
        plt.ylabel('h x LeW')

        plt.subplot(313)
        plt.xcorr(hc, sc)
        plt.ylabel('h x TeS')

    tc, hc, bc, wc, sc = tc_, hc_, bc_, wc_, sc_  # processed TS

    plt.figure(figsize=(3,9))

    plt.subplot(311)
    plt.plot(bc, hc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('Bs (s.d.)')
    plt.ylabel('h (m)')
    plt.title('Correlation of processed time series')

    plt.subplot(312)
    plt.plot(wc, hc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('LeW (s.d.)')
    plt.ylabel('h (m)')

    plt.subplot(313)
    plt.plot(sc, hc, '.')
    #plt.title('Correlation Bs x h (%s)' % str(proc))
    plt.xlabel('TeS (s.d.)')
    plt.ylabel('h (m)')


    print('Summary:')
    print('--------')
    print('cor applied: ', (h_bs[~np.isnan(h_bs)] != 0).any())
    print('std change:   %.3f m (%.1f %%)' % (round(d_std, 3), round(p_std*100, 1)))
    print('trend change: %.3f m/yr (%.1f %%)' % (round(d_trend, 3), round(p_trend*100, 1)))
    print('')
    print('r-squared: ', round(r2, 3))
    print('p-value:   ', round(pval, 3))
    print('p-values:  ', [round(p, 3) for p in pvals])
    print('')
    print('r_bs:      ', round(r_bc, 3))
    print('r_lew:     ', round(r_wc, 3))
    print('r_tes:     ', round(r_sc, 3))
    print('')                            
    print('s_bs:      ', round(s_bc, 3))
    print('s_lew:     ', round(s_wc, 3))
    print('s_tes:     ', round(s_sc, 3))

    plt.show()


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, proc=None, apply_=False):

    if is_empty(ifile):
        print('empty file... skipping!!!')
        return
    
    if TEST_MODE:
        print('*********************************************************')
        print('* RUNNING IN TEST MODE (PLOTTING ONLY, NOT SAVING DATA) *')
        print('*********************************************************')

    print('processing file:', ifile, '...')
    
    # Test if parameter file exists
    if '_scatgrd' in ifile.lower():
        return

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

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Get bbox from data
    xmin_d, xmax_d, ymin_d, ymax_d = x.min(), x.max(), y.min(), y.max()

    # If no bbox given, limits are defined by data
    if bbox[0] is None:
        xmin, xmax, ymin, ymax = xmin_d, xmax_d, ymin_d, ymax_d
    else:
        xmin, xmax, ymin, ymax = bbox

    # Grid solution - defined by nodes
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dxy, dxy)

    # Flatten prediction grid
    x_nodes, y_nodes = Xi.ravel(), Yi.ravel()

    """ Create output containers """

    N_data = len(x)
    N_nodes = len(x_nodes)

    # Values for each data point
    r2fit = np.full(N_data, 0.0)      # r2 of the multivar fit 
    pval = np.full(N_data, np.nan)    # r2 of the multivar fit 
    dstd = np.full(N_data, np.nan)    # magnitude std change after cor 
    dtrend = np.full(N_data, np.nan)  # magnitude trend change after cor 
    pstd = np.full(N_data, np.inf)    # perc std change after cor   ##NOTE: Init w/inf
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
    pstdc = np.full(N_nodes, np.inf)  ##NOTE: Init w/inf
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

    # Select cells at random (for testing)
    if TEST_MODE:
        if USE_NODES:
            # Convert into sterographic coordinates
            x_nodes = [transform_coord('4326', '3031', xp, yp)[0] for xp, yp in NODES]
            y_nodes = [transform_coord('4326', '3031', xp, yp)[1] for xp, yp in NODES]
        else:
            if USE_SEED: np.random.seed(SEED)  # not so random!
            # Select a few random nodes
            ii = np.random.randint(0, N_nodes, N_CELLS)
            x_nodes, y_nodes = x_nodes[ii], y_nodes[ii] 
        N_nodes = len(x_nodes)

    # Build KD-Tree with polar stereo coords
    x, y = transform_coord(4326, proj, lon, lat)
    Tree = cKDTree(list(zip(x, y)))

    # Loop through nodes
    for k in range(N_nodes):

        if (k%500) == 0:
            print('Calculating correction for node', k, 'of', N_nodes, '...')

        x0, y0 = x_nodes[k], y_nodes[k]

        # If search radius doesn't contain data, skip
        if not intersect(x0-radius, x0+radius, y0-radius, y0+radius, 
                xmin_d, xmax_d, ymin_d, ymax_d):
            continue

        # Get indices of data within search radius
        i_cell = get_radius_idx(x, y, x0, y0, radius, Tree, n_reloc=n_reloc)

        # If cell empty or not enough data go to next node
        if len(i_cell) < MIN_PTS: continue

        # Get all data within the grid search radius
        tc = t[i_cell]
        hc = h[i_cell]
        xc = x[i_cell]
        yc = y[i_cell]
        bc = bs[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        # Keep original (unfiltered) data
        tc_orig, hc_orig, bc_orig, wc_orig, sc_orig = tc.copy(), hc.copy(), bc.copy(), wc.copy(), sc.copy()

        PLOT_INTERP = False
        if PLOT_INTERP: bc0, wc0, sc0 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        # Filter invalid points
        tc, hc, bc, wc, sc = filter_data(tc, hc, bc, wc, sc, n_sigma=3, window=3/12.)

        if PLOT_INTERP: bc1, wc1, sc1 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        # Test minimum number of obs in all params
        nobs = min([len(v[~np.isnan(v)]) for v in [hc, bc, wc, sc]])

        # Test for enough points
        if (nobs < MIN_PTS): continue

        # Bin at monthly intervals to check temporal sampling
        h_bin = binning(tc, hc, dx=1/12., window=3/12., interp=False)[1]
        if sum(~np.isnan(h_bin)) < MIN_MONTHS: continue

        # Interpolate missing w/f params based on h series
        #bc, wc, sc = interp_params(tc, hc, bc, wc, sc)  #NOTE: Leave this out for now (only use valid entries in ALL series) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Is this a good idea?

        if PLOT_INTERP: bc2, wc2, sc2 = bc.copy(), wc.copy(), sc.copy()  #NOTE: for plotting

        # Plot interpolated time-series points
        if PLOT_INTERP:

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

        if BIN_SERIES:
            hc_bin = binning(tc, hc, median=True, window=3/12., interp=True)[1]
            bc_bin = binning(tc, bc, median=True, window=3/12., interp=True)[1]
            wc_bin = binning(tc, wc, median=True, window=3/12., interp=True)[1]
            sc_bin = binning(tc, sc, median=True, window=3/12., interp=True)[1]
        else:
            hc_bin = hc
            bc_bin = bc
            wc_bin = wc
            sc_bin = sc
        
        # Ensure zero mean on all variables
        hc, bc, wc, sc = center(hc, bc, wc, sc)
        hc_bin, bc_bin, wc_bin, sc_bin = center(hc_bin, bc_bin, wc_bin, sc_bin)

        # Normalize the w/f params to std = 1
        bc, wc, sc = normalize(bc, wc, sc)
        bc_bin, wc_bin, sc_bin = normalize(bc_bin, wc_bin, sc_bin)

        if proc == 'det':
            # Detrend time series
            hc_res = detrend_binned(tc, hc_bin, order=2)[0]
            bc_res = detrend_binned(tc, bc_bin, order=2)[0]
            wc_res = detrend_binned(tc, wc_bin, order=2)[0]
            sc_res = detrend_binned(tc, sc_bin, order=2)[0]
        else:
            # Savitzky-Golay numerical diff
            hc_res = sgolay1d(hc_bin, WINDOW, ORDER, DERIV)                         #FIXME: Think what dt to use here (uneven scattered points)!!!!!!!
            bc_res = sgolay1d(bc_bin, WINDOW, ORDER, DERIV)
            wc_res = sgolay1d(wc_bin, WINDOW, ORDER, DERIV)
            sc_res = sgolay1d(sc_bin, WINDOW, ORDER, DERIV)

        # Get coefs from multivariate fit
        b_bc, b_wc, b_sc, r2, pval, pvals = multi_fit_coef(tc, hc_res, bc_res, wc_res, sc_res)

        if sum([b_bc, b_wc, b_sc]) == 0: continue

        # Get linear combination of original FILTERED series => h_bs = a Bs + b LeW + c TeS
        hc_bs = np.dot(np.vstack((bc, wc, sc)).T, [b_bc, b_wc, b_sc])

        #NOTE 1: The correction is generated using the original w/f series
        #NOTE 2: Use np.dot so NaNs from original (filtered) series are preserved

        if np.isnan(hc_bs).all(): continue
        
        # Apply correction to height
        hc_cor = hc - hc_bs

        # Calculate correlation between h and waveform params
        r_bc, r_wc, r_sc = corr_coef(hc_res, bc_res, wc_res, sc_res)

        # Calculate sensitivity values (corr grad)
        s_bc, s_wc, s_sc = corr_grad(hc_res, bc_res, wc_res, sc_res, normalize=False)

        # Calculate variance change (magnitude and perc)
        d_std, p_std = std_change(tc, hc, hc_cor, order=1)

        # Calculate trend change (magnitude and perc)
        d_trend, p_trend = trend_change(tc, hc, hc_cor)

        if np.isnan([d_std, p_std, d_trend, p_std]).any(): continue

        # Calculate RMSE between detrended series
        #d_rmse = rmse(tc, hc, hc_cor, order=1)
        
        # Test if at least one correlation is significant
        #r_cond = (np.abs(r_bc) < R_MIN and np.abs(r_wc) < R_MIN and np.abs(r_sc) < R_MIN)

        ##NOTE: We are ignoring the 'pval' (significance of fit).
        ##NOTE: All we care about is the reduction in variance.

        # Do not apply correction if: std increases by more than 5%
        if p_std > 0.05:

            # Cor is set to zero (keeping  NaNs)
            hc_cor = hc.copy()             # back to original
            hc_bs[~np.isnan(hc_bs)] = 0.   # hc_bs keeps NaNs from filtered out values

            '''
            # All params are set to zero/one
            b_bc, b_wc, b_sc = 0., 0., 0.
            r_bc, r_wc, r_sc = 0., 0., 0.
            s_bc, s_wc, s_sc = 0., 0., 0.
            r2, pval, pvals = 0., 1e3, (1e3, 1e3, 1e3)
            d_std, p_std, d_trend, p_trend = 0., 0., 0., 0.
            '''
        
        # Plot individual grid cells for testing
        if TEST_MODE:
            plt.figure(figsize=(6,2))
            plt.plot(tc_orig, hc_orig, '.', color='0.3')
            plot(x, y, xc, yc, tc, hc, bc, wc, sc,
                 hc_res, bc_res, wc_res, sc_res, hc_cor, hc_bs,
                 r_bc, r_wc, r_sc, s_bc, s_wc, s_sc,
                 d_std, p_std, d_trend, p_trend,
                 r2, pval, pvals)

        """ Store results (while checking previously stored estimates) """

        # Check where previously stored values need update 
        #i_update, = np.where(r2fit[i_cell] <= r2)  # r2_prev <= r2_new (new r2 is larger)
        i_update, = np.where(pstd[i_cell] > p_std)  # std_prev > std_new (new std is lower)

        # Remove indices and values that will not be updated 
        i_cell_new = [i_cell[i] for i in i_update]
        hc_bs_new = hc_bs[i_update]

        # Store correction for cell: only the improved values
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

    """ Correct h (full dataset) with best values """

    if apply_: h[~np.isnan(hbs)] -= hbs[~np.isnan(hbs)]

    """ Save data """

    if not TEST_MODE:
        print('saving data ...')

        with h5py.File(ifile, 'a') as fi:

            # Update h in the file and save correction (all cells at once)
            if apply_: fi[zvar][:] = h
            
            # Try to create varibales
            try:
                # Save params for each point
                fi[H_BS] = hbs
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
            
            #FIXME: Check if this is a good idea. Content of input file is being deleted!!! Removing for now!!!
            # Update variabels instead
            except:
                """
                # Save params for each point
                fi[H_BS][:] = hbs
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
                """
                print('SOME PROBLEM WITH THE FILE')
                print('PARAMETERS NOT SAVED!')
                print(ifile)
                return

        # Only rename file if _SCAT has not been added
        if ifile.find(SUFFIX1+'.h5') < 0:
            os.rename(ifile, ifile.replace('.h5', SUFFIX1+'.h5'))
        
        # Save bs params as external file 
        with h5py.File(ifile.replace('.h5', SUFFIX2+'.h5'), 'w') as fo:
            
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
                print('COUND NOT SAVE PARAMETERS FOR EACH CELL (SCATGRD)')
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
    tmin = args.tlim[0]            # min time in decimal years
    tmax = args.tlim[1]            # max time in decimal years
    bbox = args.bbox[:]                

    print('parameters:')
    for arg in vars(args).items():
        print(arg)

    if njobs == 1:
        print('running sequential code ...')
        [main(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc, apply_) \
                for ifile in ifiles]
    else:
        print('running parallel code (%d jobs) ...' % njobs)
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc, apply_) \
                        for ifile in ifiles)

    print('done!')

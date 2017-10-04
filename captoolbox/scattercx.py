#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrects radar altimetry height to correlation with waveform parameters.

Example:
    python scattercx.py -d 5 -v lon lat h_cor t_year \
        -w bs_ice1 lew_ice2 tes_ice2 -f ~/data/envisat/all/bak/*.h5

Notes:
    The bacscatter correction is applied as:

        h_cor = h - h_bs

"""
import sys
import h5py
import pyproj
import warnings
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates


# PIG tile
#'merged_READ_SLOPE_IBE_TIDE_bbox_-1658294_-1458179_-473841_-265001_buff_5_epsg_3031_tile_114.h5'


# Supress anoying warnings
warnings.filterwarnings('ignore')

def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Correct height data for backscatter variations.')

    parser.add_argument(
            '-f', metavar='files', dest='files', type=str, nargs='+',
            help='single HDF5 files to process',
            required=True)

    parser.add_argument(
            '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
            help=('block size of grid cell (km)'),
            default=[], required=True)

    parser.add_argument(
            '-v', metavar=('lon','lat', 'h', 't'), dest='vnames',
            type=str, nargs=4,
            help=('name of x/y/z/t variables in the HDF5'),
            default=[None], required=True)

    parser.add_argument(
            '-w', metavar=('bsc', 'lew', 'tes'), dest='wnames',
            type=str, nargs=3,
            help=('name of sig0/LeW/TeS parameters in HDF5'),
            default=[None], required=True)

    parser.add_argument(
            '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help="for parallel writing of multiple tiles, optional",
            default=[1],)

    return parser.parse_args()


def linear_fit_robust(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate).
    M-stimator = HuberT (Huber, 1981)
    If `return_coef=True` returns the slope (m) and intercept (c).
    """

    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))

    if len(ind) < 2:
        return [np.nan, np.nan]

    (x, y) = x[ind], y[ind]
    X = sm.add_constant(x, prepend=False)
    y_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    y_fit = y_model.fit()

    if return_coef:
        if len(y_fit.params) < 2:
            return (y_fit.params[0], 0.)

        else:
            return y_fit.params[:]
    else:
        return (x, y_fit.fittedvalues)


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def binning(x, y, xmin, xmax, dx=1/12., window=3/12.):
    """ Time-series binning (w/overlapping windows). """

    steps = np.arange(xmin, xmax+dx, dx)
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

        yb[i] = np.nanmean(ybv)
        xb[i] = 0.5 * (t1+t2)
        #xb[i] = np.nanmedian(xbv)
        eb[i] = mad_std(ybv)
        nb[i] = len(ybv)
        sb[i] = np.sum(ybv)

    return xb, yb, eb, nb, sb


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """
    
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def _sigma_filter(x, y, n_sigma=3, frac=1/3.):
    """
    Remove values greater than n * std from the LOWESS trend.
    
    See sigma_filter()
    """
    y2 = y.copy()
    idx, = np.where(~np.isnan(y))

    # Detrend
    trend = sm.nonparametric.lowess(y[idx], x[idx], frac=frac)[:,1]
    y2[idx] = y[idx] - trend

    # Filter
    #i_outliers, = np.where(np.abs(y2) > n_sigma * np.nanstd(y2, ddof=1)) #[1]
    i_outliers, = np.where(np.abs(y2) > n_sigma * mad_std(y2)) #[1]
    y[i_outliers] = np.nan

    # [1] NaNs are not included!

    return len(i_outliers)


def sigma_filter(x, y, n_sigma=3, iterative=True, frac=1/3., maxiter=5):
    """
    Robust iterative sigma filter.

    Remove values greater than n * mad_std from the LOWESS trend.
    """
    n_iter = 0
    n_outliers = 1
    while n_outliers != 0 and not np.isnan(y).all():

        n_outliers = _sigma_filter(x, y, n_sigma=n_sigma, frac=frac)
        n_iter += 1

        if not iterative or maxiter == n_iter:
            break
    return y


def mode_filter(x, min_count=10, maxiter=2):
    """ 
    Iterative mode filter. 

    Remove values repeated 'min_count' times.
    """
    n_iter = 0
    while n_iter < maxiter:
        mod, count = mode(x)
        if count[0] > min_count:
            x[x==mod[0]] = np.nan
            n_iter += 1
        else:
            n_iter = maxiter
    return x


def detrend(x, y, frac=1/3.):
    """ Detrend using LOWESS. """
    trend = sm.nonparametric.lowess(y, x, frac=frac)[:,1]
    if np.isnan(trend).all():
        trend = np.zeros_like(x)
    elif np.isnan(y).any():
        trend = np.interp(x, x[~np.isnan(y)], trend)
    return y-trend, trend


def normalize(y):
    y /= np.nanstd(y, ddof=1)
    return y - np.nanmean(y)


def get_bboxs(lon, lat, proj='3031'):
    """ Define cells (bbox) for estimating corrections. """

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Number of tile edges on each dimension 
    Nns = int(np.abs(y.max() - y.min()) / dxy) + 1
    New = int(np.abs(x.max() - x.min()) / dxy) + 1

    # Coord of tile edges for each dimension
    xg = np.linspace(x.min(), x.max(), New)
    yg = np.linspace(y.min(), y.max(), Nns)

    # Vector of bbox for each cell
    bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg

    #print 'total grid cells:', len(bboxs)
    return bboxs


def get_cell_idx(lon, lat, bbox, proj='3031'):
    """ Get indexes of all data points in cell. """

    # Bounding box of grid cell
    xmin, xmax, ymin, ymax = bbox
    
    # Convert lon/lat to sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Get the sub-tile (grid-cell) indices
    i_cell, = np.where( (x >= xmin) & (x <= xmax) & 
                        (y >= ymin) & (y <= ymax) )

    return i_cell


def get_bs_cor(t, h, bsc, lew, tes):
    """
    Calculate backscatter correction for time series.

    Computes a multivariate fit to waveform params as:

        h(t) = a BsC(t) + b LeW(t) + c TeS(t)

    where a, b and c are the sensitivity of h to each waveform param.

    Args:
        t: time
        h: height change (residuals from mean topo)
        bsc: backscatter coefficient
        lew: leading-edge width
        tes: trailing-edge slope
    """ 

    # Iterative median filter
    h_f = mode_filter(h.copy(), min_count=10, maxiter=3)
    bsc_f = mode_filter(bsc.copy(), min_count=10, maxiter=3)
    lew_f = mode_filter(lew.copy(), min_count=10, maxiter=3)
    tes_f = mode_filter(tes.copy(), min_count=10, maxiter=3)

    # Iterative 3-sigma filter (remove gross outliers only)
    h_f = sigma_filter(t, h_f,   n_sigma=3, frac=1/3., maxiter=3)
    bsc_f = sigma_filter(t, bsc_f, n_sigma=3,frac=1/3., maxiter=3)
    lew_f = sigma_filter(t, lew_f, n_sigma=3,frac=1/3., maxiter=3)
    tes_f = sigma_filter(t, tes_f, n_sigma=3,frac=1/3., maxiter=3)

    # Remove data points if any param is missing
    i_false = np.isnan(h_f) | np.isnan(bsc_f) | np.isnan(lew_f) | np.isnan(tes_f)
    i_true = np.invert(i_false)

    h_f[i_false] = np.nan
    bsc_f[i_false] = np.nan
    lew_f[i_false] = np.nan
    tes_f[i_false] = np.nan

    if len(h_f[i_true]) < 5:
        return [None] * 9

    """ Pre-process data """

    # Bin time series
    if 0:

        # Need enough data for binning (at least 1 year)
        if t.max() - t.min() < 1.0:
            return [None] * 9

        h_f_ = h_f.copy()
        bsc_f_ = bsc_f.copy()
        lew_f_ = lew_f.copy()
        tes_f_ = tes_f.copy()
        t_ = t.copy()

        t_min, t_max = t.min(), t.max()
        _, h_f = binning(t, h_f, t_min, t_max,  1/12., 3/12.)[0:2]
        _, bsc_f = binning(t, bsc_f, t_min, t_max,1/12., 3/12.)[0:2]
        _, lew_f = binning(t, lew_f, t_min, t_max,1/12., 3/12.)[0:2]
        t, tes_f = binning(t, tes_f, t_min, t_max,1/12., 3/12.)[0:2]

    # Detrend time series
    if 0:

        h_f_ = h_f.copy()
        bsc_f_ = bsc_f.copy()
        lew_f_ = lew_f.copy()
        tes_f_ = tes_f.copy()
        t_ = t.copy()

        h_f, h_t = detrend(t, h_f, frac=1/3.)
        bsc_f, bsc_t = detrend(t, bsc_f, frac=1/3.)
        lew_f, lew_t = detrend(t, lew_f, frac=1/3.)
        tes_f, tes_t = detrend(t, tes_f, frac=1/3.)

    # Difference time series
    if 0:

        h_f_ = h_f.copy()
        bsc_f_ = bsc_f.copy()
        lew_f_ = lew_f.copy()
        tes_f_ = tes_f.copy()
        t_ = t.copy()

        h_f = np.gradient(h_f)
        bsc_f = np.gradient(bsc_f)
        lew_f = np.gradient(lew_f)
        tes_f = np.gradient(tes_f)

    """ Construct design matrix """

    # Ensure zero mean
    bsc_f -= np.nanmean(bsc_f)
    lew_f -= np.nanmean(lew_f)
    tes_f -= np.nanmean(tes_f)

    # First-order model
    Ac = np.vstack((bsc_f, lew_f, tes_f)).T

    ###FIXME: Next version use 'if-else' and catch potential errors
    try:

        # Fit robust linear model on clean data
        model = sm.RLM(h_f, Ac, M=sm.robust.norms.HuberT(), missing="drop").fit()

        # Get sensitivity gradients (coefficients) for BsC, LeW, TeS
        s_bsc, s_lew, s_tes = model.params[:3]

        # Multi-variate fit => h_bs = a BsC + b LeW + c TeS
        h_bs = np.dot(Ac, [s_bsc, s_lew, s_tes])
        #NOTE: Using np.dot instead of .fittedvalues to keep NaNs from Ac

        # Update the valid/invalid indices 
        i_false = np.isnan(h_bs)
        i_true = np.invert(i_false)

        # Compute sensitivity to multi-fit
        s_fit = np.polyfit(h_bs[i_true], h_f[i_true], 1)[0]

        # Compute correlation coefficients
        r_bsc = np.corrcoef(h_f[i_true], bsc_f[i_true])[0, 1]
        r_lew = np.corrcoef(h_f[i_true], lew_f[i_true])[0, 1]
        r_tes = np.corrcoef(h_f[i_true], tes_f[i_true])[0, 1]
        r_fit = np.corrcoef(h_f[i_true], h_bs[i_true])[0, 1] 

    except:
        return [None] * 9

    return [h_bs,
            r_bsc, r_lew, r_tes, r_fit,
            s_bsc, s_lew, s_tes, s_fit]


def apply_bs_cor(t, h, h_bs, filt=True):
    """ Apply correction if decreases std of residuals. """

    h_cor = h - h_bs 

    if filt:
        h_cor = sigma_filter(t, h_cor,  n_sigma=3,  frac=1/3., maxiter=1)

    idx, = np.where(~np.isnan(h_cor))

    # Do not apply cor if std of residuals increases 
    if h_cor[idx].std(ddof=1) > h[idx].std(ddof=1):
        h_cor = h
        h_bs[:] = np.nan

    return h_cor, h_bs


def main(ifile, args):

    print 'processing file:', ifile, '...'

    # Pass arguments 
    vnames = args.vnames[:]  # lon/lat/h/time variable names
    wnames = args.wnames[:]  # variables to use for correction
    dxy = args.dxy[0] * 1e3  # tile length (km -> m)
    proj = args.proj[0]      # EPSG proj number
    njobs = args.njobs[0]    # parallel writing

    xvar, yvar, zvar, tvar = vnames
    bpar, wpar, spar = wnames

    print 'parameters:'
    for arg in vars(args).iteritems():
        print arg

    # Load full data to memory (only once)
    fi = h5py.File(ifile, 'a')

    t = fi[tvar][:]
    h = fi[zvar][:]
    lon = fi[xvar][:]
    lat = fi[yvar][:]
    bsc = fi[bpar][:]
    lew = fi[wpar][:]
    tes = fi[spar][:]

    # Get bbox of all cells
    bboxs = get_bboxs(lon, lat, proj=proj)

    # Output containers
    hbs = np.zeros_like(h) 
    rbsc = np.full(len(bboxs), np.nan) 
    rlew = np.full(len(bboxs), np.nan) 
    rtes = np.full(len(bboxs), np.nan) 
    rfit = np.full(len(bboxs), np.nan) 
    sbsc = np.full(len(bboxs), np.nan) 
    slew = np.full(len(bboxs), np.nan) 
    stes = np.full(len(bboxs), np.nan) 
    sfit = np.full(len(bboxs), np.nan) 
    lonc = np.full(len(bboxs), np.nan) 
    latc = np.full(len(bboxs), np.nan) 

    
    #NOTE: Filter time for Envisat only. Remove this in future version?
    if 1:
        ind, = np.where(t > 2010.8)
        lon[ind] = np.nan
        lat[ind] = np.nan
        h[ind] = np.nan
        t[ind] = np.nan
        bsc[ind] = np.nan
        lew[ind] = np.nan
        tes[ind] = np.nan

    # Select cells at random (for testing)
    if 1:
        np.random.seed(999)
        ii = range(len(bboxs))
        bboxs = np.array(bboxs)[np.random.choice(ii, 10)]

    # Loop through cells
    for k,bbox in enumerate(bboxs):

        print 'Calculating correction for cell', k, '...'

        # Get indexes of cell data
        i_cell = get_cell_idx(lon, lat, bbox, proj=proj)

        # Test for enough points
        if len(i_cell) < 4:
            return

        # Get all data within the grid cell
        tc = t[i_cell]
        hc = h[i_cell]
        xc = x[i_cell]
        yc = y[i_cell]
        bc = bsc[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        # Calculate correction for grid cell
        cor = get_bs_cor(tc, hc, bc, wc, sc)

        h_bs = cor[0]
        r_bc = cor[1]
        r_wc = cor[2]
        r_sc = cor[3]
        r_fc = cor[4]
        s_bc = cor[5]
        s_wc = cor[6]
        s_sc = cor[7]
        s_fc = cor[8]

        if h_bs is None:  ###TODO: Check what happens if return here!!!
            return

        # Apply correction to grid-cell data (if improves)
        h_cor, h_bs = apply_bs_cor(tc, hc, h_bs, filt=False)


        ######NOTE: Check if transformation is needed, or median(lon) is the same!
        # Convert back to geographical coordinates
        xc, yc = transform_coord('4326', proj, xc, yc)

        # Compute centroid of cell 
        xi = np.nanmedian(xc)
        yi = np.nanmedian(yc)

        # Convert back to geographical coordinates
        lonc, latc = transform_coord(proj, '4326', xi, yi)
        ########


        # Update h vector with corrected h_cell
        h[i_cell] = h_cor
        hbs[i_cell] = h_bs

        # Store one value per cell
        rbsc[k] = r_bc
        rlew[k] = r_wc
        rtes[k] = r_sc
        rfit[k] = r_fc
        sbsc[k] = s_bc
        slew[k] = s_wc
        stes[k] = s_sc
        sfit[k] = s_fc
        loni[k] = lonc
        lati[k] = latc

    """ Save data """

    # Update h in the file and save cor (all cells at once)
    fi[zvar][:] = h
    fi['h_bs'] = hbs
    
    fi.flush()
    fi.close()
    
    # Save bs params to external file 
    with h5py.File(ifile.replace('.h5', '_params.h5'), 'w') as fo:
    
        fo['lon'] = loni
        fo['lat'] = lati 
        fo['r_bsc'] = rbsc
        fo['r_lew'] = rlew
        fo['r_tes'] = rtes
        fo['r_fit'] = rfit
        fo['s_bsc'] = sbsc 
        fo['s_lew'] = slew 
        fo['s_tes'] = stes 
        fo['s_fit'] = sfit 


if __name__ == '__main__':

    args = get_args()
    ifiles = args.files[:]   # input files

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, args) for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, args) for ifile in ifiles)

    print 'done!'


#!/usr/bin/env python
"""
Program for correcting radar altimetry surface elevations to their
correlation with waveform parameters.

Example
-------

python scattercx.py -d 5 -v lon lat h_cor t_year -w bs_ice1 lew_ice2 tes_ice2 \
    -f ~/data/envisat/all/bak/*.h5

Notes
-----

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


# Supress anoying warnings
warnings.filterwarnings('ignore')

# Defie command-line arguments
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
        '-v', metavar=('lon','lat', 'h', 't'), dest='vnames', type=str, nargs=4,
        help=('name of x/y/z/t variables in the HDF5'),
        default=[None], required=True)

parser.add_argument(
        '-w', metavar=('bsc', 'lew', 'tes'), dest='wnames', type=str, nargs=3,
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


args = parser.parse_args()

# Pass arguments 
ifiles = args.files        # input file
vnames = args.vnames       # lon/lat/h/time variable names
wnames = args.wnames       # variables to use for correction
dxy   = args.dxy[0] * 1e3  # tile length (km -> m)
proj  = args.proj[0]       # EPSG proj number
njobs = args.njobs[0]      # parallel writing

xvar, yvar, zvar, tvar = vnames
bpar, lpar, tpar = wnames

print 'input files:', len(ifiles)
print 'x/y/z/t vars:', vnames
print 'waveform params:', wnames
print 'tile length (km):', dxy * 1e-3
print 'tile projection:', proj
print 'n jobs:', njobs


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


print 'loading and transforming coords ...'


# PIG tile
#'merged_READ_SLOPE_IBE_TIDE_bbox_-1658294_-1458179_-473841_-265001_buff_5_epsg_3031_tile_114.h5'


def get_bboxs(ifile):
    """ Define cells (bbox) for estimating corrections. """

    #print 'building bboxes ...'
    with h5py.File(ifile, 'r') as fi:

        lon = fi[xvar][:]
        lat = fi[yvar][:]

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


def get_cell_corr(bbox):
    """ Calculate correction for data in cell (bbox). """

    # Bounding box of grid cell
    xmin, xmax, ymin, ymax = bbox
    
    # Convert lon/lat to sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Get the sub-tile indices
    i_cell, = np.where( (x >= xmin) & (x <= xmax) & 
                        (y >= ymin) & (y <= ymax) )

    # Test for enough points
    if len(i_cell) < 4:
        return

    # Get all points within the grid cell
    tc = t[i_cell]
    xc = x[i_cell]
    yc = y[i_cell]
    hc = h[i_cell]
    bsc = bsc_[i_cell]
    lew = lew_[i_cell]
    tes = tes_[i_cell]

    """ Filter anomalous data before anything else (lots of garbage!) """

    # Iterative median filter
    hc_f = mode_filter(hc.copy(), min_count=10, maxiter=3)
    bsc_f = mode_filter(bsc.copy(), min_count=10, maxiter=3)
    lew_f = mode_filter(lew.copy(), min_count=10, maxiter=3)
    tes_f = mode_filter(tes.copy(), min_count=10, maxiter=3)

    #FIXME: Next version use in here 3-sigma, maxiter=5
    # Iterative 5-sigma filter (remove gross outliers only)
    hc_f = sigma_filter(tc, hc_f,   n_sigma=3, frac=1/3., maxiter=2)
    bsc_f = sigma_filter(tc, bsc_f, n_sigma=3,frac=1/3., maxiter=2)
    lew_f = sigma_filter(tc, lew_f, n_sigma=3,frac=1/3., maxiter=2)
    tes_f = sigma_filter(tc, tes_f, n_sigma=3,frac=1/3., maxiter=2)

    # Plot
    if 0:
        plt.figure()
        plt.plot(tc, hc, '.')
        plt.plot(tc, hc_f, '.')

        plt.figure()
        plt.plot(tc, bsc, '.')
        plt.plot(tc, bsc_f, '.')

        plt.figure()
        plt.plot(tc, lew, '.')
        plt.plot(tc, lew_f, '.')

        plt.figure()
        plt.plot(tc, tes, '.')
        plt.plot(tc, tes_f, '.')

        plt.figure()
        plt.plot(xc, yc, '.')

        plt.show()
        #sys.exit()

    # Remove data points if any param is missing
    i_false = np.isnan(hc_f) | np.isnan(bsc_f) | np.isnan(lew_f) | np.isnan(tes_f)
    i_true = np.invert(i_false)

    hc_f[i_false] = np.nan
    bsc_f[i_false] = np.nan
    lew_f[i_false] = np.nan
    tes_f[i_false] = np.nan

    if len(hc_f[i_true]) < 5:
        return

    ###FIXME: Next version remove below. Height will be residuals 


    """ Remove mean (spatial pattern) """

    # Container for residuals
    dhc_f = hc_f.copy() * 0
    dbsc_f = bsc_f.copy() * 0
    dlew_f = lew_f.copy() * 0
    dtes_f = tes_f.copy() * 0
    h_topo = hc_f.copy() * 0

    # Deviation from centroid location 
    dx = xc - np.nanmedian(xc)
    dy = yc - np.nanmedian(yc)

    # Design matrix -> bi-quadratic surface model
    Ac = np.vstack((np.ones(dx.shape), dx, dy, dx*dy, dx*dx, dy*dy)).T
    
    try:
        # Construct robust surface model
        linear_model = sm.RLM(hc_f, Ac, M=sm.robust.norms.HuberT(), missing="drop")

        # Fit the model to the data
        linear_model_fit = linear_model.fit()

    except:
        return

    # Compute residuals => dh = h - Ax
    dhc_f[i_true] = linear_model_fit.resid

    # Get fitted topography
    h_topo[i_true] = linear_model_fit.fittedvalues


    # Remove the mean spatial pattern of waveform params
    if 1:

        # Design matrix -> bi-quadratic surface model
        #Ad = np.vstack((np.ones(dx.shape), dx, dy)).T
        #Ad = np.vstack((np.ones(dx.shape), dx, dy, dx*dy)).T
        Ad = np.vstack((np.ones(dx.shape), dx, dy, dx*dy, dx*dx, dy*dy)).T

        linear_model2 = sm.RLM(bsc_f, Ad, M=sm.robust.norms.HuberT(), missing="drop")
        linear_model3 = sm.RLM(lew_f, Ad, M=sm.robust.norms.HuberT(), missing="drop")
        linear_model4 = sm.RLM(tes_f, Ad, M=sm.robust.norms.HuberT(), missing="drop")

        dbsc_f[i_true] = linear_model2.fit().resid
        dlew_f[i_true] = linear_model3.fit().resid
        dtes_f[i_true] = linear_model4.fit().resid

    else:

        # Remove mean of waveform params
        dbsc_f = bsc_f - np.nanmean(bsc_f)
        dlew_f = lew_f - np.nanmean(lew_f)
        dtes_f = tes_f - np.nanmean(tes_f)


    ###FIXME: Next version remove up to here


    """ Clean-up data """

    # Filter the residuals
    """
    dhc_f = sigma_filter(tc, dhc_f,  n_sigma=3,  frac=1/3., maxiter=5)
    dbsc_f = sigma_filter(tc, dbsc_f, n_sigma=3, frac=1/3., maxiter=5)
    dlew_f = sigma_filter(tc, dlew_f, n_sigma=3, frac=1/3., maxiter=5)
    dtes_f = sigma_filter(tc, dtes_f, n_sigma=3, frac=1/3., maxiter=5)
    """

    # Update the valid/invalid indices 
    i_false = np.isnan(dhc_f) | np.isnan(dbsc_f) | np.isnan(dlew_f) | np.isnan(dlew_f)
    i_true = np.invert(i_false)

    # Return if no valid data points
    if not i_true.any():
        return

    # Remove points that have at least one param w/NaN
    hc_f[i_false] = np.nan
    bsc_f[i_false] = np.nan
    lew_f[i_false] = np.nan
    tes_f[i_false] = np.nan
    dhc_f[i_false] = np.nan
    dbsc_f[i_false] = np.nan
    dlew_f[i_false] = np.nan
    dtes_f[i_false] = np.nan

    """ Pre-process data """

    # Bin time series
    if 1:

        # Need enough data for binning (at least 1 year)
        if tc.max() - tc.min() < 1.0:
            return

        dhc_f_ = dhc_f.copy()
        dbsc_f_ = dbsc_f.copy()
        dlew_f_ = dlew_f.copy()
        dtes_f_ = dtes_f.copy()
        tc_ = tc.copy()

        tc_min, tc_max = tc.min(), tc.max()
        _, dhc_f = binning(tc, dhc_f, tc_min, tc_max,  1/12., 3/12.)[0:2]
        _, dbsc_f = binning(tc, dbsc_f, tc_min, tc_max,1/12., 3/12.)[0:2]
        _, dlew_f = binning(tc, dlew_f, tc_min, tc_max,1/12., 3/12.)[0:2]
        tc, dtes_f = binning(tc, dtes_f, tc_min, tc_max,1/12., 3/12.)[0:2]

    # Detrend time series
    if 0:

        dhc_f_ = dhc_f.copy()
        dbsc_f_ = dbsc_f.copy()
        dlew_f_ = dlew_f.copy()
        dtes_f_ = dtes_f.copy()
        tc_ = tc.copy()

        dhc_f, dhc_t = detrend(tc, dhc_f, frac=1/3.)
        dbsc_f, dbsc_t = detrend(tc, dbsc_f, frac=1/3.)
        dlew_f, dlew_t = detrend(tc, dlew_f, frac=1/3.)
        dtes_f, dtes_t = detrend(tc, dtes_f, frac=1/3.)

    # Difference time series
    if 0:

        dhc_f_ = dhc_f.copy()
        dbsc_f_ = dbsc_f.copy()
        dlew_f_ = dlew_f.copy()
        dtes_f_ = dtes_f.copy()
        tc_ = tc.copy()

        dhc_f = np.gradient(dhc_f)
        dbsc_f = np.gradient(dbsc_f)
        dlew_f = np.gradient(dlew_f)
        dtes_f = np.gradient(dtes_f)


    # Plot
    if 1:

        # Compute trend for ploting
        _, dhc_t = detrend(tc, dhc_f,   frac=1/3.)
        _, dbsc_t = detrend(tc, dbsc_f, frac=1/3.)
        _, dlew_t = detrend(tc, dlew_f, frac=1/3.)
        _, dtes_t = detrend(tc, dtes_f, frac=1/3.)

        plt.figure()
        plt.plot(tc, dhc_f, '-')
        plt.plot(tc, dhc_t, 'o')
        plt.figure()
        plt.plot(tc, dbsc_f, '-')
        plt.plot(tc, dbsc_t, 'o')
        plt.figure()
        plt.plot(tc, dlew_f, '-')
        plt.plot(tc, dlew_t, 'o')
        plt.figure()
        plt.plot(tc, dtes_f, '-')
        plt.plot(tc, dtes_t, 'o')
        plt.show()
        sys.exit()


    """ Construct design matrices """

    # Ensure zero mean
    #dhc_f -= np.nanmean(dhc_f)
    bsc_f -= np.nanmean(bsc_f)
    lew_f -= np.nanmean(lew_f)
    tes_f -= np.nanmean(tes_f)
    dbsc_f -= np.nanmean(dbsc_f)
    dlew_f -= np.nanmean(dlew_f)
    dtes_f -= np.nanmean(dtes_f)

    # First-order model
    Ac = np.vstack((dbsc_f, dlew_f, dtes_f)).T


    ###FIXME: Next version use 'if-else' and catch potential errors

    try:

        # Fit robust linear model on *clean* data
        sensitivity_model = sm.RLM(dhc_f, Ac, M=sm.robust.norms.HuberT(),
                                   missing="drop")

        result = sensitivity_model.fit()

        # Get sensitivity gradients of BsC, LeW, TeS
        SG = result.params
        s_bsc, s_lew, s_tes = SG[:3]

        """ Calculate correction to height """

        #FIXME: Check that this is correct!!!
        # If detrended or differenced, use original for correction
        if 1:

            dhc_f = dhc_f_
            dbsc_f = dbsc_f_
            dlew_f = dlew_f_
            dtes_f = dtes_f_
            tc = tc_

            Ac = np.vstack((dbsc_f, dlew_f, dtes_f)).T


        
        ### CHECK, SOMETHING IS WRONG WITH BINNING, DETREND, AND DIFF!!! ###



        # Multi-variate fit => h_bs = a BsC + b LeW + c TeS
        h_bs = np.dot(Ac, SG)


        ### CORRECT HEIGHTS ###
        hc_f_cor = hc_f - h_bs
        dhc_f_cor = dhc_f - h_bs


        #FIXME: Uncomment
        # Filter the corrected residuals
        #dhc_f_cor = sigma_filter(tc, dhc_f_cor,  n_sigma=3,  frac=1/3., maxiter=5)

        # Do not apply corr if std of residuals increases 
        if (np.nanstd(dhc_f_cor, ddof=1) > np.nanstd(dhc_f, ddof=1)):

            hc_f_cor = hc_f
            dhc_f_cor = dhc_f
            h_bs[:] = 0

        ###FIXME: Next version wrap 'remove NaNs' into a generic 'filter' function

        # Update the valid/invalid indices 
        i_false = np.isnan(dhc_f_cor)
        i_true = np.invert(i_false)

        # Remove points that have at least one NaN
        h_bs[i_false] = np.nan
        hc_f[i_false] = np.nan
        hc_f_cor[i_false] = np.nan
        bsc_f[i_false] = np.nan
        lew_f[i_false] = np.nan
        tes_f[i_false] = np.nan
        dhc_f[i_false] = np.nan
        dhc_f_cor[i_false] = np.nan
        dbsc_f[i_false] = np.nan
        dlew_f[i_false] = np.nan
        dtes_f[i_false] = np.nan


        # Compute sensitivity to multi-fit
        s_fit = np.polyfit(h_bs[i_true], dhc_f[i_true], 1)[0]

        # Compute correlation coefficients
        r_bsc = np.corrcoef(dhc_f[i_true], dbsc_f[i_true])[0, 1]
        r_lew = np.corrcoef(dhc_f[i_true], dlew_f[i_true])[0, 1]
        r_tes = np.corrcoef(dhc_f[i_true], dtes_f[i_true])[0, 1]
        r_fit = np.corrcoef(dhc_f[i_true], h_bs[i_true])[0, 1] 

        print 'r_fit =', round(r_bsc, 2)
        
        # Only for testing
        if 0:

            print 'before:'
            print 'CORR RESID h x BsC:', round(r_bsc, 2)
            print 'CORR RESID h x LeW:', round(r_lew, 2)
            print 'CORR RESID h x TeS:', round(r_tes, 2)
            print 'CORR RESID h x Fit:', round(r_fit, 2)

            # Compute correlation coefficients
            r_bsc = np.corrcoef(dhc_f_cor[i_true], dbsc_f[i_true])[0, 1]
            r_lew = np.corrcoef(dhc_f_cor[i_true], dlew_f[i_true])[0, 1]
            r_tes = np.corrcoef(dhc_f_cor[i_true], dtes_f[i_true])[0, 1]
            r_fit = np.corrcoef(dhc_f_cor[i_true], h_bs[i_true])[0, 1] 
            
            print 'after:'
            print 'CORR RESID h x BsC:', round(r_bsc, 2)
            print 'CORR RESID h x LeW:', round(r_lew, 2)
            print 'CORR RESID h x TeS:', round(r_tes, 2)
            print 'CORR RESID h x Fit:', round(r_fit, 2)

        # Plot
        if 0:

            if 0:
                plt.figure(figsize=(8,8))

                plt.subplot(211)
                plt.plot(tc, dhc_f, '.')
                plt.plot(tc, dhc_f_cor, '.')
                plt.title('RESID: Uncorr (std=%.2f) vs BScorr (std=%.2f)' \
                        % (np.nanstd(dhc_f), np.nanstd(dhc_f_cor)) )

                plt.subplot(212)
                plt.plot(tc, hc, '.')
                plt.plot(tc, hc_f_cor, '.')
                plt.title('FULL: Uncorr (std=%.2f, mean=%.2f) ' \
                          'vs BScorr (std=%.2f, mean=%.2f)' \
                        % (np.nanstd(hc_f), np.nanmean(hc_f), \
                        np.nanstd(hc_f_cor), np.nanmean(hc_f_cor)) )

                plt.figure()
                plt.plot(tc, h_bs, 'r.')

                plt.show()
                #sys.exit()

            else:

                print round(np.nanstd(dhc_f), 2), '=>', round(np.nanstd(dhc_f_cor), 2)

    except:
        return

    # Compute centroid of cell 
    xi = np.nanmedian(xc)
    yi = np.nanmedian(yc)

    # Convert back to geographical coordinates
    loni, lati = transform_coord(proj, '4326', xi, yi)

    # Set to zero where there is no bs corr (bad params) 
    h_bs[np.isnan(h_bs)] = 0.

    # Return corr and params for the bbox
    return [i_cell, h_bs,
            loni, lati,
            r_bsc, r_lew, r_tes, r_fit,
            s_bsc, s_lew, s_tes, s_fit]



# Loop through each input file
for ifile in ifiles:

    #print 'processing file:', ifile, '...'

    # Get bbox of all cells
    bboxs = get_bboxs(ifile)

    # Load full data to memory (only once)
    fi = h5py.File(ifile, 'a')

    t = fi[tvar][:]
    lon = fi[xvar][:]
    lat = fi[yvar][:]
    h = fi[zvar][:]
    bsc_ = fi[bpar][:]
    lew_ = fi[lpar][:]
    tes_ = fi[tpar][:]
    
    #NOTE: Filter time for Envisat only
    if 1:
        ind, = np.where(t > 2010.8)
        lon[ind] = np.nan
        lat[ind] = np.nan
        h[ind] = np.nan
        t[ind] = np.nan
        bsc_[ind] = np.nan
        lew_[ind] = np.nan
        tes_[ind] = np.nan

    # Select cells at random (for testing)
    if 1:
        np.random.seed(999)
        ii = range(len(bboxs))
        bboxs = np.array(bboxs)[np.random.choice(ii, 10)]


    # Loop trhough cells
    if njobs == 1:
        #print 'running sequential code ...'
        res = [get_cell_corr(bbox) for bbox in bboxs]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=njobs, verbose=5)(
                delayed(get_cell_corr)(bbox) for bbox in bboxs)

    """

    # Remove 'None's from result list
    res = [r for r in res if r is not None]

    # Generate stats (for testing)
    if 1:
        stds1 = np.full(len(res), np.nan) 
        stds2 = np.full(len(res), np.nan) 

        for k, r in enumerate(res):

            i_cell = r[0]
            cor_cell  = r[1]

            t_cell = t[i_cell]
            h_cell = h[i_cell]

            h_cor = h_cell - cor_cell

            # Remove all height where there is no bs corr
            # (This also filters all the bad data)
            h_cell[cor_cell==0] = np.nan
            h_cor[cor_cell==0] = np.nan

            stds1[k] = np.nanstd(h_cell)
            stds2[k] = np.nanstd(h_cor)

            '''
            plt.figure()
            plt.plot(t_cell, h_cell, '.')
            plt.plot(t_cell, h_cor, '.')
            plt.figure()
            plt.plot(t_cell, cor_cell, 'r.')
            plt.show()
            '''

        for s1,s2 in zip(stds1, stds2):
            print s1, s2

        print 'Mean of std:', np.mean(stds1), np.mean(stds2)

        #sys.exit()

    """

    """

    print 'Applying correction ...'

    # Output container for bs corr
    hbs = np.zeros_like(h) 

    # Apply correction (one cell at a time)

    # Apply corr to each time series
    for r in res:
        i_cell = r[0]
        cor_cell = r[1]
        h_cell = h[i_cell]

        ### FIXME: Do this only for xover analysis ###<<<<<<<<<<<<<<<<<<<<<<<
        cor_cell[cor_cell==0] = np.nan

        h_cor = h_cell - cor_cell

        h[i_cell] = h_cor
        hbs[i_cell] = cor_cell

    # Update h and save corr (all cells at once)
    fi[zvar][:] = h
    fi['h_bs'] = hbs

    fi.flush()
    fi.close()

    print 'Saving bs params ...'

    # Save bs params to external file 
    with h5py.File(ifile.replace('.h5', '_params.h5'), 'w') as fo:

        fo['lon']   = [r[2]  for r in res]
        fo['lat']   = [r[3]  for r in res]  
        fo['r_bsc'] = [r[4]  for r in res]
        fo['r_lew'] = [r[5]  for r in res]
        fo['r_tes'] = [r[6]  for r in res]
        fo['r_fit'] = [r[7]  for r in res]
        fo['s_bsc'] = [r[8]  for r in res]
        fo['s_lew'] = [r[9]  for r in res]
        fo['s_tes'] = [r[10] for r in res]
        fo['s_fit'] = [r[11] for r in res]

    """

print 'done all!'

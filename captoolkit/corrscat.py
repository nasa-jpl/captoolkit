#!/usr/bin/env python
"""
Waveform scattering correction of radar altimetry heights

Program corrects radar altimetry height measurements for variations in
surface scattering properties by exploiting correlations between height
residuals and waveform parameters (backscatter, leading-edge width,
trailing-edge slope or similar).

The correction is computed as a multivariate linear model where height
variability is expressed as a function of waveform parameters. The derived
sensitivities are then used to estimate and remove the scattering-related
component from the height time series:

    h_cor = h - (a*Bs + b*LeW + c*TeS)

where a, b, c are sensitivity coefficients estimated from the data.

The solution is computed locally within spatial grid cells defined by a user-
specified resolution. For each grid node, data within a search radius are
selected (with optional relocation to the data centroid), and a correction
model is fitted if sufficient observations are available.

Corrections from overlapping solutions are compared, and the best solution
(lowest RMS / best fit) is retained for each observation.

Missing waveform parameters are interpolated prior to processing, and any
remaining gaps in the correction field are filled using nearest-neighbor
interpolation.

Outputs:
    - Corrected scattering component (h_wc)
    - Fit quality metrics (R², RMS)
    - Correlation and sensitivity coefficients
    - Change in variance and trend after correction
    - Gridded parameter file summarizing spatial variability of sensitivities

Notes:
    - A minimum number of observations (default ~50) is required per grid cell
      to compute a valid solution.
    - Grid size and search radius should be chosen based on data density.
    - Optional preprocessing (-p) controls how sensitivity is estimated:
        * 'cen' : centered series (default)
        * 'det' : detrended series
        * 'dif' : differenced series
    - Time span (-t) can be specified to constrain sensitivity estimation.

Example:

    python scattcor.py -f /path/to/files/*.h5 \
        -v lon lat h_res t_year \
        -w bs lew tes \
        -d 1 -r 4 -q 2 -p cen -j 3031 -n 2

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Fernando Paolo (paolofer@jpl.nasa.gov)      (orignal author)
    Johan Nilsson  (johan.n.nilsson@geo.uu.se)  (major revisions)
    Alex Gardner   (alex.s.gardner@jpl.nasa.gov)

    Department of Earth Sciences, Uppsala University
    Jet Propulsion Laboratory, California Institute of Technology

"""

from scipy.signal import savgol_filter
from numba import jit
from altimutils import binning
from altimutils import fillnans
from altimutils import lstsq
from altimutils import transform_coord
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pyproj
import timeit
import glob
import h5py
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Minimum points per cell to compute solution
MIN_PTS = 50

def get_args():
    """ Get command-line arguments. """

    msg = 'Correct elevation data for variations in surface scattering.'
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument(
        '-f', metavar='file', dest='files', type=str, nargs='+',
        help='HDF5 file(s) to process',
        required=True)

    parser.add_argument(
        '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
        help=('block size of grid cell (km)'),
        default=[1], required=True)

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
        choices=('det', 'dif', 'cen'), default=['cen'],)

    parser.add_argument(
        '-v', metavar=('lon', 'lat', 'h', 't'), dest='vnames',
        type=str, nargs=4,
        help=('name of x/y/z/t variables in the HDF5'),
        default=['lon', 'lat', 'h', 't'], required=True)

    parser.add_argument(
        '-w', metavar=('bsc', 'lew', 'tes'), dest='wnames',
        type=str, nargs=3,
        help=('name of sig0/LeW/TeS parameters in HDF5'),
        default=['bsc', 'lew', 'tes'], required=True)

    parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

    parser.add_argument(
        '-t', metavar=('tmin','tmax'), dest='tspan', type=float, nargs=2,
        help=('time-span for sensitivity gradient fit (applied to ALL obs.)'),
        default=[None,None],)

    parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="number of jobs for parallel processing",
        default=[1],)

    parser.add_argument(
        '-c', dest='fcheck', action='store_true',
        help=('discard files already processed'),
        default=False)

    parser.add_argument(
        '-a', dest='apply', action='store_true',
        help=('apply correction to "h"'),
        default=False)

    return parser.parse_args()


def mad_std(x):
    return 1.4826 * np.nanmedian(np.abs(x-np.nanmedian(x)))


def center(*arrs):
    """ Remove median from array(s), return zeros if all elements are zero. """
    result = []
    for a in arrs:
        if np.all(a == 0):  # Check if all elements are zero
            result.append(np.zeros_like(a))  # Return an array of zeros
        else:
            result.append(a - np.nanmedian(a))  # Subtract median
    return result


def normalize(*arrs):
    """ Normalize array(s) by MAD std, return zeros if all elements are zero. """
    result = []
    for a in arrs:
        if np.all(a == 0):  # Check if all elements are zero
            result.append(np.zeros_like(a))  # Return an array of zeros
        else:
            result.append(a / mad_std(a))  # Normalize by MAD standard deviation
    return result


def corr_coef(h, bs, lew, tes):
    """ Get correlation coefficient between h and w/f params. If correlation is NaN, set it to zero. """

    # Remove NaNs
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) & ~np.isnan(lew) & ~np.isnan(tes))

    # If all values are NaN, return zero correlation
    if len(idx) == 0:
        return 0.0, 0.0, 0.0

    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]

    # Compute correlation coefficients safely
    r_bs = np.corrcoef(bs_, h_)[0, 1] if len(bs_) > 1 else np.nan
    r_lew = np.corrcoef(lew_, h_)[0, 1] if len(lew_) > 1 else np.nan
    r_tes = np.corrcoef(tes_, h_)[0, 1] if len(tes_) > 1 else np.nan

    # Replace NaN correlations with zero
    r_bs  = 0.0 if np.isnan(r_bs) else r_bs
    r_lew = 0.0 if np.isnan(r_lew) else r_lew
    r_tes = 0.0 if np.isnan(r_tes) else r_tes

    return r_bs, r_lew, r_tes


def corr_grad(h, bs, lew, tes, normalize=False):
    """ Get corr gradient (slope) between h and w/f params. """

    # Check if any array is entirely zero
    if any(np.all(arr == 0) for arr in [h, bs, lew, tes]):
        return 0.0, 0.0, 0.0

    # Remove NaNs
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) &
                    ~np.isnan(lew) & ~np.isnan(tes))
    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]

    # OLS line fit
    s_bs = np.polyfit(bs_, h_, 1)[0]
    s_lew = np.polyfit(lew_, h_, 1)[0]
    s_tes = np.polyfit(tes_, h_, 1)[0]

    if normalize:
        s_bs /= mad_std(bs_)
        s_lew /= mad_std(lew_)
        s_tes /= mad_std(tes_)

    return s_bs, s_lew, s_tes

def detrend(x, y, order=1):
    """
    Fit polynomial trend of given order on raw data and return detrended values and trend.
    """
    valid = ~np.isnan(y)
    if np.sum(valid) < order + 1:
        # Not enough points to fit
        return y.copy(), np.zeros_like(y)

    coef = np.polyfit(x[valid], y[valid], order)
    trend = np.polyval(coef, x)
    detrended = y - trend
    return detrended, trend

def sigma_filter(x, y, order=1, n_iter=3, n_sigma=3, threshold=np.inf):
    """
    Iteratively detrend data and remove outliers beyond n_sigma * MAD or exceeding threshold.

    Parameters:
        x: 1D array, independent variable (time, etc.)
        y: 1D array, observations
        order: polynomial degree for trend removal
        n_iter: number of iterations
        n_sigma: threshold in MAD units for outlier rejection
        threshold: absolute value threshold for rejection (default no threshold)

    Returns:
        y_filtered: copy of y with outliers replaced by np.nan
    """
    y_res = y.copy()

    for _ in range(n_iter):
        y_detrended, trend = detrend(x, y_res, order=order)
        scale = mad_std(y_detrended)
        if scale == 0:
            break

        # Outliers beyond n_sigma * MAD
        outliers_mad = np.abs(y_detrended) > n_sigma * scale

        # Outliers beyond threshold on original values
        outliers_thresh = np.abs(y_res) > threshold

        outliers = outliers_mad | outliers_thresh

        if not np.any(outliers):
            break

        y_res[outliers] = np.nan

        if np.sum(~np.isnan(y_res)) < order + 1:
            break

    return y_res

def detrend_vars(t, h, bs, lew, tes):
    """ detrend varibales """

    # Detrend the data
    h   -= detrend(t, h.copy(),   order=1)[1]
    bs  -= detrend(t, bs.copy(),  order=1)[1]
    lew -= detrend(t, lew.copy(), order=1)[1]
    tes -= detrend(t, tes.copy(), order=1)[1]

    return h, bs, lew, tes

def gradient(*args):
    """
    Difference consecutive samples while preserving data gaps and size.

    Differences are calculated between consecutive finite samples. The
    first sample and samples following missing data are returned as NaN.

    :param args: arrays to difference
    :return: list of differenced arrays
    """

    out = []

    for var in args:
        var = np.asarray(var)
        dv = np.full(var.shape, np.nan, dtype=float)

        good = (
            np.isfinite(var[:-1]) &
            np.isfinite(var[1:])
        )

        dv[1:][good] = np.diff(var)[good]
        out.append(dv)

    return out

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


def get_radius_idx(x, y, x0, y0, r, tree, n_reloc=0):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the node
    idx = tree.query_ball_point([x0, y0], r)

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

        # Query from the new location
        idx = tree.query_ball_point([x0_new, y0_new], r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

    return idx


def get_scatt_cor(t, h, bs, lew, tes, proc='cen', tmin=None, tmax=None):
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

    if tmin is not None:
        t_min = tmin
    else:
        t_min = t.min()
    if tmax is not None:
        t_max = tmax
    else:
        t_max = t.max()

    # copy entries
    t_   = t.copy()
    h_   = h.copy()
    bs_  = bs.copy()
    lew_ = lew.copy()
    tes_ = tes.copy()

    # Filter variables
    h_   = sigma_filter(t_, h_,   order=1, n_iter=3, n_sigma=10)
    bs_  = sigma_filter(t_, bs_,  order=1, n_iter=3, n_sigma=10)
    lew_ = sigma_filter(t_, lew_, order=1, n_iter=3, n_sigma=10)
    tes_ = sigma_filter(t_, tes_, order=1, n_iter=3, n_sigma=10)

    # Binning of the data - three month smoothing
    t_b, h_b,_, n_b = binning(t_, h_,   xmin=t_min, xmax=t_max,
                        dx=1./12, window=1./12, median=True)[0:4]
    t_b, bs_b       = binning(t_, bs_,  xmin=t_min, xmax=t_max,
                        dx=1./12, window=1./12, median=True)[0:2]
    t_b, lew_b      = binning(t_, lew_, xmin=t_min, xmax=t_max,
                        dx=1./12, window=1./12, median=True)[0:2]
    t_b, tes_b      = binning(t_, tes_, xmin=t_min, xmax=t_max,
                        dx=1./12, window=1./12, median=True)[0:2]
                        
    # Remove values that are based on to few values
    h_b[n_b < 5] = np.nan
    
    # Check for detredning
    if proc == 'det':
        # Remove the temporal trend
        h_b, bs_b, lew_b, tes_b = detrend_vars(t_b, h_b, bs_b, lew_b, tes_b)

    # Check for difference
    if proc == 'dif':
        # Difference each sample
        h_b, bs_b, lew_b, tes_b = gradient(h_b, bs_b, lew_b, tes_b)

    """
    plt.figure(figsize=(10,4))
    plt.plot(t_,h_,'.')
    plt.plot(t_b,h_b,'o')
    plt.figure(figsize=(10,4))
    plt.plot(t_,bs_,'.')
    plt.plot(t_b,bs_b,'o')
    plt.show()
    """
    
    # Construct design matrix:
    A_i = np.vstack((bs, lew, tes)).T
    A_b = np.vstack((bs_b, lew_b, tes_b)).T

    # Only get finityye data
    mask = np.isfinite(bs_b) & np.isfinite(lew_b) & np.isfinite(tes_b)

    # Estimate model parameters
    p, ep, i_bad = lstsq(A_b[mask,:], h_b[mask], n_iter=5, n_sigma=10)

    # Linear combination of time-series => h_bs = a Bs + b LeW + c TeS
    h_wc = np.dot(A_i, p)

    # Compute model residuals
    res = h - h_wc

    # Compute RMS of fit
    rms = mad_std(res)

    # Compute sum of squares and total sum of squares
    RSS = np.nansum(res**2)
    TSS = np.nansum((h - np.nanmean(h))**2)

    # Length of data
    n = len(h_b[~np.isnan(h_b)])

    # Get adjusted r-squared -> model performance metric
    r2 = 1.0 - RSS / TSS
    r2 = 1 - (1 - r2)*(n - 1)/(n - 3 - 1)

    return h_wc, p, r2, rms


def std_change(t, x1, x2):
    """
    Compute variance change from x1 to x2 (magnitude and percentage).

    """
    idx = ~np.isnan(x1) & ~np.isnan(x2)
    t_, x1_, x2_ = t[idx], x1[idx], x2[idx]
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


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, \
        proc=None, tspan=[None,None], apply=False):

    import warnings
    warnings.filterwarnings("ignore")

    print('processing file:', ifile, '...')

    # Ignore parameter file if in list
    if "SCATGRD" in ifile:
        print("Skipping file ->",ifile)
        return

    # Test if parameter file exists and skip if true
    if check is True:
        if os.path.exists(ifile.replace(".h5","_SCATGRD.h5")):
            print("Skipping file already processed ->",ifile)
            return

    # Get variable names
    xvar, yvar, zvar, tvar = vnames
    bpar, wpar, spar = wnames

    # Time span for fitting
    tmin,tmax = tspan[:]

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:

        # Check if we can read file
        try:
            t   = fi[tvar][:]
            h   = fi[zvar][:]
            lon = fi[xvar][:]
            lat = fi[yvar][:]
            bs  = fi[bpar][:] if bpar in fi else np.zeros(lon.shape)
            lew = fi[wpar][:] if wpar in fi else np.zeros(lon.shape)
            tes = fi[spar][:] if spar in fi else np.zeros(lon.shape)
        except:

            # Otherwise run next
            print('could not open file:', ifile)
            return

    # Check for empty file
    if len(h) == 0 or len(h[~np.isnan(h)]) == 0: return

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Get nodes of solution grid
    x_nodes, y_nodes = get_grid_nodes(x, y, dxy, proj=proj)

    N_data = len(x)
    N_nodes = len(x_nodes)

    # Values for each data point
    r2fit  = np.full(N_data, 0.0)       # r2 of the multivar fit
    rmsfit = np.full(N_data, 0.0)       # r2 of the multivar fit
    pval   = np.full(N_data, np.nan)    # r2 of the multivar fit
    dstd   = np.full(N_data, np.nan)    # magnitude std change after cor
    dtrend = np.full(N_data, np.nan)    # magnitude trend change after cor
    pstd   = np.full(N_data, np.nan)    # perc std change after cor
    ptrend = np.full(N_data, np.nan)    # perc trend change after cor
    hbs    = np.full(N_data, np.nan)    # scatt cor from multivar fit
    rbs    = np.full(N_data, np.nan)    # corr coef h x Bs
    rlew   = np.full(N_data, np.nan)    # corr coef h x LeW
    rtes   = np.full(N_data, np.nan)    # corr coef h x TeS
    sbs    = np.full(N_data, np.nan)    # sensit h x Bs
    slew   = np.full(N_data, np.nan)    # sensit h x LeW
    stes   = np.full(N_data, np.nan)    # sensit h x TeS
    bbs    = np.full(N_data, np.nan)    # multivar fit coef a.Bs
    blew   = np.full(N_data, np.nan)    # multivar fit coef b.LeW
    btes   = np.full(N_data, np.nan)    # multivar fit coef c.TeS
    bs_med = np.full(N_data, np.nan)    # local Bs median
    lw_med = np.full(N_data, np.nan)    # local LeW median
    ts_med = np.full(N_data, np.nan)    # local TeS median
    bs_scl = np.full(N_data, np.nan)    # local Bs MAD scale
    lw_scl = np.full(N_data, np.nan)    # local LeW MAD scale
    ts_scl = np.full(N_data, np.nan)    # local TeS MAD scale

    # Values for each node
    r2fitc  = np.full(N_nodes, 0.0)
    pvalc   = np.full(N_nodes, np.nan)
    dstdc   = np.full(N_nodes, np.nan)
    dtrendc = np.full(N_nodes, np.nan)
    pstdc   = np.full(N_nodes, np.nan)
    ptrendc = np.full(N_nodes, np.nan)
    rbsc    = np.full(N_nodes, np.nan)
    rlewc   = np.full(N_nodes, np.nan)
    rtesc   = np.full(N_nodes, np.nan)
    sbsc    = np.full(N_nodes, np.nan)
    slewc   = np.full(N_nodes, np.nan)
    stesc   = np.full(N_nodes, np.nan)
    bbsc    = np.full(N_nodes, np.nan)
    blewc   = np.full(N_nodes, np.nan)
    btesc   = np.full(N_nodes, np.nan)
    lonc    = np.full(N_nodes, np.nan)
    latc    = np.full(N_nodes, np.nan)

    # Create KD-tree
    tree = cKDTree(np.c_[x, y])
    
    # Find values that have no solution
    i_bs  = np.isnan(bs)
    i_lew = np.isnan(lew)
    i_tes = np.isnan(tes)
    i_h   = np.isnan(h)

    print("interpolating waveform data arrays ...")

    # Interpolate and fill empties using nearest data points
    bs[i_bs]   = griddata((t[~i_bs], x[~i_bs], y[~i_bs]), bs[~i_bs],
                        (t[i_bs], x[i_bs], y[i_bs]),    'nearest')
    lew[i_lew] = griddata((t[~i_lew], x[~i_lew], y[~i_lew]), lew[~i_lew],
                          (t[i_lew], x[i_lew], y[i_lew]), 'nearest')
    tes[i_tes] = griddata((t[~i_tes], x[~i_tes], y[~i_tes]), tes[~i_tes],
                          (t[i_tes], x[i_tes], y[i_tes]), 'nearest')
    
    # Loop through nodes
    for k in range(N_nodes):

        # Get current nodes
        xi, yi = x_nodes[k], y_nodes[k]

        # Get indices of data within search radius
        i_cell = get_radius_idx(x, y, xi, yi, radius, tree, n_reloc=n_reloc)

        # If cell empty or not enough data go to next node
        if len(i_cell) < MIN_PTS:
            continue

        # Get all data within the grid search radius
        tc = t[i_cell]
        hc = h[i_cell]
        xc = x[i_cell]
        yc = y[i_cell]
        bc = bs[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        # Keep original (unfiltered) data
        tc_orig, hc_orig = tc.copy(), hc.copy()

        # Test minimum number of obs in all params
        nobs = min([len(v[~np.isnan(v)]) for v in [hc, bc, wc, sc]])

        # Test for enough points
        if (nobs < MIN_PTS): continue

        # Center height and waveform parameters
        hc = hc - np.nanmedian(hc)
        bmed = np.nanmedian(bc)
        wmed = np.nanmedian(wc)
        smed = np.nanmedian(sc)
        bc = bc - bmed
        wc = wc - wmed
        sc = sc - smed

        # Normalize waveform parameters by robust MAD scale
        bscale = mad_std(bc)
        wscale = mad_std(wc)
        sscale = mad_std(sc)
        if bscale == 0 or wscale == 0 or sscale == 0:
            continue
        bc /= bscale
        wc /= wscale
        sc /= sscale

        # Calculate correction for data in search radius
        hc_wc, p_fit, r2, rms_fit = get_scatt_cor(tc, hc, bc, wc, sc, proc,\
                                                 tmin, tmax)

        # Make sure this works
        if np.any(np.isnan(p_fit)):
            continue

        # Sensitivity gradients from model
        b_bc, b_wc, b_sc = p_fit

        # Reject if not enough points
        if len(hc[~np.isnan(hc)]) < MIN_PTS: continue

        # If no correction could be generated, skip
        if (hc_wc == 0).all(): continue

        # If noise to high, skip
        if rms_fit > 5: continue

        # Apply correction to height
        hc_cor = hc - hc_wc

        # Calculate correlation between h and waveform params
        r_bc, r_wc, r_sc = corr_coef(hc, bc, wc, sc)

        # Calculate sensitivity values (corr grad)
        s_bc, s_wc, s_sc = corr_grad(hc, bc, wc, sc, normalize=False)

        # Calculate variance change (magnitude and perc)
        d_std, p_std = std_change(tc, hc, hc_cor)

        # Calculate trend change (magnitude and perc)
        d_trend, p_trend = trend_change(tc, hc, hc_cor)

        # Set filtered out values (not used in the calculation) to NaN
        hc_wc[np.isnan(hc)] = np.nan

        # Filter correction here
        hc_wc[np.abs(hc_wc) > 10] = np.nan

        # Check if previously stored values need update (r2_prev < r2_new)
        i_update, = np.where(r2fit[i_cell] <= np.abs(rms_fit))

        # Only keep the indices/values that need update
        i_cell_new = [i_cell[i] for i in i_update]
        hc_wc_new = hc_wc[i_update]

        # Make sure corrections are not to large
        hc_wc_new[np.abs(hc_wc_new) > 10] = np.nan

        # Store correction for cell (only improved values)
        hbs[i_cell_new]    = hc_wc_new
        r2fit[i_cell_new]  = r2
        dstd[i_cell_new]   = d_std
        pstd[i_cell_new]   = p_std
        dtrend[i_cell_new] = d_trend
        ptrend[i_cell_new] = p_trend
        rbs[i_cell_new]    = r_bc
        rlew[i_cell_new]   = r_wc
        rtes[i_cell_new]   = r_sc
        sbs[i_cell_new]    = s_bc
        slew[i_cell_new]   = s_wc
        stes[i_cell_new]   = s_sc
        bbs[i_cell_new]    = b_bc
        blew[i_cell_new]   = b_wc
        btes[i_cell_new]   = b_sc
        bs_med[i_cell_new] = bmed
        lw_med[i_cell_new] = wmed
        ts_med[i_cell_new] = smed
        bs_scl[i_cell_new] = bscale
        lw_scl[i_cell_new] = wscale
        ts_scl[i_cell_new] = sscale

        # Compute centroid of cell
        xc_ = np.nanmedian(xc)
        yc_ = np.nanmedian(yc)

        # Convert x/y -> lon/lat
        lon_c, lat_c = transform_coord(proj, 4326, xc_, yc_)

        # Store one s and r value per cell
        lonc[k]    = lon_c
        latc[k]    = lat_c
        r2fitc[k]  = r2
        dstdc[k]   = d_std
        pstdc[k]   = p_std
        dtrendc[k] = d_trend
        ptrendc[k] = p_trend
        rbsc[k]    = r_bc
        rlewc[k]   = r_wc
        rtesc[k]   = r_sc
        sbsc[k]    = s_bc
        slewc[k]   = s_wc
        stesc[k]   = s_sc
        bbsc[k]    = b_bc
        blewc[k]   = b_wc
        btesc[k]   = b_sc

        # Time span of data inside solution area
        tspan = np.nanmax(tc_orig) - np.nanmin(tc_orig)

        if (k % 1) == 0:
            print('Node', k, 'of', N_nodes,
                  'Trend:', np.around(d_trend, 3), 'Std.dev:',
                  np.around(d_std, 3), 'Time-Span:', np.around(tspan, 3),
                  'Nobs:', len(hc[~np.isnan(hc)]), 'Corr:', np.array([r_bc, r_wc, r_sc]).round(2))

    # Find values that have no solution
    i_ip = np.isnan(hbs) | (hbs == 0)
    print('************************************')
    print('-> filling correction to nans from nearest local model')
    try:

        # Valid observations carrying an accepted local scattering model
        i_ok = (~i_ip & np.isfinite(bbs) & np.isfinite(blew) &
                np.isfinite(btes) & np.isfinite(bs_med) &
                np.isfinite(lw_med) & np.isfinite(ts_med) &
                np.isfinite(bs_scl) & np.isfinite(lw_scl) &
                np.isfinite(ts_scl))

        if np.any(i_ip) and np.any(i_ok):
            tree_fill = cKDTree(np.c_[x[i_ok], y[i_ok]])
            _, ii = tree_fill.query(np.c_[x[i_ip], y[i_ip]], k=1)

            # Scale each target waveform using the nearest local model
            bc_i = (bs[i_ip] - bs_med[i_ok][ii]) / bs_scl[i_ok][ii]
            wc_i = (lew[i_ip] - lw_med[i_ok][ii]) / lw_scl[i_ok][ii]
            sc_i = (tes[i_ip] - ts_med[i_ok][ii]) / ts_scl[i_ok][ii]

            # Evaluate the nearest local model using the target waveform
            hbs[i_ip] = (bbs[i_ok][ii] * bc_i +
                         blew[i_ok][ii] * wc_i +
                         btes[i_ok][ii] * sc_i)

    except Exception:

        # If it does not work
        print('************************************')
        print('-> interpolation not successful')
        pass

    """ Save data """

    print('-> saving data ...')

    with h5py.File(ifile, 'a') as fi:

        # Update h in the file and save correction (all cells at once)
        fi[zvar][:] = h

        # Try to create variables
        try:

            # Save params for each point
            fi['h_wc'] = hbs
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

        # Update variables instead
        except:

            # Save params for each point
            fi['h_wc'][:] = hbs
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

        if apply:
            with h5py.File(ifile, 'a') as fi:
                try:
                    fi[zvar+"_wc"] = h - hbs
                except:
                    fi[zvar+"_wc"][:] = h - hbs

        # Only rename file if _SCAT has not been added
        if ifile.find('_SCAT.h5') < 0:

            # Rename file
            os.rename(ifile, ifile.replace('.h5', '_SCAT.h5'))

        # Compute statistics
        hbs_cm = hbs[(hbs != 0) & ~np.isnan(hbs)]
        hbs_mean = np.median(hbs_cm) * 100
        hbs_stdv = mad_std(hbs_cm) * 100
        n_before = np.around(len(hbs[i_ip])/float(len(hbs))*100)
        n_afters = np.around(len(hbs[np.isnan(hbs) | (hbs == 0)]) /
                             float(len(hbs))*100)

        o_var_name = zvar + "_wc" if apply else None

        # Some output statistics
        print('-> empty before interp (%):', n_before)
        print('-> empty after interp  (%):', n_afters)
        print('-> average magnitude of correction:', np.around(hbs_mean, 0), '+/-',
              np.around(hbs_stdv, 0), '(cm)')
        print('-> corrected data saved in variable:',o_var_name)

        # Find NaNs and remove saves space
        i_NAN = ~np.isnan(ptrendc)

        # Save bs params as external file
        with h5py.File(ifile.replace('.h5', '_SCATGRD.h5'), 'w') as fo:

            # Try to save grid variables
            try:

                # Save varibales
                fo['lon'] = lonc[i_NAN]
                fo['lat'] = latc[i_NAN]
                fo['r2'] = r2fitc[i_NAN]
                fo['d_std'] = dstdc[i_NAN]
                fo['p_std'] = pstdc[i_NAN]
                fo['d_trend'] = dtrendc[i_NAN]
                fo['p_trend'] = ptrendc[i_NAN]
                fo['r_bs'] = rbsc[i_NAN]
                fo['r_lew'] = rlewc[i_NAN]
                fo['r_tes'] = rtesc[i_NAN]
                fo['s_bs'] = sbsc[i_NAN]
                fo['s_lew'] = slewc[i_NAN]
                fo['s_tes'] = stesc[i_NAN]
                fo['b_bs'] = bbsc[i_NAN]
                fo['b_lew'] = blewc[i_NAN]
                fo['b_tes'] = btesc[i_NAN]

            # Catch any exceptions
            except:

                # Exit program
                print('could not save parameters')
                return


if __name__ == '__main__':

    # Supress any warnings
    import warnings
    warnings.filterwarnings('ignore')

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
    check = args.fcheck            # check for already processed files
    tspan = args.tspan[:]          # time span for SG fit; applied to all data
    apply = args.apply             # apply the correction to the input data

    print('parameters:')
    for arg in vars(args).items():
        print(arg)

    if njobs == 1:
        print('running sequential code ...')
        [main(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc, tspan, apply)
         for ifile in ifiles]

    else:
        print('running parallel code (%d jobs) ...' % njobs)
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("loky", inner_max_num_threads=1):
            Parallel(n_jobs=njobs,)(
                delayed(main)(ifile, vnames, wnames, dxy, proj, radius,
                              nreloc, proc, tspan, apply) for ifile in ifiles)

    print('-> program finished')
    print('************************************')

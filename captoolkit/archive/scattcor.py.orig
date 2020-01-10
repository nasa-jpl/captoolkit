#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrects radar altimetry height to correlation with waveform parameters.

Example:
    scattcor.py -d 5 -v lon lat h_cor t_year -w bs_ice1 lew_ice2 tes_ice2 \
            -n 8 -f ~/data/envisat/all/bak/*.h5

Notes:
    The (back)scattering correction is applied as:

        h_cor = h - h_bs

"""
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
            '-i', metavar=('nreloc'), dest='nreloc', type=int, nargs=1,
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
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help="number of jobs for parallel processing",
            default=[1],)

    return parser.parse_args()


""" Generic functions """


def binning(x, y, xmin=None, xmax=None, dx=1/12., window=3/12., interp=False):
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
    trend = sm.nonparametric.lowess(y[idx], x[idx], frac=frac)[:,1]
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


def detrend(x, y, frac=1/3.):
    """ Detrend using LOWESS. """
    trend = sm.nonparametric.lowess(y, x, frac=frac)[:,1]
    if np.isnan(trend).all():
        trend = np.zeros_like(x)
    elif np.isnan(y).any():
        trend = np.interp(x, x[~np.isnan(y)], trend)
    return y-trend, trend


def center(*arrs):
    """ Remove mean from array(s). """
    return [a - np.nanmean(a) for a in arrs]


def corr_coef(x, *arrs):
    """ Get corr coef between x and arrs = [arr1, arr2...]. """ 
    return [np.corrcoef(x[(~np.isnan(x))&(~np.isnan(y))],
                        y[(~np.isnan(x))&(~np.isnan(y))])[0,1] \
                        for y in arrs]


""" Helper functions """


def filter_data(t, h, bs, lew, tes):
    """ Use various filters to remove outliers. """

    # Iterative median filter
    h = mode_filter(h, min_count=10, maxiter=3)
    bs = mode_filter(bs, min_count=10, maxiter=3)
    lew = mode_filter(lew, min_count=10, maxiter=3)
    tes = mode_filter(tes, min_count=10, maxiter=3)

    # Iterative 3-sigma filter
    h = sigma_filter(t, h, n_sigma=3, frac=1/3., maxiter=3, lowess=True)
    bs = sigma_filter(t, bs, n_sigma=3,frac=1/3., maxiter=3, lowess=True)
    lew = sigma_filter(t, lew, n_sigma=3,frac=1/3., maxiter=3, lowess=True)
    tes = sigma_filter(t, tes, n_sigma=3,frac=1/3., maxiter=3, lowess=True)

    # Non-iterative median filter
    h = median_filter(h, n_median=3)

    '''
    # Remove data points if any param is missing
    i_false = np.isnan(h) | np.isnan(bs) | np.isnan(lew) | np.isnan(tes)

    h[i_false] = np.nan
    bs[i_false] = np.nan
    lew[i_false] = np.nan
    tes[i_false] = np.nan
    '''
    return t, h, bs, lew, tes


def get_bboxs(lon, lat, dxy, proj='3031'):
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


#NOTE: DEPRECATED
'''
def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0):
    """ Get indexes of all data points inside radius. """

    # Query the Tree from the center of cell 
    idx = Tree.query_ball_point((x0, y0), r)
    
    # Relocate center of search radius and query again 
    for k in range(n_reloc):
        if len(idx) < 2 :
            break
        idx = Tree.query_ball_point((np.median(x[idx]), np.median(y[idx])), r)
        print 'relocation #', k

    return idx
'''

def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0,
        min_months=24, max_reloc=4, time=None):
    """ Get indexes of all data points inside radius. """

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

        print 'query #:', k+2, '( reloc #:', k+1, ')'

        idx = Tree.query_ball_point((np.median(x[idx]), np.median(y[idx])), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

        # If time provided, keep relocating until coverage is sufficient 
        if time is not None:

            t_b, x_b = binning(time[idx], x[idx], dx=1/12., window=1/12.)[:2]

            print 'months #:', np.sum(~np.isnan(x_b))

            # If sufficient coverage, exit
            if np.sum(~np.isnan(x_b)) >= min_months:
                break

    return idx


def get_scatt_cor(t, h, bs, lew, tes, proc=None):
    """
    Calculate scattering correction for height time series.

    Computes a multivariate fit to waveform parameters as:

        h(t) = a Bs(t) + b LeW(t) + c TeS(t)

    where a, b, and c are the sensitivity of h to each waveform param.

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

        print 'BINNED'

        # Need enough data for binning (at least 1 year)
        if t.max() - t.min() < 1.0:
            return [None] * 4

        h = binning(t, h, dx=1/12., window=3/12., interp=True)[1]
        bs = binning(t, bs, dx=1/12., window=3/12., interp=True)[1]
        lew = binning(t, lew, dx=1/12., window=3/12., interp=True)[1]
        tes = binning(t, tes, dx=1/12., window=3/12., interp=True)[1]

    # Detrend time series
    elif proc == 'det':

        print 'DETRENDED'

        h = detrend(t, h, frac=1/3.)[0]
        bs = detrend(t, bs, frac=1/3.)[0]
        lew = detrend(t, lew, frac=1/3.)[0]
        tes = detrend(t, tes, frac=1/3.)[0]

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

    try:
        # Fit robust linear model on clean data
        model = sm.RLM(h, A_proc, M=sm.robust.norms.HuberT(), missing="drop").fit(maxiter=3)

        # Get sensitivity gradients (coefficients) for Bs, LeW, TeS
        s_bs, s_lew, s_tes = model.params[:3]

        # Get multivariate fit => h_bs = a Bs + b LeW + c TeS
        h_bs = np.dot(A_orig, [s_bs, s_lew, s_tes])

        #NOTE 1: Use np.dot instead of .fittedvalues to keep NaNs from A
        #NOTE 2: Correction is generated using the original parameters

    except:
        print 'COULD NOT DO MULTIVARIATE FIT. SKIPING!!!'
        return [None] * 4

    return [h_bs, s_bs, s_lew, s_tes]


def apply_scatt_cor(t, h, h_bs, filt=False, test_std=False):
    """ Apply correction (if decreases std of residuals). """

    h_cor = h - h_bs 

    #FIXME: Check if or how this should be!
    if filt:
        h_cor = sigma_filter(t, h_cor,  n_sigma=3,  frac=1/3., lowess=True, maxiter=1)

    if test_std:
        # Detrend both time series for estimating std(res)
        h_r = detrend(t, h, frac=1/3.)[0]
        h_cor_r = detrend(t, h_cor, frac=1/3.)[0]

        idx, = np.where(~np.isnan(h_r) & ~np.isnan(h_cor_r))

        # Do not apply cor if std(res) increases 
        if h_cor_r[idx].std(ddof=1) > h_r[idx].std(ddof=1):
            h_cor = h
            h_bs[:] = 0.  # cor is set to zero

    return h_cor, h_bs


def var_reduction(x1, x2, y):
    """ Compute the variance reduction from x1 to x2. """
    idx = ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(y)
    return 1 - x2[idx].std(ddof=1)/x1[idx].std(ddof=1)


def plot(xc, yc, tc, hc, bc, wc, sc, h_cor, h_bs, r_bc, r_wc, r_sc):

    # Plot only valid (corrected) points
    idx = ~np.isnan(h_cor) & ~np.isnan(h_bs) & ~np.isnan(tc)
    
    if len(idx) == 0:
        return
    
    # Bin variables
    hc_b = binning(tc[idx], h_cor[idx])[1]
    bc_b = binning(tc[idx], bc[idx])[1]
    wc_b = binning(tc[idx], wc[idx])[1]
    tc_b, sc_b = binning(tc[idx], sc[idx])[:2]
    
    # Correlate variables
    r_bc2, r_wc2, r_sc2 = corr_coef(h_cor, bc, wc, sc)
    
    plt.subplot(4,1,1)
    plt.plot(tc[idx], hc[idx], '.')
    plt.plot(tc[idx], h_cor[idx], '.')
    plt.plot(tc_b, hc_b, '-')
    
    plt.title('Height')
    plt.subplot(4,1,2)
    plt.plot(tc[idx], bc[idx], '.')
    plt.plot(tc_b, bc_b, '-')
    plt.title('Bs')
    
    plt.subplot(4,1,3)
    plt.plot(tc[idx], wc[idx], '.')
    plt.plot(tc_b, wc_b, '-')
    plt.title('LeW')
    
    plt.subplot(4,1,4)
    plt.plot(tc[idx], sc[idx], '.')
    plt.plot(tc_b, sc_b, '-')
    plt.title('TeS')
    
    plt.figure()
    plt.plot(xc[idx], yc[idx], '.')
    plt.title('Tracks')
    
    # Detrend variables
    hc_r, _ = detrend(tc, hc, frac=1/3.)
    h_cor_r, _ = detrend(tc, h_cor, frac=1/3.)
    
    print 'Summary:'
    print 'std_unc:  ', hc_r[idx].std(ddof=1)
    print 'std_cor:  ', h_cor_r[idx].std(ddof=1)
    print ''
    print 'trend_unc:', np.polyfit(tc[idx], hc[idx], 1)[0]
    print 'trend_cor:', np.polyfit(tc[idx], h_cor[idx], 1)[0]
    print ''
    print 'hxbs_unc:  ', r_bc
    print 'hxlew_unc: ', r_wc
    print 'hxtes_unc: ', r_sc
    print ''
    print 'hxbs_cor:  ', r_bc2
    print 'hxlew_cor: ', r_wc2
    print 'hxtes_cor: ', r_sc2
    plt.show()


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, proc=None):

    print 'processing file:', ifile, '...'

    xvar, yvar, zvar, tvar = vnames
    bpar, wpar, spar = wnames

    # Load full data into memory (only once)
    fi = h5py.File(ifile, 'a')

    t = fi[tvar][:]
    h = fi[zvar][:]
    lon = fi[xvar][:]
    lat = fi[yvar][:]
    bs = fi[bpar][:]
    lew = fi[wpar][:]
    tes = fi[spar][:]


    # Filter time
    if 1:
        #idx, = np.where(t >= 1992)
        idx, = np.where(t <= 2010.8)
        t = t[idx]
        h = h[idx]
        lon = lon[idx]
        lat = lat[idx]
        bs = bs[idx]
        lew = lew[idx]
        tes = tes[idx]


    #TODO: Replace by get_grid?
    # Get bbox of all cells
    bboxs = get_bboxs(lon, lat, dxy, proj=proj)

    # Output containers
    N = len(bboxs)
    hbs = np.zeros_like(h) 
    perc = np.zeros_like(h) 
    rbs = np.full(N, np.nan) 
    rlew = np.full(N, np.nan) 
    rtes = np.full(N, np.nan) 
    sbs = np.full(N, np.nan) 
    slew = np.full(N, np.nan) 
    stes = np.full(N, np.nan) 
    loni = np.full(N, np.nan) 
    lati = np.full(N, np.nan) 

    # Select cells at random (for testing)
    if 0:
        n_cells = 50
        np.random.seed(999)  # not so random!
        ii = range(len(bboxs))
        bboxs = np.array(bboxs)[np.random.choice(ii, n_cells)]

    # Build KD-Tree with polar stereo coords (if a radius is provided)
    if radius > 0:
        x, y = transform_coord(4326, proj, lon, lat)
        Tree = cKDTree(zip(x, y))

    # Loop through cells
    for k,bbox in enumerate(bboxs):

        print 'Calculating correction for cell', k, '...'

        # Get indexes of data within search radius or cell bbox
        if radius > 0:
            i_cell = get_radius_idx(x, y, bbox[0], bbox[2],
                                    radius, Tree, n_reloc=n_reloc,
                                    min_months=24, max_reloc=4, time=t)
        else:
            i_cell = get_cell_idx(lon, lat, bbox, proj=proj)

        # Test for enough points
        if len(i_cell) < 4:
            continue

        # Get all data within the grid cell/search radius
        tc = t[i_cell]
        hc = h[i_cell]
        xc = lon[i_cell]
        yc = lat[i_cell]
        bc = bs[i_cell]
        wc = lew[i_cell]
        sc = tes[i_cell]

        # Only proceed if sufficient temporal coverage
        h_b = binning(tc, hc, dx=1/12., window=1/12.)[1]

        if np.sum(~np.isnan(h_b)) < 6:
            continue

        # Filter all points that have at least one invalid param
        tc, hc, bc, wc, sc = filter_data(tc, hc, bc, wc, sc)

        # Ensure zero mean on all variables
        hc, bc, wc, sc = center(hc, bc, wc, sc)

        # Get correlations between h and waveform params
        r_bc, r_wc, r_sc = corr_coef(hc, bc, wc, sc)

        # Only proceed if corr is significant for at least one param
        r_min = 0.25                                                        #NOTE: This is important!
        if np.abs(r_bc) < r_min and np.abs(r_wc) < r_min and np.abs(r_sc) < r_min:
            continue
        
        # Calculate correction for grid cell/search radius
        h_bs, s_bc, s_wc, s_sc = get_scatt_cor(tc, hc, bc, wc, sc, proc=proc)

        if h_bs is None:
            continue

        # Apply correction to grid-cell data (if improves it)
        h_cor, h_bs = apply_scatt_cor(tc, hc, h_bs, filt=False, test_std=True)

        if (h_bs == 0).all():
            continue

        # Get percentange of variance reduction in cell
        p_new = var_reduction(hc, h_cor, h_bs)

        # Check where/if previously stored values need update
        i_update, = np.where(perc[i_cell] < p_new)

        if len(i_update) == 0:
            continue

        # Keep only improved values
        i_cell_new = [i_cell[i] for i in i_update]  # a list!
        h_bs_new = h_bs[i_update]

        # Save correction for cell (only improved values)
        perc[i_cell_new] = p_new
        hbs[i_cell_new] = h_bs_new

        # Compute centroid of cell 
        lonc = np.nanmedian(xc)
        latc = np.nanmedian(yc)

        # Store one sens. and corr. value per cell
        rbs[k] = r_bc
        rlew[k] = r_wc
        rtes[k] = r_sc
        sbs[k] = s_bc
        slew[k] = s_wc
        stes[k] = s_sc
        loni[k] = lonc
        lati[k] = latc

        # Plot individual grid cells for testing
        if 0:
            plot(xc, yc, tc, hc, bc, wc, sc, h_cor, h_bs, r_bc, r_sc, r_wc)

    """ Correct h (full dataset) with best values """

    h -= hbs

    """ Save data """

    if 1:
        print 'saving data ...'

        # Update h in the file and save cor (all cells at once)
        fi[zvar][:] = h
        fi['h_bs'] = hbs
        
        fi.flush()
        fi.close()
        
        # Save bs params as external file 
        with h5py.File(ifile.replace('.h5', '_params.h5'), 'w') as fo:
        
            fo['lon'] = loni
            fo['lat'] = lati 
            fo['r_bs'] = rbs
            fo['r_lew'] = rlew
            fo['r_tes'] = rtes
            fo['s_bs'] = sbs 
            fo['s_lew'] = slew 
            fo['s_tes'] = stes 

    """ Plot maps """

    if 0:
        # Convert into sterographic coordinates
        xi, yi = transform_coord('4326', proj, loni, lati)
        plt.scatter(xi, yi, c=rbs, s=25, cmap=plt.cm.bwr)
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

#!/usr/bin/env python
"""

Corrects radar altimetry height to correlation with waveform parameters.

Notes:
    The (back)scattering correction is applied as:

    hc_cor = h - h_bs
    
    For better speed and/or outlier rejection change: n_iter or n_sigma
    
    "lstsq(A_, h_, w=w_, n_iter=5, n_sigma=3.5)"
    
    This will be added as input in a later version
    
Example:

    scattcor.py -v lon lat h_res t_year -w bs lew tes -d 1 -r 4 -q 2 -p cen -f
        /path/to/*files.h5


Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Fernando Paolo (paolofer@jpl.nasa.gov)
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import h5py
import glob
import timeit
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from altimutils import transform_coord
from altimutils import lstsq
from altimutils import fillnans

# Max std percentage increase after correction
P_MAX = 0

# Minimum points per cell to compute solution
MIN_PTS = 35

# Default time range
TMIN, TMAX = -9999,9999

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
            choices=('det', 'dif', 'cen'), default=[None],)

    parser.add_argument(
            '-v', metavar=('lon','lat', 'h', 't','e'), dest='vnames',
            type=str, nargs=5,
            help=('name of x/y/z/t/e variables in the HDF5'),
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
            '-c', dest='fcheck', action='store_true',
            help=('discard files already processed'),
            default=False)

    return parser.parse_args()


def mad_std(x):
    return 1.4826 * np.nanmedian(np.abs(x-np.nanmedian(x)))


def center(*arrs):
    """ Remove mean from array(s). """
    return [a - np.nanmedian(a) for a in arrs]


def normalize(*arrs):
    """ Normalize array(s) by std. """
    # return [a / np.nanstd(a, ddof=1) for a in arrs]
    return [a / np.nanstd(a) for a in arrs]


def corr_coef(h, bs, lew, tes):
    """ Get corr coef between h and w/f params. """
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) &\
     ~np.isnan(lew) & ~np.isnan(tes))
    h_, bs_, lew_, tes_ = h[idx], bs[idx], lew[idx], tes[idx]
    r_bs = np.corrcoef(bs_, h_)[0,1]
    r_lew = np.corrcoef(lew_, h_)[0,1]
    r_tes = np.corrcoef(tes_, h_)[0,1]
    return r_bs, r_lew, r_tes


def corr_grad(h, bs, lew, tes, normalize=False, robust=False):
    """ Get corr gradient (slope) between h and w/f params. """
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) &\
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


def detrend(t, h, bs, lew, tes):
    """ detrend varibales """
    idx, = np.where(~np.isnan(h) & ~np.isnan(bs) &\
     ~np.isnan(lew) & ~np.isnan(tes))
    t_, h_, bs_, lew_, tes_ = t[idx], h[idx], bs[idx], lew[idx], tes[idx]

    # OLS line fit
    p_h   = np.polyfit(t_, h_, 1)
    p_bs  = np.polyfit(t_, bs_, 1)
    p_lew = np.polyfit(t_, lew_, 1)
    p_tes = np.polyfit(t_, tes_, 1)

    # Remove trend
    h   -= np.polyval(p_h,t)
    bs  -= np.polyval(p_bs,t)
    lew -= np.polyval(p_lew,t)
    tes -= np.polyval(p_tes,t)

    return h, bs, lew, tes

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
    idx = tree.query_ball_point((x0, y0), r)

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
        idx = tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_reloc == k+1:
            break

    return idx


def get_scatt_cor(t, h, e, bs, lew, tes, tmin, tmax):
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
        e: rms error of h
        bs: backscatter coefficient
        lew: leading-edge width
        tes: trailing-edge slope

    """

    # Remove the temporal trend
    #h, bs, lew, tes = detrend(t, h, bs, lew, tes)

    # copy entries
    h_   = h.copy()
    e_   = e.copy()
    bs_  = bs.copy()
    lew_ = lew.copy()
    tes_ = tes.copy()

    # Add something small to the errors to avoid singularity
    e_ += 0.01

    # Construct weight
    w_ = 1/e_**2

    # Select time spans to compute correction
    i_t = (t >= tmin) & (t <= tmax)

    # Set observations outside range to nan's
    h_[~i_t] = np.nan

    # Construct design matrix:
    A  = np.vstack((bs, lew, tes)).T
    A_ = np.vstack((bs_, lew_, tes_)).T

    # Check for division by zero
    try:

        # Estimate model parameters
        p, ep, i_bad = lstsq(A_, h_, w=w_, n_iter=5, n_sigma=3.5)

        # Linear combination of time-series => h_bs = a Bs + b LeW + c TeS
        h_wc = np.dot(A, p)

        # Compute model residuals
        res = h - h_wc

        # Compute sum of squares and total sum of squares
        RSS = np.nansum(res**2)
        TSS = np.nansum((h - np.nanmean(h))**2)

        # Get r-squared -> model performance metric
        r2 = 1.0 - RSS / TSS

    except:

        # Make sure we now if it works or not
        ('MULTIVARIATE FIT FAILED, setting h_bs -> zeros')

        # Set all params to zero if exception detecte
        h_wc = np.zeros_like(h)
        p = [np.nan, np.nan, np.nan]
        r2 = np.nan
        i_bad =  np.ones(h.shape, dtype=bool)

    return h_wc, p, r2, i_bad


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


def main(ifile, vnames, wnames, dxy, proj, radius=0, n_reloc=0, proc=None):

    import warnings
    warnings.filterwarnings("ignore")

    print('processing file:', ifile, '...')

    # Test if parameter file exists
    if '_SCATGRD' in ifile.lower():
        return

    # Get variable names
    xvar, yvar, zvar, tvar, evar = vnames
    bpar, wpar, spar = wnames

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:

        # Check if we can read file
        try:
            t = fi[tvar][:]
            h = fi[zvar][:]
            lon = fi[xvar][:]
            lat = fi[yvar][:]
            bs = fi[bpar][:] if bpar in fi else np.zeros(lon.shape)
            lew = fi[wpar][:] if wpar in fi else np.zeros(lon.shape)
            tes = fi[spar][:] if spar in fi else np.zeros(lon.shape)
            rms = fi[evar][:] if evar in fi else np.ones(lon.shape)
            rms[rms==999999] = np.nanmean(rms[rms!=999999]) + 10
        except:

            # Otherwise run next
            print('Could not open file:', ifile)
            return

    # Convert into sterographic coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Get nodes of solution grid
    x_nodes, y_nodes = get_grid_nodes(x, y, dxy, proj=proj)

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

    # Create KD-tree
    tree = cKDTree(list(zip(x, y)))

    # Time is not provided compute the range of the data
    if (tmin_ == -9999) or (tmax_ == 9999):
        tmin, tmax = np.nanmin(t), np.nanmax(t)
    else:
        tmin, tmax = tmin_, tmax_

    # Find values that have no solution
    i_bs  = np.isnan(bs)  | (bs  == 0)
    i_lew = np.isnan(lew) | (lew == 0)
    i_tes = np.isnan(tes) | (tes == 0)

    # Interpolate and fill empties using nearest data point
    bs[i_bs]   = griddata((t[~i_bs], x[~i_bs], y[~i_bs]), bs[~i_bs],\
                (t[i_bs], x[i_bs], y[i_bs]),    'nearest')
    lew[i_lew] = griddata((t[~i_lew],x[~i_lew], y[~i_lew]), lew[~i_lew],\
                (t[i_lew], x[i_lew], y[i_lew]), 'nearest')
    tes[i_tes] = griddata((t[~i_tes],x[~i_tes],y[~i_tes]), tes[~i_tes],\
                (t[i_tes], x[i_tes], y[i_tes]), 'nearest')

    # Loop through nodes
    for k in range(N_nodes):

        # Get current nodes
        xi, yi = x_nodes[k], y_nodes[k]

        # Get indices of data within search radius
        i_cell = get_radius_idx(x, y, xi, yi, radius, tree, n_reloc=n_reloc)

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
        ec = rms[i_cell]
        
        # Keep original (unfiltered) data
        tc_orig, hc_orig = tc.copy(), hc.copy()

        # Test minimum number of obs in all params
        nobs = min([len(v[~np.isnan(v)]) for v in [hc, bc, wc, sc]])

        # Test for enough points
        if (nobs < MIN_PTS): continue

        # Ensure zero mean on all variables
        hc, bc, wc, sc = center(hc, bc, wc, sc)

        # Normalize the w/f params to std = 1
        bc, wc, sc = normalize(bc, wc, sc)

        # Add something small for stability
        bc += 1e-4
        sc += 1e-4
        wc += 1e-4

        # Calculate correction for data in search radius
        hc_wc, p_fit, r2, i_bad = get_scatt_cor(tc, hc, ec, bc, wc, sc, tmin, tmax)

        # Make sure this works
        if np.any(np.isnan(p_fit)):
            continue

        # Sensitivity gradients from model
        b_bc, b_wc, b_sc = p_fit

        # Set outliers to NaN values
        hc[i_bad] = np.nan
        bc[i_bad] = np.nan
        wc[i_bad] = np.nan
        sc[i_bad] = np.nan
        tc[i_bad] = np.nan

        # Reject if not enough points
        if len(hc[~np.isnan(hc)]) < MIN_PTS: continue

        # If no correction could be generated, skip
        if (hc_wc == 0).all(): continue

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

        # Check if std.dev has increase if so set to zero
        if p_std > P_MAX :

            # Cor is set to zero
            hc_cor = hc.copy()
            hc_wc[:] = 0.

            # All params are set to zero/one
            b_bc, b_wc, b_sc = 0., 0., 0.
            r_bc, r_wc, r_sc = 0., 0., 0.
            s_bc, s_wc, s_sc = 0., 0., 0.
            r2 = 0.
            d_std, p_std, d_trend, p_trend = 0., 0., 0., 0.

        # Set filtered out values (not used in the calculation) to NaN
        hc_wc[np.isnan(hc)] = np.nan

        # Check if previously stored values need update (r2_prev < r2_new)
        i_update, = np.where(r2fit[i_cell] <= r2)

        # Only keep the indices/values that need update
        i_cell_new = [i_cell[i] for i in i_update]
        hc_wc_new = hc_wc[i_update]

        # Store correction for cell (only improved values)
        hbs[i_cell_new] = hc_wc_new
        r2fit[i_cell_new] = r2
        dstd[i_cell_new] = d_std
        pstd[i_cell_new] = p_std
        dtrend[i_cell_new] = d_trend
        ptrend[i_cell_new] = p_trend
        rbs[i_cell_new] = r_bc
        rlew[i_cell_new] = r_wc
        rtes[i_cell_new] = r_sc
        sbs[i_cell_new] = s_bc
        slew[i_cell_new] = s_wc
        stes[i_cell_new] = s_sc
        bbs[i_cell_new] = b_bc
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

        # Time span of data inside solution area
        tspan = np.nanmax(tc) - np.nanmin(tc)

        if (k % 1) == 0:
            print('Node', k, 'of', N_nodes, \
                    'Trend:', np.around(d_trend,3), 'Std.dev:', \
                    np.around(d_std,3),'Time-Span:',np.around(tspan,3),
                    'Nobs:',len(hc[~np.isnan(hc)]),'Corr:',np.array([r_bc, r_wc, r_sc]).round(2))

    # Find values that have no solution
    i_ip = np.isnan(hbs) | (hbs == 0)

    print('-> Interpolating correction to empty data')
    try:

        # Interpolate using nearest neigbour
        hbs[i_ip] = griddata((x[~i_ip], y[~i_ip]), hbs[~i_ip],
        (x[i_ip], y[i_ip]), 'nearest')

    except:

        # If it does not work
        print('-> Interpolation not sucessful')
        pass

    """ Save data """

    print('saving data ...')

    with h5py.File(ifile, 'a') as fi:

        # Update h in the file and save correction (all cells at once)
        fi[zvar][:] = h

        # Try to create varibales
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

        # Update variabels instead
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

        # Only rename file if _SCAT has not been added
        if ifile.find('_SCAT.h5') < 0:

            # Rename file
            os.rename(ifile, ifile.replace('.h5', '_SCAT.h5'))

        # Compute statistics
        hbs_cm = hbs[(hbs != 0) & ~np.isnan(hbs)]
        hbs_mean = np.median(hbs_cm) * 100
        hbs_stdv = mad_std(hbs_cm)   * 100
        n_before = np.around(len(hbs[i_ip])/float(len(hbs))*100)
        n_afters = np.around(len(hbs[np.isnan(hbs)| (hbs==0)])/\
        float(len(hbs))*100)

        # Some output statistics
        print('Empty before interp (%):', n_before)
        print('Empty after interp  (%):', n_afters)
        print('Average magnitude of correction:', np.around(hbs_mean,0),'+-', \
        np.around(hbs_stdv,0),'(cm)')

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
                print('COUND NOT SAVE PARAMETERS FOR EACH CELL')
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
    tmin_ = args.tlim[0]           # min time in decimal years
    tmax_ = args.tlim[1]           # max time in decimal years
    check = args.fcheck          # check for already processed tiles

    # Check if files already have been processed
    if check: ifiles = fcheck(ifiles)

    print('parameters:')
    for arg in vars(args).items():
        print(arg)

    if njobs == 1:
        print('running sequential code ...')
        [main(ifile, vnames, wnames, dxy, proj, radius, nreloc, proc) \
                for ifile in ifiles]

    else:
        print('running parallel code (%d jobs) ...' % njobs)
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("loky", inner_max_num_threads=1):
            Parallel(n_jobs=njobs,)(
            delayed(main)(ifile, vnames, wnames, dxy, proj, radius, \
            nreloc, proc) for ifile in ifiles)

    print('done!')

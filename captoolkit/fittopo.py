#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""

Surface topography detrending of satellite and airborne altimetry

Program computes surface elevation residuals, containing only the temporal
component, by removing the static topography.

Depending on the number of observations in each solution one of three models
are used to solve for the topography (1) Bi-quadratic, (2) Bilinear and (3)
the average.

User specifies a grid resolution, search radius and the number of
relocations that should be used to detrend the observations. Inside each
search area the model is centered (relocated) to the centroid of the data,
given the provided number of allowed relocations.

Given the possible overlap between solutions the solution with the smallest
RMS is used and data of poorer quality overwritten.

An a-priori DEM can be provided to perform initial detrending of all
observations. fittopo.py then removes any residual topography where a valid
local solution can be obtained. Where the local fit fails, the DEM-detrended
height residual is retained. DEM must have the same projection as provided by
the "-j" option.

Notes:
    For mission in reference track configuration a dx = dy = 250 m and a
    search radius of 350 m is appropriate, and less than n=3 relocations is
    usually needed to center the data (depends on search radius)

    This program can be run in parallel to processes several files at the same
    time (tiles or missions etc).

    Good threshold ("-m" option) for switching from biquadratic to bilinear
    model is around 10-15 points.

Example:

    python fittopo.py /path/to/files/*.h5 -v lon lat t_year h_cor \
            -d 1 1 -r 1 -q 3 -i 5 -z 5 -m 15 -k 1 -t 2012 -j 3031 -n 2

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Johan Nilsson   (johan.n.nilsson@geo.uu.se)
    Fernando Paolo  (fernando@globalfishingwatch.org)
    Alex Gardner    (alex.s.gardner@jpl.nasa.gov)

    Department of Earth Sciences, Uppsala University
    Jet Propulsion Laboratory, California Institute of Technology

"""

import warnings
warnings.filterwarnings("ignore")
import os
import h5py
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from altimutils import tiffread
from altimutils import interp2d
from altimutils import make_grid
from altimutils import transform_coord
from altimutils import mad_std
from altimutils import binning
from altimutils import median_filter
from altimutils import lstsq

DXY = [1, 1]
RADIUS = [1, 0.5]
MINOBS = 3
NITER = 5
TREF = None
TLIM = 0
PROJ = 3031
COLS = ['lon', 'lat', 't_year', 'h_elv', 'm_idx']
EXPR = None
ORDER = 2
MLIM = 10
NJOBS = 1
NSIGMA, THRES = [None, None]
DEM = [None]

description = ('Compute surface elevation residuals '
               'from satellite/airborne altimetry.')

parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or km)'),
        default=DXY,)

parser.add_argument(
        '-r', metavar=('rmax','rcor'), dest='radius', type=float, nargs=2,
        help=('max search radius and corr. length (km)'),
        default=RADIUS,)

parser.add_argument(
        '-q', metavar=('n_reloc'), dest='nreloc', type=int, nargs=1,
        help=('number of relocations for search radius'),
        default=[0],)

parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations for model solution'),
        default=[NITER],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-m', metavar=('mod_lim'), dest='mlim', type=int, nargs=1,
        help=('minimum obs for higher order models'),
        default=[MLIM],)

parser.add_argument(
        '-k', metavar=('mod_order'), dest='order', type=int, nargs=1,
        help=('order of the surface fit model: 1=lin or 2=quad'),
        default=[ORDER],)

parser.add_argument(
        '-t', metavar=('ref_time'), dest='tref', type=str, nargs=1,
        help=('ref. time for fit (default mean of solution)'),
        default=[TREF],)

parser.add_argument(
        '-l', metavar=('tlim'), dest='tlim', type=float, nargs=1,
        help=('min time span of solution (e.g reject single orbits)'),
        default=[TLIM],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','m'), dest='vnames', type=str, nargs=5,
        help=('name of lon/lat/t/h/m in the HDF5'),
        default=COLS,)

parser.add_argument(
        '-x', metavar=('expr'), dest='expr', type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[EXPR],)

parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[NJOBS],)

parser.add_argument(
        '-e', metavar=('nsigma','thres'), dest='filter', type=float, nargs=2,
        help="Number of std.dev and cutoff values to filter data",
        default=[NSIGMA, THRES],)

parser.add_argument(
        '-s', metavar=('fdem'), dest='fdem', type=str, nargs=1,
        help="DEM file for centering data - proj as in '-j' (.tif)",
        default=[None],)

parser.add_argument(
        '-w', metavar=('fmsk','ikeep'), dest='fmsk', type=str, nargs=2,
        help="mask for on(1)/off(0) ROI - proj as in '-j' (.tif)",
        default=[None,None])

parser.add_argument(
        '-a', dest='edit', action='store_true',
        help=('remove identified outliers before saving'),
        default=False)

parser.add_argument(
        '-p', dest='pshow', action='store_true',
        help=('verbose mode - print statsistics'),
        default=False)

args = parser.parse_args()

files  = args.files
dx     = args.dxy[0] * 1e3
dy     = args.dxy[1] * 1e3
dmax   = args.radius[0] * 1e3
dcor   = args.radius[1] * 1e3
nreloc = args.nreloc[0]
nlim   = args.minobs[0]
nmod   = args.mlim[0]
niter  = args.niter[0]
tref_  = args.tref[0]
proj   = args.proj[0]
icol   = args.vnames[:]
expr   = args.expr[0]
njobs  = args.njobs[0]
order  = args.order[0]
diag   = args.pshow
nsigma = args.filter[0]
thres  = args.filter[1]
dtlim  = args.tlim[0]
fdem   = args.fdem[0]
fmsk   = args.fmsk[0]
keep   = args.fmsk[1]
edit   = args.edit

print('parameters:')
for p in list(vars(args).items()):
    print(p)

def get_radius_idx(x, y, x0, y0, r, Tree, n_reloc=0,
        min_months=24, max_reloc=3, time=None, height=None):
    """ Get indices of all data points inside radius. """

    idx = Tree.query_ball_point((x0, y0), r)

    if len(idx) < 2:
        return idx

    if time is not None:
        n_reloc = max_reloc

    if n_reloc < 1:
        return idx

    for k in range(n_reloc):

        x0_new, y0_new = np.median(x[idx]), np.median(y[idx])

        reloc_dist = np.hypot(x0_new-x0, y0_new-y0)

        if reloc_dist > r:
            break

        idx = Tree.query_ball_point((x0_new, y0_new), r)

        if n_reloc == k+1:
            break

        if time is not None:

            t_b, x_b = binning(
                time[idx], height[idx], dx=1/12., window=1/12.)[:2]

            print(('months #:', np.sum(~np.isnan(x_b))))

            if np.sum(~np.isnan(x_b)) >= min_months:
                break

    return idx


def main(ifile, n=''):

    set_use = 1

    import warnings
    warnings.filterwarnings("ignore")

    if os.stat(ifile).st_size == 0:
        print('-> Input file is empty!')
        return

    startTime = datetime.now()

    print('-> Loading data ...')

    if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        print("-> Input file must be in hdf5-format")
        return

    xvar, yvar, tvar, zvar, bvar = icol

    with h5py.File(ifile, 'r') as fi:

        lon    = fi[xvar][:]
        lat    = fi[yvar][:]
        time   = fi[tvar][:]
        height = fi[zvar][:]
        bias   = fi[bvar][:] if bvar in fi else np.zeros(lon.shape)

    print('-> Converting lon/lat to x/y ...')

    (x, y) = transform_coord('4326', proj, lon, lat)

    if fmsk is not None:

        Xm, Ym, Zm = tiffread(fmsk)[0:3]

        imsk = interp2d(Xm, Ym, Zm, x, y, order=0)

        imsk = imsk == int(keep)

        x      = x[imsk]
        y      = y[imsk]
        lon    = lon[imsk]
        lat    = lat[imsk]
        time   = time[imsk]
        height = height[imsk]
        bias   = bias[imsk]

    (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    if expr:
        time = eval(expr.replace('t', 'time'))

    t_mean = np.round(np.nanmean(time), 2)

    (Xi, Yi) = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    xi = Xi.ravel()
    yi = Yi.ravel()

    # Initial DEM detrending at every observation. This provides a fallback
    # residual even where the local topographic fit cannot be computed.
    if fdem is not None:

        print('-> Detrending observations with DEM ...')

        Xd, Yd, Zd = tiffread(fdem)[0:3]

        # DEM sampled at the actual observation locations.
        hdem = interp2d(Xd, Yd, Zd, x, y, order=1)

        # First-pass residual available for every observation covered by DEM.
        height_fit = height - hdem

    else:

        hdem = np.full(height.shape, np.nan)
        height_fit = height.copy()

    print('-> Building the KD-tree ...')
    Tree = cKDTree(np.c_[x, y], copy_data=False)

    # DEM residuals are the fallback output where a DEM is supplied.
    if fdem is not None:
        dh_topo = height_fit.copy()
        mi_topo = np.zeros(height.shape)
        hm_topo = hdem.copy()
    else:
        dh_topo = np.full(height.shape, np.nan)
        mi_topo = np.full(height.shape, np.nan)
        hm_topo = np.full(height.shape, np.nan)

    # Saved fit error. DEM-only observations remain NaN because no local-fit
    # RMS exists for them.
    de_topo = np.full(height.shape, np.nan)

    # Internal comparison metric for overlapping solutions.
    best_rms = np.full(height.shape, np.inf)

    sx_topo = np.full(height.shape, np.nan)
    sy_topo = np.full(height.shape, np.nan)
    tr_topo = np.full(height.shape, np.nan)

    print('-> Predicting values ...')

    for i in range(len(xi)):

        idx = get_radius_idx(
            x, y, xi[i], yi[i], dmax, Tree, n_reloc=nreloc)

        nobs = len(time[idx])

        if nobs < nlim:
            continue

        xcap = x[idx]
        ycap = y[idx]
        tcap = time[idx]

        # Fit h-DEM in DEM mode; otherwise preserve the original elevation
        # fit and temporal-reference behaviour.
        hcap = height_fit[idx]
        bvar = bias[idx]

        x0 = np.median(xcap)
        y0 = np.median(ycap)
        horg = hcap.copy()

        xc = np.median(xcap)
        yc = np.median(ycap)

        c0 = np.ones(len(xcap))
        c1 = xcap - xc
        c2 = ycap - yc
        c3 = c1 * c2
        c4 = c1 * c1
        c5 = c2 * c2
        c7 = bvar

        nb = len(hcap)

        if fdem is not None:

            # External-DEM mode: no temporal rate is fitted. The intercept
            # absorbs the local mean h-DEM offset but is NOT removed from dh.
            tref = np.nan
            Acap = np.vstack((c0, c1, c2, c7)).T
            mi = 2

            if nobs > nmod and order > 1:
                Acap = np.vstack(
                    (c0, c1, c2, c3, c4, c5, c7)).T
                mi = 1

        else:

            # Original no-DEM behaviour, including the temporal rate term.
            tmin, tmax = tcap.min(), tcap.max()
            if (tmax - tmin) < dtlim:
                continue

            if tref_ is not None:
                tref = float(tref_)
                nbef = len(tcap[tcap < tref])
                naft = len(tcap[tcap > tref])
                if nbef == 0 or naft == 0:
                    continue
            else:
                tref = 0
                set_use = 0

            c6 = (tcap - tref) * set_use
            Acap = np.vstack((c6, c0, c1, c2, c7)).T
            mi = 2

            if nobs > nmod and order > 1:
                Acap = np.vstack(
                    (c6, c0, c1, c2, c3, c4, c5, c7)).T
                mi = 1

        if dcor > 0:

            dr = np.sqrt((xcap - x0)**2 + (ycap - y0)**2)

            wcap = 1.0 / (1.0 + (dr / dcor)**2)

        else:

            wcap = None

        x_hat, e_hat, i_bad = lstsq(
            Acap, hcap, w=wcap, n_iter=niter,
            n_sigma=nsigma, ylim=thres)

        if fdem is not None:

            # Remove only residual spatial topography and mission offset.
            # The fitted intercept x_hat[0] is deliberately retained in dh.
            if mi == 1:
                hmod = (c1 * x_hat[1] + c2 * x_hat[2] +
                        c3 * x_hat[3] + c4 * x_hat[4] +
                        c5 * x_hat[5] + c7 * x_hat[6])
            else:
                hmod = (c1 * x_hat[1] + c2 * x_hat[2] +
                        c7 * x_hat[3])

            sx, sy = x_hat[1], x_hat[2]

            # _mod remains tied to the external DEM in DEM mode.
            h0 = interp2d(
                Xd, Yd, Zd, np.array([xc]), np.array([yc]), order=1)[0]

        else:

            # Original no-DEM behaviour: remove all fitted non-rate terms.
            hmod = np.dot(Acap[:,1:], x_hat[1:])
            sx, sy = x_hat[2], x_hat[3]
            h0 = x_hat[1]

        slope = np.arctan(np.sqrt(sx**2 + sy**2)) * (180 / np.pi)

        dh = horg - hmod

        na = len(dh)

        RMSE = mad_std(dh)

        if nsigma is not None and edit is True:
            dh[np.abs(dh) > nsigma * mad_std(dh)] = np.nan

        if thres is not None and edit is True:
            dh[np.abs(dh) > thres] = np.nan
            RMSE = mad_std(dh)
            if np.isnan(RMSE):
                continue
            if RMSE > thres:
                continue

        # Only a valid local fit competes with other local fits.
        iup = RMSE < best_rms[idx]

        dh_cap = dh_topo[idx].copy()
        de_cap = de_topo[idx].copy()
        br_cap = best_rms[idx].copy()
        hm_cap = hm_topo[idx].copy()
        mi_cap = mi_topo[idx].copy()
        tr_cap = tr_topo[idx].copy()
        sx_cap = sx_topo[idx].copy()
        sy_cap = sy_topo[idx].copy()

        dh_cap[iup] = dh[iup]
        de_cap[iup] = RMSE
        br_cap[iup] = RMSE
        hm_cap[iup] = h0
        mi_cap[iup] = mi
        tr_cap[iup] = tref
        sx_cap[iup] = np.arctan(sx) * (180 / np.pi)
        sy_cap[iup] = np.arctan(sy) * (180 / np.pi)

        dh_topo[idx] = dh_cap
        de_topo[idx] = de_cap
        best_rms[idx] = br_cap
        hm_topo[idx] = hm_cap
        mi_topo[idx] = mi_cap
        tr_topo[idx] = tr_cap
        sx_topo[idx] = sx_cap
        sy_topo[idx] = sy_cap

        if (i % 100) == 0 and diag is True:
            print(('%s %i %s %2i %s %i %s %03d %s %.3f %s %.3f' % \
                    ('#',i,'/',len(xi),'Model:',mi,'Nobs:',nb,'Slope:',\
                    np.around(slope,3),'Residual:',np.around(RMSE,3))))

    print(('Total NaNs (percent): %.2f' % \
            (100 * float(len(dh_topo[np.isnan(dh_topo)])) /\
             float(len(dh_topo)))))

    zero = np.sum(mi_topo == 0)
    one = np.sum(mi_topo == 1)
    two = np.sum(mi_topo == 2)
    tre = np.sum(mi_topo == 3)

    N = float(len(mi_topo))

    if fdem is not None:
        print(('Model types (percent): DEM = %.2f, 1 = %.2f, '
               '2 = %.2f, 3 = %.2f' %
               (100 * zero/N, 100 * one/N, 100 * two/N, 100 * tre/N)))
    else:
        print(('Model types (percent): 1 = %.2f, 2 = %.2f, 3 = %.2f' % \
               (100 * one/N, 100 * two/N, 100 * tre/N)))

    with h5py.File(ifile, 'a') as fi:

        try:

            fi[zvar+'_res'] = dh_topo
            fi[zvar+'_mod'] = hm_topo
            fi[zvar+'_err_res'] = de_topo
            fi[zvar+'_m_deg'] = mi_topo
            fi[zvar+'_t_ref'] = tr_topo
            fi[zvar+'_slp_x'] = sx_topo
            fi[zvar+'_slp_y'] = sy_topo

        except:

            fi[zvar+'_res'][:] = dh_topo
            fi[zvar+'_mod'][:] = hm_topo
            fi[zvar+'_err_res'][:] = de_topo
            fi[zvar+'_m_deg'][:] = mi_topo
            fi[zvar+'_t_ref'][:] = tr_topo
            fi[zvar+'_slp_x'][:] = sx_topo
            fi[zvar+'_slp_y'][:] = sy_topo

    if ifile.find('TOPO') < 0:
        os.rename(ifile, ifile.replace('.h5', '_TOPO.h5'))

    print(('*' * 75))
    print(('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' % \
        ('Statistics',
         'Mean:',    np.nanmedian(dh_topo),
         'Std.dev:', mad_std(dh_topo),
         'Min:',     np.nanmin(dh_topo),
         'Max:',     np.nanmax(dh_topo),
         'RMSE:',    np.nanmedian(de_topo),)))
    print(('*' * 75))
    print('')

    print(('Execution time: '+ str(datetime.now()-startTime)))

if njobs == 1:
    print('running sequential code ...')
    [main(f, n) for n,f in enumerate(files)]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(f, n) for n, f in enumerate(files))


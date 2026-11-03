#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Surface elevation changes and associated time series from satellite and
airborne altimetry.

Program computes surface elevation changes, time series and seasonal parameters
from either space or airborne altimetry. It also has the capability of merging
elevation data from two different sources. This is similar to fittopo.py but
allows for more generic use by combining "fittopo.py" and "corrscat.py".

The input of the software is very similar to "fittopo.py" with the choice of
number of relocations, spatial correlation lengths and local data editing.
The software provides several different options to the user to process the data
in the form of several surface models (bilinear and biquadratic), polynomial
orders (trend and acceleration) and estimation of seasonal parameters
(amplitude and phase). It further allows for estimation of data offsets which
are used to cross-calibrate datasets or solve for ascending/descending biases.

If radar data is used the user can provide waveform parameters (max three) and
activate the waveform correction in the model setup to reduce the effects of
changes in scattering regime on the time series and trends in the solution.
The user can provide a maximum of three or just use one if needed. This is
controlled by the -v option. If scattering correction is enabled, NaNs or other
non-finite values in available waveform parameters are interpolated using the
nearest valid values in time and space before constructing the model.

The optional -binwf flag bins available waveform parameters (bs, lew and tes)
using the same temporal binning as the surface elevation time series. This
option is independent of the scattering correction. If scattering correction
is disabled, waveform parameters are not interpolated and only finite waveform
observations are used for binning. Only waveform parameters that actually exist
in the input HDF5 file are saved. The output variables are named bs(t), lew(t)
and tes(t).

The software allows for other data sources to be merged if called for and two
inputs are needed: (1) an input variable (m_idx) classifying each mission with
an integer (0 to n - where 0 is the reference data). These vectors will be added
as dummy variables (0/1) to the design matrix and offsets solved for and removed.
The user needs to activate this option in the model setup (-m).

Further the software also allows for two datasets to be differenced given 0 or 1
index in m_idx (CS2=1 and IS2=0) using combined '-m' and '-dif' option. Point
differences are computed by interpolating data "1" to data "0" using their
timestamps. Estimated offset bias between "0" and "1" is then added back and the
data is binned given user defined resolution and saved in sec(t).

Output of the program is a file containing the following variables

    *** Static Parameters: ****
    lon, lat = longitude and latitude
    p0, p1, p2 = intercept, trend and acceleration
    p0_error, p1_error, p2_error = corresponding standard errors
    amplitude, phase = seasonal amplitude and phase
    rmse = rmse of residuals
    nobs = number of data points in solution
    dmin = distance to closest point in solution
    tspan = time span of data in each solution
    offset = local offsets between two datasets

    *** Time Variable Parameters (n x t): ***
    sec(t), rms(t) = elevation change and associated errors
    time = time vector

    Optional with -binwf:
    bs(t) = binned backscatter
    lew(t) = binned leading edge width
    tes(t) = binned trailing edge slope

    *** Model order format: -m t p s w b ***
    t = surface order       (x,y)    0 to 2
    p = polynomial order    (t)      0 to 3
    s = seasonal            (on/off) 0 or 1
    w = waveform corr       (on/off) 0 or 1
    b = offset/bias         (on/off) 0 or 1

    *** Explanation ***
    t: surface order 0 = mean, 1 = bilinear and 2 = biquadratic
    p: polynomial order sum(ci*t**i)
    s: estimate annual seasonal a*cos(wt) + b*sin(wt): 0=OFF and 1=ON
    w: apply scattering correction (bs,lew,tes): 0=OFF and 1=ON
    b: mission index array 0 - N (0,1,2,...) "0" is considered reference

Example:

    python fitsec.py file.h5 -d 1 1 -r 1 -q 1 -i 5 -z 10 -t 2010.5 2022 \
        -f 2020 -l 10 -k 1 -w 10 10 10 -j 3031 -v \
        lon lat t_year h_cor h_rms bs lew tes m_idx \
        -n 1 -m 2 1 1 1 1 -s 0.0833 0.085 -binwf

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
import sys
import h5py
import pyproj
import argparse
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata
from altimutils import transform_coord
from altimutils import make_grid
from altimutils import lstsq
from altimutils import mad_std
from altimutils import tiffread
from altimutils import interp2d


# Output description of solution
description = ('Computes surface-elevation change and time series '
               'from satellite/airborne altimetry.')


# =============================================================================
# Command-line arguments
# =============================================================================

parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    'files', metavar='file', type=str, nargs='+',
    help='file(s) to process (HDF5)')

parser.add_argument(
    '-o', metavar='outfile', dest='ofile', type=str, nargs=1,
    help='output file name, default same as input',
    default=[None])

parser.add_argument(
    '-b', metavar=('w', 'e', 's', 'n'), dest='bbox', type=float, nargs=4,
    help='bounding box for geographic region (deg or m)',
    default=[None, None, None, None])

parser.add_argument(
    '-d', metavar=('dx', 'dy'), dest='dxy', type=float, nargs=2,
    help='spatial resolution of grid (km)',
    default=[1, 1])

parser.add_argument(
    '-r', metavar='radius', dest='radius', type=float, nargs=1,
    help='search radius for data (km)',
    default=[1])

parser.add_argument(
    '-c', metavar=('rcor', 'tcor'), dest='corr', type=float, nargs=2,
    help='spatio-temporal correlations (km/dyr)',
    default=[None, None])

parser.add_argument(
    '-q', metavar='nrel', dest='nrel', type=int, nargs=1,
    help='number of relocations for search radius',
    default=[0])

parser.add_argument(
    '-i', metavar='niter', dest='niter', type=int, nargs=1,
    help='max number of iterations for least-squares solution',
    default=[1])

parser.add_argument(
    '-z', metavar='zmin', dest='zmin', type=int, nargs=1,
    help='min data to compute solution',
    default=[1])

parser.add_argument(
    '-t', metavar=('tmin', 'tmax'), dest='tspan', type=float, nargs=2,
    help='min/max time for solutions and time-series',
    default=[False, False])

parser.add_argument(
    '-f', metavar='tref', dest='tref', type=float, nargs=1,
    help='reference time for solution',
    default=[None])

parser.add_argument(
    '-l', metavar='ratelim', dest='ratelim', type=float, nargs=1,
    help='discard if |dh/dt| > ratelim',
    default=[1e6])

parser.add_argument(
    '-k', metavar='dtlim', dest='dtlim', type=float, nargs=1,
    help='discard if tspan < dtlim (yrs)',
    default=[0])

parser.add_argument(
    '-w', metavar=('nsig', 'rlim', 'alim'), dest='thres', type=float, nargs=3,
    help='nsig and reject if |res| > rlim or |value| > alim (m)',
    default=[None, None, None])

parser.add_argument(
    '-s', metavar=('tstep', 'tres'), dest='tsteps', type=float, nargs=2,
    help='time and window resolution for time series',
    default=[1./12, 1./12])

parser.add_argument(
    '-j', metavar='proj', dest='proj', type=str, nargs=1,
    help='projection: EPSG (AnIS=3031, GrIS=3413)',
    default=['4326'])

parser.add_argument(
    '-v', metavar=('x', 'y', 't', 'h', 's', 'b', 'l', 'e', 'b'),
    dest='vnames', type=str, nargs=9,
    help='names of needed variables',
    default=['lon', 'lat', 't_year', 'h_elv', 'h_rms',
             'bs', 'lew', 'tes', 'bias'])

parser.add_argument(
    '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
    help='for parallel processing of multiple files',
    default=[1])

parser.add_argument(
    '-m', metavar='', dest='model', type=int, nargs=5,
    help='models selection: see notes in file',
    default=[[0, 0, 0, 0, 0]])

parser.add_argument(
    '-dem', metavar='fdem', dest='fdem', type=str, nargs=1,
    help="name of DEM file for detrending - needs '-j' proj (.tif)",
    default=[None])

parser.add_argument(
    '-mask', metavar=('fmsk', 'keep'), dest='fmsk', type=str, nargs=2,
    help="mask for on(1)/off(0) ROI - needs '-j' proj (.tif)",
    default=[None, None])

parser.add_argument(
    '-dif', dest='diff', action='store_true',
    help='compute difference of time series before binning',
    default=False)

parser.add_argument(
    '-binwf', dest='binwf', action='store_true',
    help='bin available waveform parameters (bs, lew, tes) to sec(t) time bins',
    default=False)


# =============================================================================
# Functions
# =============================================================================

def make_time(tmin, tmax):
    """Make monthly vector."""

    import datetime
    from astropy.time import Time

    mm = np.arange(1, 12 + 1, 1)
    yy = np.arange(tmin, tmax, 1)

    time = []

    for y in yy:
        for m in mm:
            date = datetime.datetime(int(y), int(m), 15, 0, 0, 0)
            date = Time(date, format='datetime')
            time.append(date.decimalyear)

    return np.asarray(time)


def get_radius_idx(x, y, x0, y0, r, tree, n_rel=0):
    """Get indices of all data points inside radius."""

    idx = tree.query_ball_point((x0, y0), r)
    reloc_dist = 0.

    if n_rel < 1 or len(idx) < 2:
        return idx, reloc_dist

    for k in range(n_rel):

        x0_new = np.median(x[idx])
        y0_new = np.median(y[idx])

        reloc_dist = np.hypot(x0_new - x0, y0_new - y0)

        if reloc_dist > r:
            break

        idx = tree.query_ball_point((x0_new, y0_new), r)

        if n_rel == k + 1:
            break

    return idx, reloc_dist


def model_order(order, dt, bo, pxy, wf):
    """Set model parameters and return design matrix A."""

    t, p, s, w, b = order

    # Topography terms
    if t == 0:
        ct0, ct1 = 0, 0
    elif t == 1:
        ct0, ct1 = 1, 0
    elif t == 2:
        ct0, ct1 = 1, 1
    else:
        raise ValueError("Topography order (t) must be 0-2")

    # Polynomial terms
    cp0 = cp1 = cp2 = 0

    if p == 1:
        cp0 = 1
    elif p == 2:
        cp0 = cp1 = 1
    elif p == 3:
        cp0 = cp1 = cp2 = 1
    elif p != 0:
        raise ValueError("Time polynomial order (p) must be 0-3")

    cs0 = 1 if s == 1 else 0
    cw0 = 1 if w == 1 else 0

    dx, dy = pxy
    dbsc, dlew, dtes = wf

    # Center waveform parameters only when scattering correction is enabled
    if w > 0:
        dbsc = np.asarray(dbsc) - np.nanmedian(dbsc)
        dlew = np.asarray(dlew) - np.nanmedian(dlew)
        dtes = np.asarray(dtes) - np.nanmedian(dtes)
    else:
        dbsc[:] = 0
        dlew[:] = 0
        dtes[:] = 0

    A = np.vstack((
        np.ones_like(dx),
        cp0 * dt,
        0.5 * cp1 * dt**2,
        (1.0 / 6.0) * cp2 * dt**3,
        cs0 * np.cos(2.0 * np.pi * dt),
        cs0 * np.sin(2.0 * np.pi * dt),
        ct0 * dx,
        ct0 * dy,
        ct0 * dx * dy,
        ct1 * dx**2,
        ct1 * dy**2,
        cw0 * dbsc,
        cw0 * dlew,
        cw0 * dtes
    )).T

    if b == 1:

        mi = np.unique(bo)

        if len(mi) == 1:
            return A

        for i in range(len(mi)):

            if mi[i] == 0:
                continue

            dummy = np.zeros((len(bo), 1))
            dummy[bo == mi[i]] = 1.0

            A = np.hstack((A, dummy))

    return A


@jit(nopython=True)
def resample(x, y, w, xi, dx=1/12., window=3/12.):
    """Time-series binning using overlapping windows and weights."""

    N = len(xi)

    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)

    for i in range(N):

        idx = (x >= (xi[i] - 0.5 * window)) & \
              (x <= (xi[i] + 0.5 * window))

        ybv = y[idx]
        wbv = w[idx]

        if len(ybv) == 0:
            continue

        m0 = np.median(ybv)
        s0 = 1.4826 * np.median(np.abs(ybv - m0))

        ind = np.abs(ybv - m0) > 3.5 * s0

        if len(ybv[~ind]) == 0:
            continue

        ybi = np.sum(wbv[~ind] * ybv[~ind]) / np.sum(wbv[~ind])
        ebi = np.sum(wbv[~ind] * (ybv[~ind] - ybi)**2) / \
              np.sum(wbv[~ind])

        xb[i] = xi[i]
        yb[i] = ybi
        eb[i] = ebi

    return xb, yb, eb


# =============================================================================
# Parse arguments
# =============================================================================

args = parser.parse_args()

files = args.files
ofile = args.ofile[0]
bbox = args.bbox
dx_ = args.dxy[0] * 1e3
dy_ = args.dxy[1] * 1e3
tstep = args.tsteps[0]
tres = args.tsteps[1]
dxy = args.radius[0] * 1e3
rcc = args.corr[0]
tcc = args.corr[1]
nrel = args.nrel[0]
zlim = args.zmin[0]
niter = args.niter[0]
tspan = args.tspan
tref = args.tref[0]
dtlim = args.dtlim[0]
dhlim = args.ratelim[0]
nsig = args.thres[0]
rlim = args.thres[1]
alim = args.thres[2]
proj = args.proj[0]
njobs = args.njobs[0]
order = args.model[:]
names = args.vnames[:]
fdem = args.fdem[0]
fmsk = args.fmsk[0]
keep = args.fmsk[1]
tdiff = args.diff
binwf = args.binwf

print('parameters:')
for p in list(vars(args).items()):
    print(p)


# =============================================================================
# Main
# =============================================================================

def main(file, n=''):

    import warnings
    warnings.filterwarnings("ignore")

    if "SEC" in file:
        print("Input file is an existing output file -> exiting")
        return

    f_check = file.replace('.h5', '_SEC.h5')

    if os.path.exists(f_check):
        print("File processed:", file)
        return

    dx, dy = dx_, dy_

    print('-> loading data ...')

    xvar, yvar, tvar, zvar, svar, wbvar, wlvar, wtvar, bvar = names

    with h5py.File(file, 'r') as f:

        lon = f[xvar][:]
        lat = f[yvar][:]
        time = f[tvar][:]
        elev = f[zvar][:]
        rmse = f[svar][:] if svar in f else np.ones(lon.shape)

        has_bsc = wbvar in f
        has_lew = wlvar in f
        has_tes = wtvar in f

        bsc = f[wbvar][:] if has_bsc else np.zeros(lon.shape)
        lew = f[wlvar][:] if has_lew else np.zeros(lon.shape)
        tes = f[wtvar][:] if has_tes else np.zeros(lon.shape)

        bias = f[bvar][:] if bvar in f else np.zeros(lon.shape)

    # Report waveform availability when needed
    if binwf or order[3] == 1:

        if order[3] == 1:
            print('-> scattering correction enabled ...')

        if binwf:
            print('-> waveform binning enabled ...')

        print('-> bs :', 'available' if has_bsc else 'not available')
        print('-> lew:', 'available' if has_lew else 'not available')
        print('-> tes:', 'available' if has_tes else 'not available')


    if alim is not None:
        elev[np.abs(elev) > alim] = np.nan

    # Convert coordinates
    x, y = transform_coord('4326', proj, lon, lat)

    # Apply ROI mask
    if fmsk is not None:

        Xm, Ym, Zm = tiffread(fmsk)[0:3]

        imsk = interp2d(Xm, Ym, Zm, x, y, order=0)
        imsk = imsk == int(keep)

        x = x[imsk]
        y = y[imsk]
        time = time[imsk]
        elev = elev[imsk]
        bias = bias[imsk]
        rmse = rmse[imsk]
        bsc = bsc[imsk]
        lew = lew[imsk]
        tes = tes[imsk]

    # -------------------------------------------------------------------------
    # Interpolate waveform parameters ONLY if scattering correction is enabled
    # -------------------------------------------------------------------------

    if order[3] == 1:

        if has_bsc:

            i_bsc = ~np.isfinite(bsc)

            if np.any(i_bsc) and np.any(~i_bsc):

                print("-> nan's in BSC parameter - fixing by interpolating ...")

                bsc[i_bsc] = griddata(
                    (time[~i_bsc], x[~i_bsc], y[~i_bsc]),
                    bsc[~i_bsc],
                    (time[i_bsc], x[i_bsc], y[i_bsc]),
                    method='nearest'
                )

        if has_lew:

            i_lew = ~np.isfinite(lew)

            if np.any(i_lew) and np.any(~i_lew):

                print("-> nan's in LEW parameter - fixing by interpolating ...")

                lew[i_lew] = griddata(
                    (time[~i_lew], x[~i_lew], y[~i_lew]),
                    lew[~i_lew],
                    (time[i_lew], x[i_lew], y[i_lew]),
                    method='nearest'
                )

        if has_tes:

            i_tes = ~np.isfinite(tes)

            if np.any(i_tes) and np.any(~i_tes):

                print("-> nan's in TES parameter - fixing by interpolating ...")

                tes[i_tes] = griddata(
                    (time[~i_tes], x[~i_tes], y[~i_tes]),
                    tes[~i_tes],
                    (time[i_tes], x[i_tes], y[i_tes]),
                    method='nearest'
                )

    # Bounding box
    if bbox[0] is not None:

        xmin, xmax, ymin, ymax = bbox

    else:

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

    # Time span
    if tspan[0] is not False:

        tmin, tmax = tspan

        i_time = (time > tmin) & (time < tmax)

        x = x[i_time]
        y = y[i_time]
        time = time[i_time]
        elev = elev[i_time]
        rmse = rmse[i_time]
        bsc = bsc[i_time]
        lew = lew[i_time]
        tes = tes[i_time]
        bias = bias[i_time]

    else:

        tmin = time.min()
        tmax = time.max()

    if np.abs(tmax - tmin) < dtlim:
        print("-> time-range (-t) less than allowed min time-span (-k)")
        print("-> exiting ...")
        sys.exit()

    # DEM detrending
    if fdem is not None:

        Xd, Yd, Zd = tiffread(fdem)[0:3]

        elev_dem = interp2d(Xd, Yd, Zd, x, y, order=1)

        elev = elev - elev_dem

        print("-> detrending elevations using:", fdem)

    n_bias = np.unique(bias)

    if len(n_bias) > 2 and tdiff:
        print("-> can't use -dif option when missions are M > 2")
        print("exiting ...")
        sys.exit()

    # Solution grid
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    xi = Xi.ravel()
    yi = Yi.ravel()

    lonc, latc = transform_coord(proj, '4326', xi, yi)

    print('-> building kd-tree ...')
    coord = np.c_[x, y]
    tree = cKDTree(coord)

    print('-> predicting values ...')

    f0 = np.full((len(xi), 15), np.nan)

    sec = []
    err = []

    bs_bin = []
    lew_bin = []
    tes_bin = []

    tbin = np.arange(tmin, tmax, tstep) + 0.5 * tstep

    # =========================================================================
    # Prediction loop
    # =========================================================================

    for i in range(len(xi)):

        idx, rdist = get_radius_idx(x, y, xi[i], yi[i], dxy, tree, n_rel=nrel)

        if len(idx) < zlim:
            continue

        t_span = time[idx].max() - time[idx].min()

        if t_span < dtlim:
            continue

        xc = x[idx]
        yc = y[idx]
        tc = time[idx]
        zc = elev[idx]
        sc = rmse[idx]
        bc = bias[idx]
        bs = bsc[idx]
        lw = lew[idx]
        ts = tes[idx]

        if nrel > 0:
            x_i = np.median(xc)
            y_i = np.median(yc)
        else:
            x_i = xi[i]
            y_i = yi[i]

        dx = xc - x_i
        dy = yc - y_i
        dt = tc - tref

        n_bef = len(tc[tc < tref])
        n_aft = len(tc[tc > tref])

        if n_bef == 0 or n_aft == 0:
            continue

        dr = np.sqrt(dx**2 + dy**2)

        if rcc is not None:
            dc = rcc * 1e3
            expr = np.exp(-dr / dc)
        else:
            expr = np.ones(len(dr))

        if tcc is not None:
            expt = np.exp(-dt / tcc)
        else:
            expt = np.ones(len(tc))

        wc = (1.0 / sc**2) * expr * expt

        Ac = model_order(
            order.copy(),
            dt,
            bc,
            pxy=[dx, dy],
            wf=[bs.copy(), lw.copy(), ts.copy()]
        )

        try:

            xhat, ehat = lstsq(
                Ac.copy(),
                zc.copy(),
                w=wc.copy(),
                n_iter=niter,
                n_sigma=nsig,
                ylim=rlim,
                cov=True
            )[0:2]

        except:

            print("-> can't solve least-squares system ...")
            continue


        if np.abs(xhat[1]) > dhlim or np.isnan(xhat[1]):
            continue

        dz = zc - np.dot(Ac, xhat)

        ibad = np.abs(dz) > nsig * mad_std(dz)

        dz[ibad] = np.nan

        rms = mad_std(dz)

        if np.isnan(rms) or rms == 0:
            continue

        cols = [1, 2, 3, 4, 5]

        hc = dz + np.dot(Ac[:, cols], xhat[cols])

        # Difference two datasets
        if tdiff and order[-1] == 1:

            ind = np.unique(bc)

            if len(ind) < 2:
                continue

            if len(tc[bc == ind[0]]) == 0:
                continue

            if len(tc[bc == ind[1]]) == 0:
                continue

            tr = tc[bc == ind[0]]
            hr = hc[bc == ind[0]]

            ti = tc[bc == ind[1]]
            hi = hc[bc == ind[1]]

            #plt.figure()
            #plt.plot(ti,hi,'.')
            #plt.plot(tr,hr,'.')
            #plt.title(xhat[-1])
            #plt.show()

            hc_i = np.full(len(tc), np.nan)

            h_dif = hi - np.interp(ti, tr, hr)

            hc_i[bc == ind[1]] = h_dif.copy() + xhat[-1]

            hc = hc_i[:]

            #plt.figure()
            #plt.plot(ti, h_dif.copy() + xhat[-1],'.')
            #plt.title(xhat[-1])
            #plt.show()

        # Seasonal parameters
        s_amp = np.nan
        s_phs = np.nan

        s_cos = np.nan if xhat[4] == 0 else xhat[4]
        s_sin = np.nan if xhat[5] == 0 else xhat[5]

        if ~np.isnan(s_sin) and ~np.isnan(s_cos):

            s_amp = np.sqrt(s_sin**2 + s_cos**2)

            s_phs = int(365.25 * np.arctan2(s_sin, s_cos) / (2 * np.pi))

            if s_phs < 0:
                s_phs += 365

        # Elevation time series
        inan = ~np.isnan(hc)

        tb, zb, eb = resample(
            tc[inan].copy(),
            hc[inan].copy(),
            w=wc[inan].copy(),
            xi=tbin,
            dx=tstep,
            window=tres
        )

        # ---------------------------------------------------------------------
        # Optional waveform time-series binning
        #
        # This does NOT trigger waveform interpolation. If scattering
        # correction is disabled, only the original finite waveform
        # observations are used.
        # ---------------------------------------------------------------------

        if binwf:

            iwf = (
                ~ibad &
                np.isfinite(tc) &
                np.isfinite(wc) &
                (wc > 0)
            )

            if has_bsc:

                ii = iwf & np.isfinite(bs)

                if np.any(ii):

                    _, bs_b, _ = resample(
                        tc[ii].copy(),
                        bs[ii].copy(),
                        w=wc[ii].copy(),
                        xi=tbin,
                        dx=tstep,
                        window=tres
                    )

                else:

                    bs_b = np.full(len(tbin), np.nan)

            if has_lew:

                ii = iwf & np.isfinite(lw)

                if np.any(ii):

                    _, lew_b, _ = resample(
                        tc[ii].copy(),
                        lw[ii].copy(),
                        w=wc[ii].copy(),
                        xi=tbin,
                        dx=tstep,
                        window=tres
                    )

                else:

                    lew_b = np.full(len(tbin), np.nan)

            if has_tes:

                ii = iwf & np.isfinite(ts)

                if np.any(ii):

                    _, tes_b, _ = resample(
                        tc[ii].copy(),
                        ts[ii].copy(),
                        w=wc[ii].copy(),
                        xi=tbin,
                        dx=tstep,
                        window=tres
                    )

                else:

                    tes_b = np.full(len(tbin), np.nan)

        #plt.figure()
        #plt.plot(tb,zb-np.nanmean(zb),'o')
        #plt.plot(tb,bs_b-np.nanmean(bs_b),'o')
        #plt.plot(tb,lew_b-np.nanmean(lew_b),'o')
        #plt.plot(tb,tes_b-np.nanmean(tes_b),'o')
        #plt.show()

        # Calibration bias
        if order[-1] == 1 and len(n_bias) == 2:
            cal_logic = True
        else:
            cal_logic = False

        # Static output
        f0[i, 0] = x_i
        f0[i, 1] = y_i
        f0[i, 2] = xhat[0]
        f0[i, 3] = xhat[1]
        f0[i, 4] = xhat[2]
        f0[i, 5] = ehat[0]
        f0[i, 6] = ehat[1]
        f0[i, 7] = ehat[2]
        f0[i, 8] = s_amp
        f0[i, 9] = s_phs
        f0[i, 10] = rms
        f0[i, 11] = len(zc)
        f0[i, 12] = np.min(dr)
        f0[i, 13] = t_span
        f0[i, 14] = xhat[-1] if cal_logic else np.nan

        sec.append(zb)
        err.append(eb)

        if binwf and has_bsc:
            bs_bin.append(bs_b)

        if binwf and has_lew:
            lew_bin.append(lew_b)

        if binwf and has_tes:
            tes_bin.append(tes_b)

        # Progress
        if (i % 100) == 0:

            print(
                'cell#', str(i) + "/" + str(len(xi)),
                'trend:', np.around(xhat[1], 2), 'm/yr',
                'n_pts:', len(dz),
                'reloc_dist:', np.around(rdist),
                'rmse:', np.around(rms, 2),
                'offset:', np.around(xhat[14:], 2)
            )

    # =========================================================================
    # Stack output
    # =========================================================================

    try:

        sec = np.vstack(sec)
        err = np.vstack(err)

        if binwf and has_bsc:
            bs_bin = np.vstack(bs_bin)

        if binwf and has_lew:
            lew_bin = np.vstack(lew_bin)

        if binwf and has_tes:
            tes_bin = np.vstack(tes_bin)

    except:

        return

    vars = [
        'lon',
        'lat',
        'p0',
        'p1',
        'p2',
        'p0_error',
        'p1_error',
        'p2_error',
        'amplitude',
        'phase',
        'rmse',
        'nobs',
        'dmin',
        'tspan',
        'offset'
    ]

    if ofile:
        outfile = ofile
    else:
        outfile = file

    # Convert solution coordinates back to lon/lat
    f0[:, 0], f0[:, 1] = transform_coord(proj, '4326', f0[:, 0], f0[:, 1])

    path, ext = os.path.splitext(outfile)
    ofile0 = path + '_SEC.h5'

    inan = np.isnan(f0[:, 3])

    f0 = f0[~inan, :]

    # =========================================================================
    # Save
    # =========================================================================

    with h5py.File(ofile0, 'w') as foo:

        for v, g in zip(vars, f0.T):
            foo[v] = g

        foo['time'] = tbin
        foo['sec(t)'] = sec
        foo['rms(t)'] = err

        if binwf and has_bsc:
            foo['bs(t)'] = bs_bin

        if binwf and has_lew:
            foo['lew(t)'] = lew_bin

        if binwf and has_tes:
            foo['tes(t)'] = tes_bin

    print('*' * 100)

    print('%s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' % (
        'Mean:', np.nanmean(f0[:, 3]),
        'Std:', np.nanstd(f0[:, 3]),
        'Min:', np.nanmin(f0[:, 3]),
        'Max:', np.nanmax(f0[:, 3]),
        'RMSE:', np.nanmean(f0[:, 10])
    ))

    print('*' * 100)


# =============================================================================
# Run
# =============================================================================

if njobs == 1:

    print('running sequential code ...')

    [main(f, n) for n, f in enumerate(files)]

else:

    print('running parallel code (%d jobs) ...' % njobs)

    from joblib import Parallel, delayed, parallel_backend

    with parallel_backend("loky", inner_max_num_threads=1):

        Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(f, n)
            for n, f in enumerate(files)
        )

#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Surface elevation changes and associated time series from satellite and
airborne altimetry.

Program computes surface elevation changes, time series and seasoanl parameters
from either space or airborne altimetry. It also has the capability of merging
elevation data from two different sources. The is similar to fittopo.py but
allows for more generic use by combining "fittopo.py" and "corrscat.py".

The input of the software is very similar to "fittopo.py" with the choice of
number of relocations, spatial correlation lengths and local data editing.
The software provides sevearl different options to the user to process the data
in the form of several surface models (bilinear and biquadratic), polynomial
orders (trend and acceleartion) and estiamtion of seasonal parameters
(ampliutude and phase). It further allows for estimation of data offsets which
are used to cross-calibrate datasets or solve for ascending/descenifng biases.

If radar data is used the user can provide waveform parameters (max three) to
and activate the waveform correction on the model setup to reduce the effects of
changes in scattering regime on the time series and trends in the solution. The
user can provide a maximum of three of just use one if needed. This is
controlled by the -v option. There should be no NaN's on these parameters, but
the program will try to interpolate nan's if they are there.

The software allows for other data sources to be merged if called for and two
inputs are needed: (1) is a input variable (m_idx) classifying each mission with
an integer (0 to n - where 0 is the reference data). These vector will be added
as dummy variable (0/1) to the design matrix and offsets solved for and removed.
The user needs activate this option in the model setup (-m).

Further the software also allows for two datasets to be differenced given 0 or 1
index in m_idx (CS2=1 and IS2=0) using combined '-m' and '-diff' option. Point
differences are computed by interpolating data "1" to data "0" using their
timestamps. Estimated offset bias between "0" and "1" is then added back and the
data is binned given user defined resolution and saved in sec(t).

Output of the program is a file contaning the following variables

    *** Static Parameters: ****
    lon, lat = longitude and latitude
    p0, p1, p2, = intercept, trend and acceleration
    p0_error, p1_error, p2_error = corresponding standard errors
    amplitude, phase = seasonal amplitude and phase
    rmse = rmse of residuals
    nobs = number of data points in solution
    dmin = distance to closest point in solution
    tspan = time span of data in each solution
    offset = local offsets between two datasets

    *** Time Variable Parameters (n x t): ***
    sec(t),rms(t) = elevation change and associated errors
    time = time vector

    *** Model order format: -m t p s w b ***
    t = surface order       (x,y)    0 to 2
    p = polynomial order    (t)      0 to 3
    s = seasonal            (on/off) 0 or 1
    w = waveform corr       (on/off) 0 or 1
    b = offset/bias         (on/off) 0 or 1

    *** Explanation ***
    t: is the surface order 0 = mean, 1 = bilinear and 2 = biquadratic
    p: is the polynomial order sum(ci*t**i)
    s: estimate annual seasonal seasonal a*cos(wt) + b*sin(wt): 0=OFF and 1=ON
    w: apply scattering correction (bs,lew,tes) 0=OFF and 1=ON
    b: mission index array 0 - N (0,1,2,...) "0" is considered reference

Example:

    python fitsec.py file.h5 -d 1 1 -r 1 -q 1 -i 5 -z 10 -t 2010.5 2022 \
            -f 2020 -l 10 -k 1 -w 10 10 -j 3031 -v \
            lon lat t_year h_cor h_rms bs lew tes m_idx \
            -n 1 -m 2 1 1 1 1 -s 0.0833 0.085

    python fitsec.py file.h5 -d 1 1 -r 1 -q 1 -i 5 -z 10 -t 2010.5 2022 \
            -f 2020 -l 10 -k 1 -w 10 10 -j 3031 -v \
            lon lat t_year h_cor dum dum dum dum dum \
            -n 1 -m 2 2 1 0 0 -s 0.0833 0.085

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
description = ('Computes surface-elevation change and time series\
                from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
        help='output file name, default same as input',
        default=[None],)

parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m)'),
        default=[None,None,None,None],)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution of grid (km)'),
        default=[1,1],)

parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
        help=('search radius for data (km)'),
        default=[1,1],)

parser.add_argument(
        '-c', metavar=('rcor','tcor'), dest='corr', type=float, nargs=2,
        help=('spatio-temporal correlations (km/dyr)'),
        default=[None,None],)

parser.add_argument(
        '-q', metavar=('nrel'), dest='nrel', type=int, nargs=1,
        help=('number of relocations for search radius'),
        default=[0],)

parser.add_argument(
        '-i', metavar='niter', dest='niter', type=int, nargs=1,
        help=('max number of iterations for least-squares sol.'),
        default=[1],)

parser.add_argument(
        '-z', metavar='zmin', dest='zmin', type=int, nargs=1,
        help=('min data to compute solution'),
        default=[1],)

parser.add_argument(
        '-t', metavar=('tmin','tmax'), dest='tspan', type=float, nargs=2,
        help=('min/max time for solutions and time-series'),
        default=[False,False],)

parser.add_argument(
        '-f', metavar=('tref'), dest='tref', type=float, nargs=1,
        help=('reference time for solution'),
        default=[None],)

parser.add_argument(
        '-l', metavar=('ratelim'), dest='ratelim', type=float, nargs=1,
        help=('discard if |dh/dt| > ratelim'),
        default=[1e6],)

parser.add_argument(
        '-k', metavar=('dtlim'), dest='dtlim', type=float, nargs=1,
        help=('discard if tspan < dtlim (yrs)'),
        default=[0],)

parser.add_argument(
        '-w', metavar=('nsig','rlim','alim'), dest='thres', type=float, nargs=3,
        help=('nsig and reject if |res| > rlim or |value| > alim (m)'),
        default=[None,None,None],)

parser.add_argument(
        '-s', metavar=('tstep','tres'), dest='tsteps', type=float, nargs=2,
        help=('time and window resolution for time series'),
        default=[1./12,1./12],)

parser.add_argument(
        '-j', metavar=('proj'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG (AnIS=3031, GrIS=3413)'),
        default=['4326'],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','s','b','l','e','b'),
        dest='vnames', type=str, nargs=9, help=('names of needed varibales'),
        default=['lon','lat','t_year','h_elv','h_rms','bs','lew','tes','bias'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files",
        default=[1],)

parser.add_argument(
        '-m', metavar=(''), dest='model', type=int, nargs=5,
        help=('models selection: see notes in file'),
        default=[[0,0,0,0,0]],)

parser.add_argument(
        '-dem', metavar=('fdem'), dest='fdem', type=str, nargs=1,
        help="Name of DEM file for detrending -needs '-j' proj (.tif)",
        default=[None],)

parser.add_argument(
        '-mask', metavar=('fmsk','keep'), dest='fmsk', type=str, nargs=2,
        help="mask for on(1)/off(0) ROI -needs '-j' proj (.tif)",
        default=[None, None],)

parser.add_argument(
        '-dif', dest='diff', action='store_true',
        help=('compute difference of time series before binning'),
        default=False,)

def make_time(tmin,tmax):
    """ Make monthly vector """
    import datetime
    from astropy.time import Time

    mm = np.arange(1,12+1,1)
    yy = np.arange(tmin,tmax, 1)
    time = []
    for y in yy:
        for m in mm:
            date = datetime.datetime(int(y), int(m), 15, 0, 0, 0)
            date = Time(date,format='datetime')
            time.append(date.decimalyear)
    time = np.asarray(time)
    return time


def get_radius_idx(x, y, x0, y0, r, tree, n_rel=0):
    """ Get indices of all data points inside radius. """

    # Query the Tree from the node
    idx = tree.query_ball_point((x0, y0), r)

    # Set start value
    reloc_dist = 0.

    # Either no relocation or not enough points to do relocation
    if n_rel < 1 or len(idx) < 2: return idx, reloc_dist

    # Relocate center of search radius and query again
    for k in range(n_rel):

        # Compute new search location => relocate initial center
        x0_new, y0_new = np.median(x[idx]), np.median(y[idx])

        # Compute relocation distance
        reloc_dist = np.hypot(x0_new-x0, y0_new-y0)

        # Do not allow total relocation to be larger than the search radius
        if reloc_dist > r: break

        # Query from the new location
        idx = tree.query_ball_point((x0_new, y0_new), r)

        # If max number of relocations reached, exit
        if n_rel== k + 1:
            break

    return idx, reloc_dist


def model_order(order, dt, bo, pxy, wf):
    """
    Set model parameters and return design matrix A.

    Args:
        order : tuple (t, p, s, w, b)
            - t = topography order (0–2)
            - p = polynomial order (0–3)
            - s = seasonal on/off (0–1)
            - w = waveform correction on/off (0–1)
            - b = offset/dummy type (0–2)
                0 = none
                1 = include dummy variables for each mission in `bo`
                2 = residual calibration (handled outside)
        dt : array_like, shape (N,)
            Time vector (independent variable).
        bo : array_like, shape (N,)
            Mission index per sample (integers, e.g. 1..n).
        pxy : tuple of arrays (dx, dy), each shape (N,)
            Spatial coordinates.
        wf : tuple of arrays (dbsc, dlew, dtes), each shape (N,)
            Waveform parameters.

    Returns:
        A : ndarray, shape (N, M)
            Design matrix with polynomial, spatial, seasonal, waveform,
            and optionally mission dummy columns appended at the end.
    """

    # Unpack order
    t, p, s, w, b = order

    # --- Topography terms ---
    if t == 0:
        ct0, ct1 = 0, 0
    elif t == 1:
        ct0, ct1 = 1, 0
    elif t == 2:
        ct0, ct1 = 1, 1
    else:
        raise ValueError("Topography order (t) must be 0–2")

    # --- Polynomial terms (up to cubic) ---
    cp0 = cp1 = cp2 = 0
    if p == 1:
        cp0 = 1
    elif p == 2:
        cp0 = cp1 = 1
    elif p == 3:
        cp0 = cp1 = cp2 = 1
    elif p != 0:
        raise ValueError("Time polynomial order (p) must be 0–3")

    # --- Seasonal and waveform flags ---
    cs0 = 1 if s == 1 else 0
    cw0 = 1 if w == 1 else 0

    # --- Coordinates and waveform parameters ---
    dx, dy = pxy
    dbsc, dlew, dtes = wf

    # Center waveform parameters if enabled
    if w > 0:
        dbsc = np.asarray(dbsc) - np.nanmedian(dbsc)
        dlew = np.asarray(dlew) - np.nanmedian(dlew)
        dtes = np.asarray(dtes) - np.nanmedian(dtes)

    # --- Base design matrix ---
    A = np.vstack((
        np.ones_like(dx),
        cp0 * dt,
        0.5 * cp1 * dt**2,
        (1.0 / 6.0) * cp2 * dt**3,
        cs0 * np.cos(2.0 * np.pi * dt),
        cs0 * np.sin(2.0 * np.pi * dt),
        ct0 * dx, ct0 * dy,
        ct0 * dx * dy,
        ct1 * dx**2, ct1 * dy**2,
        cw0 * dbsc, cw0 * dlew, cw0 * dtes
    )).T

    # --- Append mission dummy variables if b == 1 ---
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
    """Time-series binning (w/overlapping windows)
        and weights.

    Args:
        x,y,w : time, value and weight of time series.
        xi    : time vector of returned binned series.
        dx    : time step of binning.
        window: size of binning window.
    """

    N = len(xi)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)

    for i in range(N):

        # Window of data centered on time index
        idx = (x >= (xi[i] - 0.5*window)) & \
              (x <= (xi[i] + 0.5*window))

        # Get weights and data
        ybv = y[idx]
        wbv = w[idx]

        # Skip if no data
        if len(ybv) == 0: continue

        # Compute initial stats
        m0 = np.median(ybv)
        s0 = 1.4826 * np.median(np.abs(ybv - m0))

        # Index of outliers using 3.5 robust sigma rule
        ind = np.abs(ybv - m0) > 3.5 * s0

        # Check for issues
        if len(ybv[~ind]) == 0: continue

        # Weighted solution - if active
        ybi = np.sum(wbv[~ind] * ybv[~ind]) / np.sum(wbv[~ind])
        ebi = np.sum(wbv[~ind] * (ybv[~ind] - ybi)**2) / np.sum(wbv[~ind])

        # Save values and error
        xb[i] = xi[i]
        yb[i] = ybi
        eb[i] = ebi

    return xb, yb, eb

# Parser for input
args = parser.parse_args()

# Pass arguments
files  = args.files              # input file(s)
ofile  = args.ofile[0]           # output file
bbox   = args.bbox               # bounding box EPSG (m) or geographical (deg)
dx_    = args.dxy[0] * 1e3       # grid spacing in x (km -> m)
dy_    = args.dxy[1] * 1e3       # grid spacing in y (km -> m)
tstep  = args.tsteps[0]          # time spacing in t
tres   = args.tsteps[1]          # averaging window for time series
dxy    = args.radius[0] * 1e3    # min search radius (km -> m)
rcc    = args.corr[0]            # correlation length space
tcc    = args.corr[1]            # correlation length time
nrel   = args.nrel[0]            # number of relocations
zlim   = args.zmin[0]            # min obs for solution
niter  = args.niter[0]           # number of iterations for solution
tspan  = args.tspan              # min/max time for solution (d.yr)
tref   = args.tref[0]            # ref time for solution (d.yr)
dtlim  = args.dtlim[0]           # min time difference needed for solution
dhlim  = args.ratelim[0]         # discard estimate if |dh/dt| > value (m)
nsig   = args.thres[0]           # outlier rejection criteria n x std.dev
rlim   = args.thres[1]           # remove residual if |resid| > limit (m)
alim   = args.thres[2]           # remove value if |value| > limit (m)
proj   = args.proj[0]            # EPSG number (GrIS=3413, AnIS=3031)
njobs  = args.njobs[0]           # number of parallel processes
order  = args.model[:]           # model order selection
names  = args.vnames[:]          # name of parameters of interest
fdem   = args.fdem[0]            # name of DEM file used for detrending
fmsk   = args.fmsk[0]            # name of masking file
keep   = args.fmsk[1]            # name of masking file
tdiff  = args.diff               # compute differences of time series (yes/no)

print('parameters:')
for p in list(vars(args).items()): print(p)

# Start of main function
def main(file, n=''):

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Don't read our output
    if "SEC" in file:
        print("Input files is an existing output file -> exiting")
        return

    # Check if we have processed it
    f_check = file.replace('.h5','_SEC.h5')

    # Check if file exists
    if os.path.exists(f_check) is True:
        print("File processed:", file)
        return
    
    # Global to local inside function
    dx, dy = dx_, dy_

    print('loading data ...')

    # Get variable names
    xvar, yvar, tvar, zvar, svar, wbvar, wlvar, wtvar, bvar = names

    # Read needed/wanted variables
    with h5py.File(file, 'r') as f:
        lon  = f[xvar][:]
        lat  = f[yvar][:]
        time = f[tvar][:]
        elev = f[zvar][:]
        rmse = f[svar][:]  if svar  in f else np.ones(lon.shape)
        bsc  = f[wbvar][:] if wbvar in f else np.zeros(lon.shape)
        lew  = f[wlvar][:] if wlvar in f else np.zeros(lon.shape)
        tes  = f[wtvar][:] if wtvar in f else np.zeros(lon.shape)
        bias = f[bvar][:]  if bvar  in f else np.zeros(lon.shape)

    # Remove large values
    if alim is not None:
        elev[np.abs(elev) > alim] = np.nan

    # Convert data to wanted projection
    x, y = transform_coord('4326', proj, lon, lat)

    # Remove data that is not inside the ROI
    if fmsk is not None:

        # Read DEM file
        Xm, Ym, Zm = tiffread(fmsk)[0:3]

        # Bi-linear interpolation of grid to points
        imsk = interp2d(Xm, Ym, Zm, x, y, order=0)

        # Create the boolean
        imsk = imsk == int(keep)

        # Edit all the data
        x    = x[imsk]
        y    = y[imsk]
        time = time[imsk]
        elev = elev[imsk]
        bias = bias[imsk]
        rmse = rmse[imsk]
        bsc  = bsc[imsk]
        lew  = lew[imsk]
        tes  = tes[imsk]

    # Find NaNs in waveform parameters
    i_bsc = np.isnan(bsc)
    i_lew = np.isnan(lew)
    i_tes = np.isnan(tes)

    # Check for NaN's in waveform parameters
    if len(bsc[np.isnan(bsc)]) > 0:
        print("nan's in BSC parameter - fixing by interpolating ...")
        bs[i_bsc] = griddata((time[~i_bsc], x[~i_bsc], y[~i_bsc]), bs[~i_bsc],
                             (time[i_bsc], x[i_bsc], y[i_bsc]), 'nearest')
    if len(lew[np.isnan(lew)]) > 0:
        print("nan's in LEW parameter - fixing by interpolating ...")
        lew[i_lew] = griddata((time[~i_lew], x[~i_lew], y[~i_lew]), lew[~i_lew],
                              (time[i_lew], x[i_lew], y[i_lew]), 'nearest')
    if len(tes[np.isnan(tes)]) > 0:
        print("nan's in TES parameter - fixing by interpolating ...")
        tes[i_tes] = griddata((time[~i_tes], x[~i_tes], y[~i_tes]), tes[~i_tes],
                              (time[i_tes], x[i_tes], y[i_tes]), 'nearest')

    # Set bounding box for the data if needed
    if bbox[0] is not None:
        xmin, xmax, ymin, ymax = bbox
    else:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

    # Determine the time span of the data
    if tspan[0] is not False:

        # Get time provided time span
        tmin, tmax = tspan

        # Select only observations inside time interval
        i_time = (time > tmin) & (time < tmax)

        # Keep only data inside time span
        x    = x[i_time]
        y    = y[i_time]
        time = time[i_time]
        elev = elev[i_time]
        rmse = rmse[i_time]
        bsc  = bsc[i_time]
        lew  = lew[i_time]
        tes  = tes[i_time]
        bias = bias[i_time]

    else:

        # Set time spans to data
        tmin, tmax = time.min(),time.max()

    # Check time span and time limit for solution
    if np.abs(tmax - tmin) < dtlim:
        print("-> time-range (-t) less than allowed min time-span (-k) ")
        print("-> exiting ...")
        sys.exit()

    # Detrend data if a DEM-file is provided
    if fdem is not None:

        # Read DEM file
        Xd, Yd, Zd = tiffread(fdem)[0:3]

        # Bilinear interpolation of grid to points
        elev_dem = interp2d(Xd, Yd, Zd, x, y, order=1)

        # Detrend you elevation data
        elev = elev - elev_dem

        print("detrending elevations using:",fdem)

    # Get me number of missions
    n_bias = np.unique(bias)

    if len(n_bias) > 2 and tdiff:
        print("-> can't use -dif option when missions are M > 2")
        print("exiting ...")
        sys.exit()

    # Solution grid
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Flatten grid coordinates 2d -> 1d
    xi, yi = Xi.ravel(), Yi.ravel()

    # Convert centroid location to latitude and longitude
    lonc, latc = transform_coord(proj, '4326', xi, yi)

    # Make a list of data coords
    coord = list(zip(x.ravel(), y.ravel()))

    print('building kd-tree ...')

    # Construct KD-tree to query
    tree = cKDTree(coord)

    print('predicting values ...')

    # Output data containers
    f0  = np.full((len(xi), 15), np.nan)
    geo = []
    sec = []
    err = []

    # Time vector
    tbin = np.arange(tmin, tmax, tstep) + 0.5 * tstep

    # Prediction loop
    for i in range(len(xi)):

        # Relocation of data
        idx, rdist  = get_radius_idx(x, y, xi[i], yi[i], dxy, tree, n_rel=nrel)

        # Reject if not enough data
        if len(idx) < zlim: continue

        # Compute time span of data inside radius
        t_span = time[idx].max() - time[idx].min()

        # Reject if time span is to short
        if t_span < dtlim: continue

        # Parameters for model-solution
        xc = x[idx]
        yc = y[idx]
        tc = time[idx]
        zc = elev[idx]
        sc = rmse[idx]
        bc = bias[idx]
        bs = bsc[idx]
        lw = lew[idx]
        ts = tes[idx]

        # Recenter coordinates
        if nrel > 0:
            x_i = np.median(xc)
            y_i = np.median(yc)
        else:
            x_i = xi[i]
            y_i = yi[i]

        # Centering of needed variables
        dx = xc - x_i
        dy = yc - y_i
        dt = tc - tref

        # Compute data before and post tref
        n_bef = len(tc[tc < tref])
        n_aft = len(tc[tc > tref])

        # Reject solution if any it true
        if n_aft == 0 or n_aft == 0: continue

        # Distance from prediction point
        dr = np.sqrt(dx**2 + dy**2)

        # Construct weights
        if rcc is not None:
            dc = rcc*1e3
            expr = np.exp(-dr/dc)
        else:
            expr = np.ones(len(dr))
        if tcc is not None:
            expt =  np.exp(-dt/tcc)
        else:
            expt = np.ones(len(tc))

        # Final weight vector
        wc = (1.0 / sc**2) * expr * expt

        # Set correct model for each solution
        Ac = model_order(order.copy(), dt, bc, pxy=[dx,dy], wf=[bs,lw,ts])

        try:
            # Solve system and invert for model parameters
            xhat, ehat = lstsq(Ac.copy(), zc.copy(), w=wc.copy(),
                                n_iter=niter, n_sigma=nsig,
                                ylim=rlim, cov=True)[0:2]
        except:
            print("can't solve least-squares system ...")
            continue

        # Check if rate is within bounds or nan
        if np.abs(xhat[1]) > dhlim or np.isnan(xhat[1]): continue

        # Residuals to model
        dz = zc - np.dot(Ac, xhat)

        # Filter residuals - robust MAD
        ibad = np.abs(dz) > nsig * mad_std(dz)

        # Remove bad data from solution
        dz[ibad] = np.nan

        # RMS error of residuals
        rms = mad_std(dz)

        # If this is true something is wrong
        if np.isnan(rms) or rms == 0: continue

        # Time columns in design matrix
        cols = [1,2,3,4,5]

        # Recover temporal trends
        hc = dz + np.dot(Ac[:,cols], xhat[cols])

        # Compute differences between data
        if tdiff and order[-1] == 1:

            # Get the indexes
            ind = np.unique(bc)

            # Check data lengths
            if len(ind) < 2: continue
            if len(tc[bc == ind[0]]) == 0: continue
            if len(tc[bc == ind[1]]) == 0: continue

            # Get the data
            tr, hr = tc[bc == ind[0]], hc[bc == ind[0]]
            ti, hi = tc[bc == ind[1]], hc[bc == ind[1]]

            # Create new data
            hc_i = np.full(len(tc),np.nan)

            # Interpolate the data inside the search area
            h_dif = hi - np.interp(ti, tr, hr)

            # Save points and add the mean offset back
            hc_i[bc == ind[1]] = h_dif.copy() + xhat[-1]

            # Replace old data with differences
            hc = hc_i[:]

        # Initialize them
        s_amp = np.nan
        s_phs = np.nan

        # Seasonal model coefficients
        s_sin = np.nan if xhat[3] == 0 else xhat[3]
        s_cos = np.nan if xhat[4] == 0 else xhat[4]

        # Check if we have issues with coeff.
        if ~np.isnan(s_sin) and ~np.isnan(s_cos):

            # Compute amplitude and phase
            s_amp = np.sqrt(s_sin**2 + s_cos**2)
            s_phs = int(365.25 * np.arctan(s_sin/s_cos) / (2*np.pi))

            # Maks sure phase is from 0-365 days
            if s_phs < 0: s_phs += 365

        # Identify NaN values in array
        inan = ~np.isnan(hc)

        # Bin data to wanted resolution
        tb, zb, eb = resample(tc[inan].copy(), hc[inan].copy(), xi=tbin,\
                                w=wc[inan].copy(), dx=tstep, window=tres)

        # Check if we need to save calibration bias
        if order[-1] == 1 and len(n_bias) == 2:
            cal_logic = True
        else:
            cal_logic = False

        # Output data
        f0[i,0]  = x_i
        f0[i,1]  = y_i
        f0[i,2]  = xhat[0] # Height/Intercept
        f0[i,3]  = xhat[1] # Rate
        f0[i,4]  = xhat[2] # Acceleration
        f0[i,5]  = ehat[0] # Height error
        f0[i,6]  = ehat[1] # Rate error
        f0[i,7]  = ehat[2] # Acceleration error
        f0[i,8]  = s_amp
        f0[i,9]  = s_phs
        f0[i,10] = rms
        f0[i,11] = len(zc)
        f0[i,12] = np.min(dr)
        f0[i,13] = t_span
        f0[i,14] = xhat[-1] if cal_logic else np.nan

        # Stack time series
        sec.append(zb)
        err.append(eb)

        # Print progress (every n-th iterations)
        if (i % 1) == 0:
            print('cell#', str(i) + "/" + str(len(xi)),\
            'trend:', np.around(xhat[1],2), 'm/yr',\
            'n_pts:', len(dz),
            'reloc_dist:', np.around(rdist),
            'rmse:',np.around(rms,2),
            'offset:',np.around(xhat[14:],2))
    try:
        # Change into arrays
        sec = np.vstack(sec)
        err = np.vstack(err)
    except:
        return

    # Name of output variables
    vars = ['lon', 'lat', 'p0', 'p1', 'p2', 'p0_error',
            'p1_error', 'p2_error','amplitude','phase',
            'rmse', 'nobs', 'dmin','tspan','offset']

    # Define output file name
    if ofile:
        outfile = ofile
    else:
        outfile = file

    # Convert back to lon/lat for 1D
    f0[:,0], f0[:,1] = transform_coord(proj,'4326', f0[:,0], f0[:,1])

    # Output file names - strings
    path, ext = os.path.splitext(outfile)
    ofile0 = path + '_SEC.h5'

    # Find NaNs in height vector
    inan = np.isnan(f0[:,3])

    # Remove all NaNs from data sets
    f0 = f0[~inan,:]

    # Save surface fit parameters
    with h5py.File(ofile0, 'w') as foo:

        # Save model solutions
        for v, g in zip(vars, f0.T):
            foo[v] = g

        # Save binned time series
        foo['time']   = tbin
        foo['sec(t)'] = sec
        foo['rms(t)'] = err

    print(('*'*100))
    print(('%s %.5f %s %.2f %s %.2f %s %.2f %s %.2f' %
    ('Mean:',np.nanmean(f0[:,3]), 'Std:',np.nanstd(f0[:,3]), 'Min:',
    np.nanmin(f0[:,3]), 'Max:', np.nanmax(f0[:,3]), 'RMSE:', np.nanmean(f0[:,10]))))
    print(('*'*100))

# Run main program
if njobs == 1:
    print('running sequential code ...')
    [main(f,n) for n, f in enumerate(files)]
else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) \
            for n, f in enumerate(files))

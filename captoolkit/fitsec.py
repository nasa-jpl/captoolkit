#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Surface elevation changes and associated time series from satellite and
airborne altimetry.

Program computes surface elevation changes, time series and seasoanl parameters
from either space or airborne altimetry. It also has the capability of merging
elevation data from two different sources. The is similar to fittopo.py but
allows for more generic use by combining "fittopo.py" and "scattcor.py".

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
controlled by the -v option. There should be non NaN's on these parameters

The software allows for two data sources to be merged and for this to work two
input are needed (1) is a input variable (m_idx) classifies each mission using
zero or ones (zero will be considered referece). This vector will be added as
dummy variable to the design matrix and an offset solved for and removed. (2)
the user needs activate this option in the model setup (-m). Two types of cross-
calibration or offsets methods are avaliable: regression (1) or regression plus
residual calibration (2). The latest option estimates the offset from the model
residuals over the overlapping time period of the two datasets. If you are using
the -b 2 option do not use any distance weights as that will more likly add an
bias.

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
    lon(t),lat(t) = longitude and latitude arrays
    sec(t),rms(t) = elevation change and associated errors
    time = time vector

    *** Model order format: -m t p s w b ***
    t = surface order (x,y)    0 to 2
    p = polynomial order (t)   0 to 2
    s = seasonal (on/off)      0 or 1
    w = waveform corr (on/off) 0 or 1
    b = offset/bias (on/off)   0 or 2

    t: is the surface order 0 = None, 1 = bilinear and 2 = biquadratic
    p: is polynomial order in time 0 = a, 1 = a + b*x and 2 = a + bx + cx^2
    s: estimate annual seasonal seasonal a*cos(wt) + b*sin(wt) 0=OFF and 1=ON
    w: apply scattering correction (bs,lew,tes) 0=OFF and 1=ON
    b: apply data source offset 0=None, 1=regression and 2=regression+residual

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

    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

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
from altimutils import transform_coord
from altimutils import make_grid
from altimutils import lstsq
from altimutils import mad_std

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
        help=('spatial resolution of solution grid (deg/km)'),
        default=[1,1],)

parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
        help=('search radius for data (km)'),
        default=[1,1],)

parser.add_argument(
        '-c', metavar='rcorr', dest='rcorr', type=float, nargs=1,
        help=('correlation length if weights are used (km)'),
        default=[1],)

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
        default=[None],)

parser.add_argument(
        '-k', metavar=('dtlim'), dest='dtlim', type=float, nargs=1,
        help=('discard if tspan < dtlim (yr)'),
        default=[None],)

parser.add_argument(
        '-w', metavar=('nsigma','reslim'), dest='thres', type=float, nargs=2,
        help=('nsigma and reject if |residual| > reslim (m)'),
        default=[None,None,False],)

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
        '-a', dest='weights', action='store_true',
        help=('apply distance and error weighting'),
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
    """ Set model parameters """
    # Model settings
    t, p, s, w, b = order

    # t = topography order      0 to 2
    # p = order of polynomial   0 to 2
    # s = seasonal on/off       0 or 1
    # w = waveform corr on/off  0 or 1
    # b = offset/bias on/off    0 or 1

    ##################
    if t == 0:
        ct0, ct1 = 0,0
    if t == 1:
        ct0, ct1 = 1,0
    if t == 2:
        ct0, ct1 = 1,1
    ##################
    if p == 0:
        cp0, cp1 = 0,0
    if p == 1:
        cp0, cp1 = 1,0
    if p == 2:
        cp0, cp1 = 1,1
    ##################
    if s == 0:
        cs0 = 0
    if s == 1:
        cs0 = 1
    ##################
    if w == 0:
        cw0 = 0
    if w == 1:
        cw0 = 1
    ##################
    if b == 0:
        cb0 = 0
    if b == 1:
        cb0 = 1
    if b == 2:
        cb0 = 1
    ##################

    # Get centered coord
    dx, dy = pxy

    # Get waveform pparameters
    dbsc, dlew, dtes = wf

    # Check if we need to center
    if w > 0:

        # Center waveform pparameters
        dbsc -= np.nanmedian(dbsc)
        dlew -= np.nanmedian(dlew)
        dtes -= np.nanmedian(dtes)

    # Set to zero if all are 0 or 1
    if len(np.unique(bo)) == 1: bo[:] = 0

    # Setup design matrix
    A = np.vstack((np.ones(dx.shape),cp0*dt, 0.5*cp1*dt**2,\
                   cs0*np.cos(2*np.pi*dt), cs0*np.sin(2*np.pi*dt),\
                   ct0*dx, ct0*dy, ct0*dx*dy, ct1*dx**2, ct1*dy**2,\
                   cw0*dbsc, cw0*dlew, cw0*dtes, cb0*bo)).T

    return A


@jit(nopython=True)
def resample(x, y, xi, w=None, dx=1/12., window=3/12, weights=False, median=False):
    """Time-series binning (w/overlapping windows)
        and weights.

    Args:
        x,y,: time, value and weight of time series.
        xi  : time vector of returned binned series.
        dx  : time step of binning.
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

        # Check use for median or mean (weighted)
        if median is not True:

            # Compute initial stats
            m0 = np.median(ybv)
            s0 = 1.4826 * np.median(np.abs(ybv - m0))

            # Index of outliers using 3.5 robust sigma rule
            ind = np.abs(ybv - m0) > 3.5 * s0

            # Check for issues
            if len(ybv[~ind]) == 0: continue

            # Weighted (spatially) or raw average
            if weights:
                ybi = np.sum(wbv[~ind] * ybv[~ind]) / np.sum(wbv[~ind])
                ebi = np.sum(wbv[~ind] * (ybv[~ind] - ybi)**2) / np.sum(wbv[~ind])

            else:
                ybi = np.mean(ybv[~ind])
                ebi = np.std(ybv[~ind])

        else:

            # Median and error for all points
            ybi = np.median(ybv)
            ebi = np.std(ybv)

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
cdr    = args.rcorr[0]           # correlation length for fit
nrel   = args.nrel[0]            # number of relocations
zlim   = args.zmin[0]            # min obs for solution
niter  = args.niter[0]           # number of iterations for solution
tspan  = args.tspan              # min/max time for solution (d.yr)
tref   = args.tref[0]            # ref time for solution (d.yr)
dtlim  = args.dtlim[0]           # min time difference needed for solution
dhlim  = args.ratelim[0]         # discard estimate if |dh/dt| > value (m)
nsig   = args.thres[0]           # outlier rejection criteria n x std.dev
rlim   = args.thres[1]           # remove residual if |resid| > value (m)
proj   = args.proj[0]            # EPSG number (GrIS=3413, AnIS=3031)
njobs  = args.njobs[0]           # number of parallel processes
order  = args.model[:]           # model order selection
names  = args.vnames[:]          # name of parameters of interest
weight = args.weights            # use distance weighting

print('parameters:')
for p in list(vars(args).items()): print(p)

# Start of main function
def main(file,cdr, n=''):

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Check if we have processed it
    f_check = file.replace('.h5','_SEC.h5')

    # Don't read our output
    if "SEC" in file: return

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

    # Check for NaN's in waveform parmeters
    if len(bsc[np.isnan(bsc)]) > 0:
        print("You have Nan's in BSC parameter that you plan on using - you need fill or remove them.")
        sys.exit()
    if len(lew[np.isnan(lew)]) > 0:
        print("You have Nan's in LEW parameter that you plan on using - you need fill or remove them.")
        sys.exit()
    if len(tes[np.isnan(tes)]) > 0:
        print("You have Nan's in TES parameter that you plan on using - you need fill or remove them.")

    # Converte data to wanted projection
    x, y = transform_coord('4326', proj, lon, lat)

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
    #tbin = make_time(tmin,tmax)

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

        # Recenter coords
        if nrel > 0:
            x_i = np.median(xc)
            y_i = np.median(yc)
        else:
            x_i = xi[i]
            y_i = yi[i]

        # Centering of needed varibales
        dx = xc - x_i
        dy = yc - y_i
        dt = tc - tref

        # Distance from prediction point
        dr = np.sqrt(dx**2 + dy**2)

        # Variance of provided error
        sv = sc ** 2

        # Apply distance weighting
        if weight:

            # Multiply to meters
            cdr = cdr*1e3

            # Weights using distance and error
            wc = 1. / (sv * (1. + (dr / cdr) ** 2))
            wbool = True

        else:

            # Set weighets
            wc = np.ones(dr.shape)
            wbool = False

        # Set correct model for each solution
        Ac = model_order(order.copy(), dt, bc, pxy=[dx,dy], wf=[bs,lw,ts])

        try:
        	# Solve system and invert for model parameters
            xhat, ehat = lstsq(Ac.copy(), zc.copy(),
                            n_iter=niter, n_sigma=nsig,
                            ylim=rlim, cov=True)[0:2]

        except:
            print("Can't solve least-squares system ...")
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
        rms = np.nanstd(dz)

        # Time columns in design matrix
        cols = [1,2,3,4]

        # Set residual offset to zero
        res_offset = np.nan

        # Check and add offsets to residuals
        if order[-1] > 1:
            try:
                # Get overlapping window for missions
                tmin_, tmax_ = tc[bc == 1].min(), tc[bc == 0].max()

                # Get overlapping data points
                dz0 = dz[bc == 0][(tc[bc == 0] > tmin_) & (tc[bc == 0] < tmax_)]
                dz1 = dz[bc == 1][(tc[bc == 1] > tmin_) & (tc[bc == 1] < tmax_)]

                # Check that we have enough points for overlap
                if len(dz0) > zlim and len(dz1) > zlim:

                    # Compute median values over both overlapping parts of data
                    b0 = np.nanmedian(dz0)
                    b1 = np.nanmedian(dz1)

                else:
                    # Dont use
                    b0 = np.nan
                    b1 = np.nan

                # Compute offset
                res_offset = b1 - b0

                # Check if any sub offset is NaN
                if ~np.isnan(res_offset):

                    # Apply offset to index=1
                    dz[bc == 1] -=  res_offset

            except:
                pass

        # Recover temporal trends
        hc = dz + np.dot(Ac[:,cols], xhat[cols])

        # Initialze them
        s_amp = np.nan
        s_phs = np.nan

        # Seasonal model coeffcinats
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
                             w=wc[inan].copy(), dx=tstep, window=tres,
                             weights=weight, median=False)

	    # Convert relocated position to geographical coords.
        if nrel > 0:
            lonc[i], latc[i] = transform_coord(proj,'4326', x_i, y_i)

        # Output data
        f0[i,0]  = lonc[i]
        f0[i,1]  = latc[i]
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
        f0[i,14] = xhat[-1] if order[-1] > 0 else np.nan

        # Stack time series
        geo.append([lonc[i], latc[i]])
        sec.append(zb)
        err.append(eb)

        # Print progress (every n-th iterations)
        if (i % 1) == 0:
            print('cell#', str(i) + "/" + str(len(xi)),  \
            'trend:', np.around(xhat[1],2), 'm/yr', \
            'n_pts:', len(dz),
            'reloc_dist:', np.around(rdist),
            'offset:',np.around(xhat[-1],2),
            'res_offset',np.around(res_offset,2))
    try:
        # Change into arrays
        sec = np.vstack(sec)
        err = np.vstack(err)
        geo = np.vstack(geo)
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

    # Output file names - strings
    path, ext = os.path.splitext(outfile)
    ofile0 = path + '_SEC.h5'

    # Find NaNs in height vector
    inan = np.isnan(f0[:,2])

    # Remove all NaNs from data sets
    f0 = f0[~inan,:]

    # Save surface fit parameters
    with h5py.File(ofile0, 'w') as foo:

        # Save model solutions
        for v, g in zip(vars, f0.T):
            foo[v] = g

        # Save binned time series
        foo['lon(t)'] = geo[:,0]
        foo['lat(t)'] = geo[:,1]
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
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f,cdr, n) \
            for n, f in enumerate(files))

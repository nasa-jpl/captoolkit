#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Program for computing robust surface elevation changes
from satellite and airborne altimetry.

Example:

    python secfit.py /pth/to/file.txt -m grid -d 10 10 -r 1 1 \
            -p 1 -z 10 -t 2005 2015 -e 2010 -l 15 -q 1 -s 10 \
            -j 3031 -c 2 1 3 4 -1 -1 -x 't + 2000'

Change Log:

    - added imports
    - added argparse (command-line args)
    - added extra input args
    - added HDF5 I/O
    - added functions
    - added optional bbox in lon/lat or x/y => Now is a TODO!
    - added optional time interval
    - added parallelization
    - added editable default parameters
    - added getting bbox from fname
    - formatted code (to reduce code size)
    - changed constraint loop - for instead of while    (JN 25/05/17)
    - changed outlier editing - iterative               (JN 25/05/17)
    - fixed code error - adding back terms correctly    (JN 25/05/17)
    - changed to centroid position for grid only        (JN 25/05/17)
    - added acceleration in design matrix               (JN 25/05/17)
    - added iteration input argument for solution       (JN 18/06/17)
    - CHANGED SEVERAL THINGS SO IT SAVES GRIDS PROPERLY!

Real use cases (Ross):

    python secfit.py ~/data/ers2/floating/ANT_ER2_ISHELF_READ_A_RM_TOPO_IBE_TIDE_SCAT.h5 -v lon lat t_year h_cor None None None -m g -b -610000 500000 -1400000 -800000 -d 3 3 -r 1 3 -e 2000 -s 12 -p 3 -o ~/data/ers2/floating/junk.h5

    python secfit.py ~/data/ers2/floating/ANT_ER2_ISHELF_READ_A_RM_TOPO_IBE_TIDE_SCAT.h5 -v lon lat t_year h_cor None None None -m g -b -610000 500000 -1400000 -800000 -d 1 1 -r 1 5 -e 1997 -s 12 -p 3 -o ~/data/ers2/floating/h1997.h5

    python secfit.py ~/data/ers2/floating/latest/AntIS_ERS2_ICE_READ_A_ROSS_RM_IBE_TIDE_MERGED_FILT_TOPO.h5 -v lon lat t_year h_cor None None None -m g -b -610000 500000 -1400000 -800000 -d 1 1 -r 1 5 -s 12 -p 3 -o ~/data/ers2/floating/latest/DEM_ERS2_ICE_A_3.h5
    
"""
__version__ = 0.2

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gdalconst import *
from osgeo import gdal, osr
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

# Default output file name, None = same as input
OUTFILE = None

# Standard name for variables
VNAMES = ['lon', 'lat', 't_year', 'h_cor', 'm_rms','m_id','h_bs']

# Default solution mode
MODE = 'g'

# Default geographic domain for solution, [] = defined by data
BBOX = None

# Defaul grid spacing in x and y (km)
DXY = [1, 1]

# Defaul min and max search radius (km)
RADIUS = [1, 1]

# Default resolution param for weighting function (km)
# If larger than max radius, set to max radius.
RESPARAM = 0.25

# Default min obs within search radius to compute solution
MINOBS = 10

# Default number of iterations for solution
NITER = 5

# Default time interval for solution [yr1, yr2], [] = defined by data
TSPAN = []

# Default reference time for solution (yr), None = mean time
TREF = None

# Default |dh/dt| limit accept estimate (m/yr)
DHDTLIM = 15

# Default time-span limit to accept estimate (yr)
DTLIM = 0.1

# Default number of missions to merge (e.g. SIM and LRM)
NMISSIONS = 1

# Default ID for solution if merging (0=SIN, 1=LRM)
IDMISSION = 0

# Default |residual| limit to accept estimate (m)
RESIDLIM = 10

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ_OBS = 3031

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ_DEM = 3031

# Default data columns (lon,lat,time,height,error,id)
COLS = [2, 1, 3, 4, -1, -1]

# Default DEM file for detrending data, None = no detrending
DEM = None

# Default expression to transform time variable
EXPR = None

# Default njobs for parallel processing
NJOBS = 1

# Default time resolution of binned time series (months)
TSTEP = 1.0

# Order of design matrix
ORDER = 3

# Output description of solution
description = ('Computes robust surface-height changes '
               'from satellite/airborne altimetry.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (ASCII, HDF5 or Numpy)')

parser.add_argument(
        '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
        help='output file name, default same as input, optional',
        default=[OUTFILE],)

parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('prediction mode: (p)oint or (g)rid'),
        choices=('p', 'g'), default=[MODE],)

parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geograph. region (deg or m), optional'),
        default=BBOX,)

parser.add_argument(
        '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
        help=('spatial resolution for grid-solution (deg or m)'),
        default=DXY,)

parser.add_argument(
        '-r', metavar=('r_min','r_max'), dest='radius', type=float, nargs=2,
        help=('min and max search radius (km)'),
        default=RADIUS,)

parser.add_argument(
        '-a', metavar='res_param', dest='resparam', type=float, nargs=1,
        help=('resolution param for weighting function (km)'),
        default=[RESPARAM],)

parser.add_argument(
        '-i', metavar='n_iter', dest='niter', type=int, nargs=1,
        help=('maximum number of iterations to solve model'),
        default=[NITER],)

parser.add_argument(
        '-z', metavar='min_obs', dest='minobs', type=int, nargs=1,
        help=('minimum obs. to compute solution'),
        default=[MINOBS],)

parser.add_argument(
        '-t', metavar=('t_min','t_max'), dest='tspan', type=float, nargs=2,
        help=('min and max time for solution (yr), optional'),
        default=TSPAN,)

parser.add_argument(
        '-e', metavar=('ref_time'), dest='tref', type=float, nargs=1,
        help=('time to reference the solution to (yr), optional'),
        default=[TREF],)

parser.add_argument(
        '-l', metavar=('dhdt_lim'), dest='dhdtlim', type=float, nargs=1,
        help=('discard estimate if |dh/dt| > dhdt_lim (m/yr)'),
        default=[DHDTLIM],)

parser.add_argument(
        '-q', metavar=('dt_lim'), dest='dtlim', type=float, nargs=1,
        help=('discard estimates if data-span < dt_lim (yr)'),
        default=[DTLIM],)

parser.add_argument(
        '-s', metavar=('tstep'), dest='tstep', type=float, nargs=1,
        help=('time resolution of binned time series (months)'),
        default=[TSTEP],)

parser.add_argument(
        '-k', metavar=('n_missions'), dest='nmissions', type=int, nargs=1,
        help=('min number of modes (merge sin and lrm), optional'),
        default=NMISSIONS,)

parser.add_argument(
        '-u', metavar=('id_mission'), dest='idmission', type=int, nargs=1,
        help=('reference id for merging (0=sin, 1=lrm), optional'),
        default=[IDMISSION],)

parser.add_argument(
        '-w', metavar=('resid_lim'), dest='residlim', type=float, nargs=1,
        help=('discard residual if |residual| > resid_lim (m)'),
        default=[RESIDLIM],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='projo', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=[str(PROJ_OBS)],)

parser.add_argument(
        '-g', metavar=('epsg_dem'), dest='projd', type=str, nargs=1,
        help=('projection: EPSG number of DEM'),
        default=[str(PROJ_DEM)],)

parser.add_argument(
        '-v', metavar=('x','y','t','h','s','i','c'), dest='vnames', type=str, nargs=7,
        help=('name of varibales in the HDF5-file'),
        default=[VNAMES],)

parser.add_argument(
        '-f', metavar=('dem.tif'), dest='dem',  type=str, nargs=1,
        help='detrend data using a-priori DEM, optional',
        default=[DEM],)

parser.add_argument(
        '-x', metavar=('expr'), dest='expr',  type=str, nargs=1,
        help="expression to apply to time (e.g. 't + 2000'), optional",
        default=[EXPR],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files, optional",
        default=[NJOBS],)

parser.add_argument(
        '-p', metavar=None, dest='model', type=int, nargs=1,
        help=('select design matrix, see line 744 in program'),
        choices=(0,1,2,3), default=[ORDER],)

args = parser.parse_args()

# Pass arguments
mode  = args.mode[0]                # prediction mode: point or grid solution
files = args.files                  # input file(s)
ofile = args.ofile[0]               # output directory
bbox_ = args.bbox                   # bounding box EPSG (m) or geographical (deg)
dx    = args.dxy[0] * 1e3           # grid spacing in x (km -> m)
dy    = args.dxy[1] * 1e3           # grid spacing in y (km -> m)
tstep_= args.tstep[0]               # time spacing in t (months)
dmin  = args.radius[0] * 1e3        # min search radius (km -> m)
dmax  = args.radius[1] * 1e3 + 1e-4 # max search radius (km -> m)
dres_ = args.resparam[0] * 1e3      # resolution param for weighting func (km -> m) [1]
nlim  = args.minobs[0]              # min obs for solution
niter = args.niter[0]               # number of iterations for solution
tspan = args.tspan                  # min/max time for solution (d.yr)
tref_ = args.tref[0]                # ref time for solution (d.yr)
dtlim = args.dtlim[0]               # min time difference needed for solution
dhlim = args.dhdtlim[0]             # discard estimate if |dh/dt| > value (m)
nmlim = args.nmissions              # min number of missions for solution [2]
nmidx = args.idmission[0]           # id to tie the solution to if merging [3]
slim  = args.residlim[0]            # remove residual if |resid| > value (m)
projo = args.projo[0]               # EPSG number (GrIS=3413, AnIS=3031) for OBS
projd = args.projd[0]               # EPSG number (GrIS=3413, AnIS=3031) for DEM
fdem  = args.dem[0]                 # detrend data using a-priori DEM (obs. proj_dem == proj_data)
expr  = args.expr[0]                # expression to transform time
njobs = args.njobs[0]               # for parallel processing
model = args.model[0]               # least-squares model order "lin"=trend+acceleration, "biq" = linear + topo
names = args.vnames[:]              # Name of hdf5 parameters of interest

print 'parameters:'
for p in vars(args).iteritems(): print p


# [1] This defines the shape (correlation length) of the
# weighting function inside the search radius. If set to a
# value larger than max-radius, then is equal to max-radius.
# [2] For Cryosat-2 only (to merge SIN and LRM mode).
# [3] ID for different mode data: 0=SIN, 1=LRM.
# [4] If err and id cols = -1, then they are not used.


def binning(x, y, xmin, xmax, dx):
    """Time series binning func."""

    bins = np.arange(xmin, xmax, dx)

    yb = np.ones(len(bins) - 1) * np.nan
    xb = np.ones(len(bins) - 1) * np.nan
    eb = np.ones(len(bins) - 1) * np.nan
    nb = np.ones(len(bins) - 1) * np.nan
    sb = np.ones(len(bins) - 1) * np.nan

    for i in xrange(len(bins)-1):

        idx = (x >= bins[i]) & (x <= bins[i+1])

        ybv = y[idx]

        if len(ybv) == 0: continue

        yb[i] = np.nanmean(ybv)
        xb[i] = 0.5*(bins[i]+bins[i+1])
        nb[i] = len(ybv)
        sb[i] = np.sum(ybv)
        eb[i] = mad_std(ybv)

    return xb, yb, eb, nb, sb


def geotiffread(ifile,metaData):
    """Read raster."""

    file = gdal.Open(ifile, GA_ReadOnly)

    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    proj = src.ExportToWkt()

    Nx = file.RasterXSize
    Ny = file.RasterYSize

    trans = file.GetGeoTransform()

    dx = trans[1]
    dy = trans[5]

    if metaData == "A":

        xp = np.arange(Nx)
        yp = np.arange(Ny)

        (Xp, Yp) = np.meshgrid(xp,yp)

        X = trans[0] + (Xp+0.5) * trans[1] + (Yp+0.5) * trans[2]
        Y = trans[3] + (Xp+0.5) * trans[4] + (Yp+0.5) * trans[5]

    if metaData == "P":

        xp = np.arange(Nx)
        yp = np.arange(Ny)

        (Xp, Yp) = np.meshgrid(xp,yp)

        X = trans[0] + Xp*trans[1] + Yp*trans[2]
        Y = trans[3] + Xp*trans[4] + Yp*trans[5]

    band = file.GetRasterBand(1)

    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def bilinear2d(xd,yd,data,xq,yq, **kwargs):
    """Raster to point interpolation."""

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0,:]
    yd = yd[:,0]

    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1]-xd[0]), (yd[1]-yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq*ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq*ones(xq.size)

    xp = (xq-xd[0])*(nx-1)/(xd[-1]-xd[0])
    yp = (yq-yd[0])*(ny-1)/(yd[-1]-yd[0])

    coord = np.vstack([yp,xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq

def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Construct output grid-coordinates."""

    # Setup grid dimensions
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    # Initiate x/y vectors for grid
    x_i = np.linspace(xmin, xmax, num=Ne)
    y_i = np.linspace(ymin, ymax, num=Nn)

    return np.meshgrid(x_i, y_i)


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def get_bbox(fname):
    """Extract bbox info from file name."""

    fname = fname.split('_')  # fname -> list
    i = fname.index('bbox')
    return map(float, fname[i+1:i+5])  # m


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def get_cap_index(x, y, t, dr, id, tree, t1lim, t2lim, nlim, dtlim, nmlim):
    """ """
    # Number of observations
    nobs = 0
        
    # Time difference
    dti = 0
        
    # Temporal sampling
    npct = 1
        
    # Number of sensors
    nsen = 0

    # Meet data constraints
    for i in xrange(len(dr)):

        # Query the Tree with data coordinates
        idx = tree.query_ball_point((x, y), dr[i])

        # Check for empty arrays
        if len(t[idx]) == 0: continue

        # Constraints parameters
        dti  = np.max(t[idx]) - np.min(t[idx])
        nobs = len(t[idx])
        nsen = len(np.unique(id[idx]))

        # Bin time vector
        t_sample = binning(t[idx], t[idx], t1lim, t2lim, 1./12.)[1]

        # Test for null vector
        if len(t_sample) == 0: continue

        # Sampling fraction
        npct = np.float(len(t_sample[~np.isnan(t_sample)])) / len(t_sample)

        # Constraints
        if nobs > nlim:
            #print 'not enough data points (nlim)!'
            if dti > dtlim:
                #print 'min time span not covered (dtlim)!'
                if nsen >= nmlim:
                    if npct > 0.70:
                        #print 'min coverage not achieved (npct)!'
                        break

    return idx, nobs, dti, dr[i], npct


def is_empty(ifile):
    """ Check for empty file. """
    if os.stat(ifile).st_size == 0:
        print 'input file is empty!'
        return True
    else:
        return False
    

# Main function for computing parameters
def main(ifile, n=''):
    
    #Check for empty file
    if is_empty(ifile):
        return

    # Start timing of script
    startTime = datetime.now()

    print 'loading data ...'

    # Get variable names
    xvar, yvar, tvar, zvar, svar, ivar, cvar = names

    # Load all 1d variables needed
    with h5py.File(ifile, 'r') as fi:

        # Read variables
        lon    = fi[xvar][:]
        lat    = fi[yvar][:]
        time   = fi[tvar][:]
        height = fi[zvar][:]
        sigma  = fi[svar][:] if svar in fi else np.ones(lon.shape)
        id     = fi[ivar][:] if ivar in fi else np.ones(lon.shape) * nmidx
        cal    = fi[cvar][:] if cvar in fi else np.zeros(lon.shape)

        # Apply scatter correction if available
        cal[np.isnan(cal)] = 0
        height -= cal

    # EPSG number for lon/lat proj
    projGeo = '4326'

    # EPSG number for grid proj
    projGrd = projo

    print 'converting lon/lat to x/y ...'

    # If no bbox was given
    if bbox_ is None:
        try:
            # Try reading bbox from file name
            bbox = get_bbox(ifile)
        except:
            bbox = None
    else:
        bbox = bbox_

    # Get geographic boundaries + max search radius
    if bbox:

        # Extract bounding box
        (xmin, xmax, ymin, ymax) = bbox

        # Transform coordinates
        (x, y) = transform_coord(projGeo, projGrd, lon, lat)

        # Select data inside bounding box
        Ig = (x >= xmin - dmax) & (x <= xmax + dmax) & (y >= ymin - dmax) & (y <= ymax + dmax)

        # Check bbox for obs.
        if len(x[Ig]) == 0:
            print 'no data points inside bbox!'
            return

        print 'Number of obs. edited by bbox!', 'before:', len(x), 'after:', len(x[Ig])

        # Only select wanted data
        x = x[Ig]
        y = y[Ig]
        id = id[Ig]
        time = time[Ig]
        height = height[Ig]
        sigma = sigma[Ig]

    else:

        # Convert into stereographic coordinates
        (x, y) = transform_coord(projGeo, projGrd, lon, lat)

        # Get bbox from data
        (xmin, xmax, ymin, ymax) = x.min(), x.max(), y.min(), y.max()

    # Apply transformation to time
    if expr:

        time = eval(expr.replace('t', 'time'))

    # Define time interval of solution
    if tspan:

        # Time interval = given time span
        t1lim, t2lim = tspan

        # Select only observations inside time interval
        Itime = (time > t1lim) & (time < t2lim)

        # Keep only data inside time span
        x = x[Itime]
        y = y[Itime]
        id = id[Itime]
        time = time[Itime]
        height = height[Itime]
        sigma = sigma[Itime]

    else:

        # Time interval = all data
        t1lim, t2lim = time.min(), time.max()

    if mode == 'p':

        # Point solution - all points
        xi = np.copy(x)
        yi = np.copy(y)

    else:

        # Grid solution - defined by nodes
        (Xi, Yi) = make_grid(xmin, xmax, ymin, ymax, dx, dy)

        # Flatten prediction grid
        xi = Xi.ravel()
        yi = Yi.ravel()

        # Zip data to vector
        coord = zip(x.ravel(), y.ravel())

        # Construct cKDTree
        print 'building the k-d tree ...'
        Tree = cKDTree(coord)

    # Remove topography before solution
    if fdem:

        print 'removing topography ...'

        # Read DEM to memory
        (Xd, Yd, Zd) = geotiffread(fdem, "A")[0:3]

        # DEM projection test
        if projd == projGeo:

            # Interpolate raster to obs. (lat/lon)
            h_dem = bilinear2d(Xd, Yd, Zd, lon, lat, order=1)

        else:

            # Interpolate raster to obs. (x/y)
            h_dem = bilinear2d(Xd, Yd, Zd, x, y, order=1)

        # Remove topography from obs.
        height -= h_dem

    # Number of nodes
    nodes = len(xi)

    # Create bias
    bias = np.ones(lon.shape) * np.nan

    # Time resolution to years
    tstep = tstep_ / 12.0

    # Number of months of time series
    months = len(np.arange(t1lim, t2lim, tstep))

    # Total number of columns
    ntot = months + 4

    # Create output arrays
    OFILE0 = np.ones((nodes, 21))   * 9999
    OFILE1 = np.ones((nodes, ntot)) * 9999
    OFILE2 = np.ones((nodes, ntot)) * 9999

    # Set res. param to max radius if larger than r_max
    dres = dres_ if dres_ <= dmax else dmax

    # Search radius array
    dr = np.arange(dmin, dmax, 1e3)

    # Enter prediction loop
    print 'predicting values ...'
    for i in xrange(len(xi)):

        # Center coordinates
        xc, yc = xi[i], yi[i]

        # Get index and parameters
        (idx, nobs, dt, dri, npct) = get_cap_index(xc, yc, time, dr, id, Tree,\
                                                   t1lim, t2lim, nlim, dtlim, nmlim)

        # Continue to next solution if true
        if (nobs < nlim) or (dt < dtlim): continue

        # Parameters for model-solution
        xcap  = x[idx]
        ycap  = y[idx]
        tcap  = time[idx]
        Hcap  = height[idx]
        mcap  = id[idx]
        scap  = sigma[idx]

        ##NOTE: Center data before least-saquare fit
        Hcap_mean = np.nanmean(Hcap)
        Hcap -= Hcap_mean

        # Estimate variance
        vcap = scap * scap

        ##NOTE: The reference time needs to be "created" with the centered data
        # If reference time not given, use mean
        #tref = tref_ if tref_ else np.mean(tcap)
        tref = np.nanmean(tcap)

        # Design matrix elements
        c0 = np.ones(len(xcap))         # intercept    (0)
        c1 = xcap - xc                  # dx           (1)
        c2 = ycap - yc                  # dy           (2)
        c3 = c1 * c2                    # dx*dy
        c4 = c1 * c1                    # dx*dx
        c5 = c2 * c2                    # dy*dy
        c6 = tcap - tref                # trend        (6)
        c7 = 0.5 * (c6 * c6)            # acceleration (7)
        c8 = np.sin(2 * np.pi * c6)     # seasonal sin (8)
        c9 = np.cos(2 * np.pi * c6)     # seasonal cos (9)

        # Compute distance from prediction point to data inside cap
        d = np.sqrt((xcap - xc) * (xcap - xc) + (ycap - yc) * (ycap - yc))

        # Add small value to stabilize SVD solution
        vcap += 1e-6

        # Weighting factor - distance and error
        Wcap = 1.0 / (vcap * (1.0 + (d / dres) * (d / dres)))

        # Create some intermediate output variables
        sx, sy, at, ae, bi = -9999 , -9999, -9999, -9999, -9999

        # Setup design matrix
        if model == 0:

            # Trend and seasonal
            Acap = np.vstack((c0, c8, c9, c6)).T

            # Wanted columns to add back
            mcol = [1, 2, 3]

        elif model == 1:

            # Trend, acceleration and seasonal
            Acap = np.vstack((c0, c7, c8, c9, c6)).T

            # Wanted columns to add back
            mcol = [1, 2, 3, 4]

        elif model == 2:

            # Trend, acceleration, seasonal and bi-linear surface
            Acap = np.vstack((c0, c1, c2, c7, c8, c9, c6)).T

            # Wanted columns to add back
            mcol = [3, 4, 5, 6]

        else:

            # Trend, acceleration, seasonal and bi-quadratic surface (full model)
            Acap = np.vstack((c0, c1, c2, c3, c4, c5, c7, c8, c9, c6)).T

            # Wanted columns to add back
            mcol = [6, 7, 8, 9]

        # Initiate bias flag
        f_bias = False
        
        # Check if bias is needed
        if len(np.unique(mcap)) > 1:
            
            # Add bias to design matrix
            Acap = np.vstack((Acap.T, mcap)).T
            
            # Set bias flag
            f_bias = True

        # Initiate counter
        ki = 0

        # Create outlier boolean vector
        Io = np.ones(Hcap.shape, dtype=bool)

        # Length before editing
        Nb = len(Hcap)
        
        # Break flag
        i_flag = 0

        # Enter iterative solution
        while ki < niter:

            # Remove outlier if detected
            xcap, ycap, tcap, Hcap, Acap, Wcap = xcap[Io], ycap[Io], tcap[Io], Hcap[Io], Acap[Io], Wcap[Io]

            # Check constrains before solving model
            if len(xcap) < nlim:

                # Set flag for number of points!
                i_flag = 1
                break

            elif (np.max(tcap) - np.min(tcap)) < dtlim:

                # Set flag for time span!
                i_flag = 2
                break

            else:

                # Accepted!
                pass

            # Least-squares model
            linear_model = sm.WLS(Hcap, Acap, weights=Wcap, missing='drop')

            # Fit the model to the data,
            linear_model_fit = linear_model.fit()

            # Residuals dH = H - AxCm (remove model)
            res = Hcap - np.dot(Acap,linear_model_fit.params)
           
            # Outlier indexing
            Io = (np.abs(res) < 3.5 * mad_std(res)) & (np.abs(res) < slim) & ~np.isnan(res)
            
            # Exit loop if no outliers found
            if len(res[~Io]) == 0:

                # Exit loop
                break
            
            # Update counter
            ki += 1
        
        # Check if iterative editing failed
        if i_flag > 0: continue

        # Length after editing
        Na = len(xcap)
        
        # Coefficients and standard errors
        Cm = linear_model_fit.params
        Ce = linear_model_fit.bse
        
        # Check rate and rate error
        if np.abs(Cm[-1]) > dhlim or np.isinf(Ce[-1]): continue
        
        # Residuals dH = H - A * Cm (remove linear trend)
        dh = Hcap - np.dot(Acap,Cm)

        # Chi-Square of model
        chisq = linear_model_fit.rsquared_adj

        # Residuals dH = H - A * Cm (remove seasonal signal)
        resid = linear_model_fit.resid

        # Compute amplitude of seasonal signal
        asea = np.sqrt(Cm[-2] * Cm[-2] + Cm[-3] * Cm[-3])

        # Compute phase offset
        psea = np.arctan2(Cm[-2], Cm[-3])

        # Convert to phase to decimal years
        psea /= (2 * np.pi)

        # Compute root-mean-square of full model residuals
        rms = mad_std(resid)

        # Add back wanted model parameters
        dh += np.dot(Acap[:, mcol], Cm[mcol])

        # Simple binning of residuals
        (tb, hb, eb, nb) = binning(tcap.copy(), dh.copy(), t1lim, t2lim, tstep)[0:4]

        # Convert centroid location to latitude and longitude
        (lon_c, lat_c) = transform_coord(projGrd, projGeo, xc, yc)

        # Position
        OFILE0[i, 0] = lat_c
        OFILE0[i, 1] = lon_c

        # Elevation Change
        OFILE0[i, 2] = Cm[-1]  # trend
        OFILE0[i, 3] = Ce[-1]  # trend error

        # Acceleration Change
        if model > 0:

            # Compute acceleration and error
            at, ae  = Cm[-4], Ce[-4]

        OFILE0[i, 4] = at  # acceleration
        OFILE0[i, 5] = ae  # acceleration error

        # Surface Elevation
        OFILE0[i, 6] = Cm[0] + Hcap_mean  ##FIXME: Where to put '+ Hcap_mean'
        OFILE0[i, 7] = Ce[0]

        # Model RMS
        OFILE0[i, 8] = rms

        # Check surface slope estimates
        if model > 1:

            # Compute x,y slopes in degrees
            sx, sy  = np.arctan(Cm[1])*(180 / np.pi), np.arctan(Ce[2])*(180 / np.pi)

        # Surface slope values
        OFILE0[i, 9] = sx
        OFILE0[i,10] = sy

        # Time span of data in cap
        OFILE0[i, 11] = dt
        OFILE0[i, 12] = tref

        # Seasonal signal
        OFILE0[i, 13] = asea
        OFILE0[i, 14] = psea

        # Check for bias
        if f_bias:
    
            # Bias magnitude
            bi = Cm[-1]

        # Aux-data from solution
        OFILE0[i, 15] = len(Hcap)
        OFILE0[i, 16] = dmin
        OFILE0[i, 17] = dri
        OFILE0[i, 18] = (Nb - Na)
        OFILE0[i, 19] = chisq
        OFILE0[i, 20] = bi

        # Time series values
        OFILE1[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(tb), hb))
        OFILE2[i, :] = np.hstack((lat_c, lon_c, t1lim, t2lim, len(tb), eb))

        # Print progress (every N iterations)
        if (i % 100) == 0:
            print str(ifile)+": "+str(i) + "/" + str(len(xi))+' Iterations: ' \
                    +str(ki)+ ' Rate: '+str(np.around(Cm[mcol[-1]],2))+ \
                    ' m/yr',' Sampling: '+str(np.around(npct * 100.0, 2))

    # Find any no-data value
    if mode == 'p':

        I09999 = np.where(OFILE0[:, 3] == 9999)
        I19999 = np.where(OFILE1[:, 3] == 9999)
        I29999 = np.where(OFILE2[:, 3] == 9999)

        # Delete np-data from output file
        OFILE0 = np.delete(OFILE0.T, I09999, 1).T
        OFILE1 = np.delete(OFILE1.T, I19999, 1).T
        OFILE2 = np.delete(OFILE2.T, I29999, 1).T

    else:

        DATA = OFILE0.copy()
        DATA[DATA==9999] = np.nan
        DATA[DATA==-9999] = np.nan

        variables = ['lat', 'lon', 'trend', 'trend_err', 'accel', 'accel_err',
                     'height', 'height_err', 'model_rms', 'slope_x', 'slope_y',
                     't_span', 't_ref', 'amp_seas', 'pha_seas', 'n_obs',
                     'd_min', 'd_ri', 'n_edited', 'chi2', 'bias']

        grids = [d.reshape(Xi.shape) for d in DATA.T]

    # Check if output arrays are empty
    if (OFILE0[:, 3] == 9999).all():
        print 'no predictions to save!', ifile
        return

    # Define output file name
    n = str(n)
    if ofile:
        outfile = ofile
        if n:
            n = '_' + n
    else:
        outfile = ifile

    n = ''

    # Output file names - strings
    path, ext = os.path.splitext(outfile)

    ofile0 = path + '_sf' + n + '.h5'
    ofile1 = path + '_ts' + n + '.h5'
    ofile2 = path + '_es' + n + '.h5'

    # Save data
    print 'saving data ...'

    # Save surface fits parameters
    with h5py.File(ofile0, 'w') as fo0:
        if mode == 'p':
            fo0['sf'] = OFILE0
        else:
            fo0['x'] = Xi[0,:]
            fo0['y'] = Yi[:,0]
            for v,g in zip(variables, grids):
                fo0[v] = g

    # Save binned time series values
    with h5py.File(ofile1, 'w') as fo1:
        fo1['ts'] = OFILE1

    # Save binned time series errors
    with h5py.File(ofile2, 'w') as fo2:
        fo2['es'] = OFILE2


    # Print some statistics
    print '*************************************************************************'
    print('%s %s %.5f %s %.2f %s %.2f %s %.2f %s %s %s' %
    ('* Statistics','Mean:',np.mean(OFILE0[:,2]),'Std.dev:',np.std(OFILE0[:,2]),'Min:',
         np.min(OFILE0[:,2]),'Max:',np.max(OFILE0[:,2]),'Model:',model,'*'))
    print '*************************************************************************'

    # Print execution time of algorithm
    print 'Execution time: '+ str(datetime.now()-startTime)
    print 'Surface fit results:', ofile0
    print 'Time series values:' , ofile1
    print 'Time series errors:' , ofile2


# Run main program
if njobs == 1:
    print files
    print 'running sequential code ...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(files))

    '''
    from dask import compute, delayed
    from distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=8, threads_per_worker=None,
                          scheduler_port=8002, diagnostics_port=8003)
    client = Client(cluster)  # connect to cluster
    print client

    #values = [delayed(main)(f) for f in files]
    #results = compute(*values, get=client.get)
    values = [client.submit(main, f) for f in files]
    results = client.gather(values)
    '''

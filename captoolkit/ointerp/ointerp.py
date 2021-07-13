# -*- coding: utf-8 -*-
"""Optimal Interpolation of spatial data.

Interpolate spatial data using a modeled (analytical) covariance function.

Example:
    python ointerp.py ~/data/ers1/floating/filt_scat_det/joined_pts_ad.h5_ross
        -d 3 3 0.25 -r 15 -k 0.125 -e .2 -v t_year lon lat h_res None 
        -x -152 0 -t 1995.0 1995.25

Test bbox:
    -b 6000 107000 -1000000 -900000  
    -b -520000 -380000 -1230000 -1030000

Good performance:
    (w/N_SAMPLES=0.2)
    python ointerp_new.py -v t_ref lon lat trend trend_err trk 
        ~/data/ers2/floating/latest/SECFIT*_AD*_q75* -d 2 2 -r 15

"""
##TODO: Implement variable search radius (not sure it's needed)?

import os
import sys
import h5py
import pyproj
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
from numba import jit, int32, float64
from scipy.spatial.distance import cdist, pdist, squareform

import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

#=== Edit ==================================================

MIN_OBS = 25

N_SAMPLES = None  # If number, random sector sampling 

DEM_RA = False  ##FIXME: Only use this for 2d DEM interpolation of standard RA (height, trend, accel)

##NOTE: Passing a fixed error provides a smoother result: -v lon lat time height 0.3
##NOTE: optimal radius seems to be -r 15 (for all missions)
##NOTE: optimal sub-sampling seems to be 20% (on Ross!)
##NOTE: MIN_OBS: use 5 for DEM, 50 for resid pts

#-----------------------------------------------------------

# Modeled parameters

## height
#model, R, s = "gauss", 2377.6532, 24.4167
#model, R, s = "markov", 2167.7809, 23.1550
#model, R, s = "generic", 2926.1775, 23.5546
 
## trend 
#model, R, s = "gauss", 3420.2284, 0.3211
#model, R, s = "markov", 1568.8203, 0.3327
#model, R, s = "generic", 3521.8096, 0.3000
 
## accel
#model, R, s = "gauss", 989.2990, 1.2241
#model, R, s = "markov", 972.1889, 0.6611
#model, R, s = "generic", 2011.6638, 0.8891

## resid
#model, R, s = "gauss", 2342.3279, 0.6467
model, R, s = "markov", 985.3735, 0.6825
#model, R, s = "generic", 1742.6984, 0.6514

#===========================================================

""" Covariance models. """

def gauss(r, s, R):
    return s**2 * np.exp(-r**2/R**2)

def markov(r, s, R):
    return s**2 * (1 + r/R) * np.exp(-r/R)

def generic(r, s, R):
    return s**2 * (1 + (r/R) - 0.5 * (r/R)**2) * np.exp(-r/R)

def exp(t, tau):
    return np.exp(-t**2/tau**2)

def covxt(r, t, s, R, tau):
    """ C(r,t) = C(r) * C(t). """
    return markov(r, s, R) * exp(t, tau) 

if model == 'markov':
    covmodel = markov
elif model == 'gauss':        
    covmodel = gauss
elif model == 'covxt':
    covmodel = covxt  # with t=0 and tau=0 -> markov
else:
    covmodel = generic

#----------------------------------------------------------

def get_args():
    """ Get command-line arguments. """
    des = 'Optimal Interpolation of spatial data'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='name of input file (HDF5)')
    parser.add_argument(
            '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
            help='name of output file (HDF5)',
            default=[None])
    parser.add_argument(
            '-s', metavar='suffix', dest='suffix', type=str, nargs=1,
            help='suffix to add to output file after ext (e.g. .h5_interp)',
            default=['_interp'])
    parser.add_argument(
            '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geograph. region (deg or m), optional'),
            default=[],)
    parser.add_argument(
            '-x', metavar=('x1', 'x2'), dest='xlim', type=float, nargs=2,
            help=('x-lim to subset data prior interp.'),
            default=[None],)
    parser.add_argument(
            '-y', metavar=('y1', 'y2'), dest='ylim', type=float, nargs=2,
            help=('y-lim to subset data prior interp.'),
            default=[None],)
    parser.add_argument(
            '-z', metavar=('t1', 't2'), dest='tlim', type=float, nargs=2,
            help=('t-lim to subset data prior interp.'),
            default=[None],)
    parser.add_argument(
            '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
            help=('grid resolution (km km)'),
            default=[1, 1],)
    parser.add_argument(
            '-m', metavar='nobs', dest='nobs', type=int, nargs=1,
            help=('number of obs. for each quadrant'),
            default=[100],)
    parser.add_argument(
            '-r', metavar='radius', dest='radius', type=float, nargs=1,
            help=('search radius for each inversion cell (km)'),
            default=[1],)
    parser.add_argument(
            '-v', metavar=('x','y', 'z', 'e'), dest='vnames',
            type=str, nargs=4,
            help=('name of lon/lat/height/sigma vars (sigma can be a number)'),
            default=[None], required=True)
    parser.add_argument(
            '-t', metavar='tvar', dest='tvar', type=str, nargs=1,
            help=('name of time var (can also be a number, or ignored'),
            default=['2000'],)
    parser.add_argument(
            '-k', metavar='kvar', dest='kvar', type=str, nargs=1,
            help=('name of track id var (if ignored, computes on the fly)'),
            default=[None],)
    parser.add_argument(
            '-e', metavar='sigma_corr', dest='sigmacorr', type=float, nargs=1,
            help=('along-track long-wavelength correlated error'),
            default=[None],)
    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)
    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help='for parallel processing of multiple files, optional',
            default=[1],)
    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Stereo AnIS (x/y):  3031
        Stereo GrIS (x/y):  3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def get_bbox(fname, key='bbox'):
    """Extract tile bbox info from file name."""
    fname = fname.split('_')  # fname -> list
    i = fname.index(key)
    return map(float, fname[i+1:i+5])  # m


def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Generate a regular grid."""
    # Setup grid dimensions
    Nx = int((np.abs(xmax - xmin)) / dx) + 1
    Ny = int((np.abs(ymax - ymin)) / dy) + 1
    # Initiate lat/lon vectors for grid
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    return np.meshgrid(x, y)


def get_limits(x, y, bbox):
    """Get indices (where) of tile limits from bbox."""
    xmin, xmax, ymin, ymax = bbox
    i, = np.where((y >= ymin) & (y <= ymax))
    j, = np.where((x >= xmin) & (x <= xmax))
    return (i[0], i[-1]+1, j[0], j[-1]+1)


def get_track_id(time_, tmax=10, years=False):
    """Partition time array into segments with breaks > tmax.

    Returns an array w/unique identifiers for each segment.

    Args:
        tmax: break interval in secs.
        time_: time var in secs (default) or years (years=True).
    """
    time = time_.copy()
    if years: time *= 3.154e7  # year -> sec
    n, trk = 0, np.zeros(time.shape)
    for k in xrange(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax: n += 1
        trk[k] = n
    return trk


def adjust_tracks(z, trk, median=False):
    """Remove offset from each individual track (FOR TESTING ONLY)."""
    # Get global mean
    if median:
        ref_mean = np.nanmedian(z)
    else:
        ref_mean = np.nanmean(z)
    # Remove track offsets
    for k in np.unique(trk): 
        i_trk, = np.where(trk == k)
        z_trk = z[i_trk]
        if median:
            trk_mean = np.nanmedian(z_trk)
        else:
             trk_mean = np.nanmean(z_trk)
        # Bring each track to global mean
        z[i_trk] -= (trk_mean + ref_mean)
    return z


""" Compiled functions. """

@jit(nopython=True)
def add_off_diag_err(A, B, C, err):
    """Add correlated (off-diagonal) errors to C.

    If i,j belong to the same track (aij == bij)
    and they are not in the diagonal (i != j), then:
        cij += sigma
    """
    M, N = A.shape
    for i in range(M):
        for j in range(N):
            aij = A[i,j]
            bij = B[i,j]
            if i != j and aij == bij:
                C[i,j] += err


@jit(nopython=True)
def space_dist_grid_data(x0, y0, x, y):
    """ Compute spatial distance between prediction pt and obs. """
    return np.sqrt((x-x0) * (x-x0) + (y-y0) * (y-y0))


@jit(nopython=True)
def time_dist_grid_data(t0, tc):
    """ Compute time distance between prediction pt and obs. """
    return np.abs(tc - t0)


def space_dist_data_data(x, y):
    """ Compute spatial distances between obs. """
    X = np.column_stack((x, y))
    return cdist(X, X, "euclidean")


def time_dist_data_data(t):
    """ Compute time distances between obs. """
    X = t[:,np.newaxis]
    return np.abs(cdist(X, X, "euclidean"))

#-------------

""" Helper functions. """

def subset_data(t, x, y, z, e, k, tlim=(1995.25, 1995.5),
                xlim=(-1, 1), ylim=(-1, 1)):
    """ Subset data domain (add NaNs to undesired values). """
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    ii, = np.where(tt & xx & yy)
    return t[ii], x[ii], y[ii], z[ii], e[ii], k[ii]


def remove_invalid(z, variables):
    """Filter NaNs using z var."""
    ii, = np.where(np.isfinite(z))
    return [v[ii] for v in variables]


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def has_alpha(string):
    """Return True if any char is alphabetic."""
    return any(c.isalpha() for c in string)


def load_data(ifile, xvar, yvar, zvar, evar, tvar, kvar, step=1):
    with h5py.File(ifile, 'r') as f:
        lon = f[xvar][::step]
        lat = f[yvar][::step]
        obs = f[zvar][::step]
        sigma = f[evar][::step] if has_alpha(evar) else np.full_like(obs, float(evar))
        time = f[tvar][::step] if has_alpha(tvar) else np.full_like(obs, float(tvar))
        trk = f[kvar][::step] if kvar is not None else np.full_like(obs, 0)
    return lon, lat, obs, sigma, time, trk

#---------------------------------------------------

""" Main interpolation functions. """

def get_cell_data(data, (x0,y0), radius, Tree):
    """ Get data within search radius (inversion cell). """
    i_cell = Tree.query_ball_point((x0, y0), radius)
    return [d[i_cell] for d in data]


def rand(x, n):
    """ Draw random samples from array. """
    if len(x) > n:
        return np.random.choice(x, int(n), replace=False)
    else:
        return x  # return original


def sample_sectors(x, y, x0, y0, n_samples=0.5):
    """ Sample data at random within sectors.
    
    If n_samples == float -> total percent of sampled data.
    If n_samples == int -> number of samples drawn per sector.
    """
    # Compute angle to data points
    theta = (180./np.pi) * np.arctan2(y-y0, x-x0) + 180

    # Get index for data in 8 sectors
    i_sec1, = np.where((theta > 0) & (theta < 45))
    i_sec2, = np.where((theta > 45) & (theta < 90))
    i_sec3, = np.where((theta > 90) & (theta < 135))
    i_sec4, = np.where((theta > 135) & (theta < 180))
    i_sec5, = np.where((theta > 180) & (theta < 225))
    i_sec6, = np.where((theta > 225) & (theta < 270))
    i_sec7, = np.where((theta > 270) & (theta < 315))
    i_sec8, = np.where((theta > 315) & (theta < 360))

    # Percent of total data
    if isinstance(n_samples, float):
        p_samples = len(x) * n_samples
        n_samples = int(np.ceil(p_samples/8.))

    # Draw random samples from each sector
    i_sec1 = rand(i_sec1, n_samples)
    i_sec2 = rand(i_sec2, n_samples)
    i_sec3 = rand(i_sec3, n_samples)
    i_sec4 = rand(i_sec4, n_samples)
    i_sec5 = rand(i_sec5, n_samples)
    i_sec6 = rand(i_sec6, n_samples)
    i_sec7 = rand(i_sec7, n_samples)
    i_sec8 = rand(i_sec8, n_samples)

    return np.r_[i_sec1, i_sec2, i_sec3, i_sec4,
                 i_sec5, i_sec6, i_sec7, i_sec8] 


def ointerp2d(data, (xi,yi), radius=1, t_ref=None, sigma_corr=None,
              min_obs=10, s=0, R=0, tau=0, n_samples=None):
    """Optimal Interpolation of spatial data to a 2d grid."""

    t, x, y, z, e, k = data

    # Construct cKDTree with all data available
    Tree = cKDTree(np.column_stack((x, y)))

    # Use mean time as the ref time
    #if t_ref is None: t_ref = t.min() + (t.max()-t.min())/2.
    if t_ref is None: t_ref = np.nanmean(t)                     ##FIXME: Check this.

    # Create output containers for predictions
    zi = np.full_like(xi, np.nan)
    ei = np.full_like(xi, np.nan)
    ni = np.full_like(xi, np.nan)
    ti = np.full_like(xi, np.nan)
    si = np.full_like(xi, np.nan)
    di = np.full_like(xi, np.nan)

    # Enter prediction loop
    for i_node in xrange(xi.shape[0]):

        x0, y0 = xi[i_node], yi[i_node]  # prediction pt (grid node)

        # Get data within inversion cell
        tc, xc, yc, zc, ec, kc = get_cell_data(data, (x0,y0), radius, Tree)

        if 1:
            # Quick outlier editing
            zc[np.abs(zc-np.nanmedian(zc))>mad_std(zc)*3] = np.nan
            tc, xc, yc, zc, ec, kc = \
                    remove_invalid(zc, [tc, xc, yc, zc, ec, kc])

        if len(zc) < min_obs: continue

        if n_samples is not None:
            # Draw random sector samples
            i_sec = sample_sectors(xc, yc, x0, y0, n_samples)
            tc, xc, yc, zc, ec, kc = [d[i_sec] \
                    for d in [tc, xc, yc, zc, ec, kc]]
            tc, xc, yc, zc, ec, kc = \
                    remove_invalid(zc, [tc, xc, yc, zc, ec, kc])

        if len(zc) < min_obs: continue

        # Plot individual tracks within search radius
        if 0:
            if (i_node % 500 == 0):
                print 'Node:', i_node
                print 'Trk#:', np.unique(kc)
                plt.figure()
                plt.scatter(x, y, c='0.5', s=5, rasterized=True)
                plt.scatter(xc, yc, c=kc, s=30, cmap='tab10')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.figure()
                plt.scatter(np.hypot(xc, yc), zc, c=kc, s=30, cmap='tab10')
                plt.xlabel('Distance along track (m)')
                plt.ylabel('Input variable (units)')
                plt.show()
            continue

        """ Compute space-time distances. """

        # Spatial distance model-data (x-dist to prediction pt) 
        Dxj = space_dist_grid_data(x0, y0, xc, yc)  # -> vec

        # Temporal distance model-data (t-dist to prediction pt)
        Dxk = time_dist_grid_data(t_ref, tc)  # -> vec

        # Spatial distance data-data (x-dist between data pts) 
        Dij = space_dist_data_data(xc, yc)  # -> mat

        # Temporal distance data-data (t-dist between data pts) 
        Dik = time_dist_data_data(tc)  # -> mat
        
        """ Build covariance matrices. """

        m0 = np.nanmedian(zc)         # local median (robust) 
        c0 = np.nanvar(zc)            # local variance of data
        c0_mod = covmodel(0, s, R)    # global variance of data
        #c0_mod = covmodel(0, 0, s, R, tau)

        # Scaling factor to convert: global cov -> local cov
        ##scale = 1. #c0/c0_mod       ##NOTE: Not using scaling

        # Covariance vector: model-data 
        #Cxj = covmodel(Dxj, Dxk, s, R, tau) * scale  ##NOTE: Not using time dependence for now!
        Cxj = covmodel(Dxj, s, R)

        # Covariance matrix: data-data 
        #Cij = covmodel(Dij, Dik, s, R, tau) * scale  ##NOTE: Not using time dependence for now!
        Cij = covmodel(Dij, s, R)

        '''
        print 0., Dxj.min(), Dxj.max()
        print covmodel(np.array([0., Dxj.min(), Dxj.max()]), s, R)
        print Dij.min(), Dij.max()
        print covmodel(np.array([Dij.min(), Dij.max()]), s, R)
        '''

        ######

        # Plot covarainces
        if 0 and (i_node % 1000 == 0):
            title1 = 'Covariance model-data'
            title2 = 'Covariance data-data'

            # Cov mat -> Corr mat
            if 1:
                D = np.diag(np.sqrt(np.diag(Cij)))
                Dinv = np.linalg.inv(D)
                Rij = np.dot(np.dot(Dinv, Cij), Dinv.T)
                Cij = Rij

                title2 = 'Correlation data-data'

            # Plot cov values vs space-time distance
            plt.figure(figsize=(12,5))

            plt.subplot(121)
            plt.scatter(Dxj, Cxj, c=Dxk, s=30, cmap='hot',
                        linewidth=.5, edgecolor='k', alpha=.7)
            plt.ylim(Cxj.min(), Cxj.max())
            plt.colorbar(label='Temporal distance (yr)')
            plt.xlabel('Spatial distance (m)')
            plt.ylabel('Covariance or Correlation')
            plt.title(title1)

            plt.subplot(122)
            plt.scatter(Dij, Cij, c=Dik, s=30, cmap='hot',
                        linewidth=.5, edgecolor='k', alpha=.7)
            plt.colorbar(label='Temporal distance (yr)')
            plt.xlabel('Spatial distance (m)')
            plt.title(title2)

            plt.figure()
            plt.scatter(xc, yc, s=30, c=tc, cmap='hot',
                        linewidth=.5, edgecolor='k')
            plt.colorbar(label='Time (yr)')
            plt.scatter(x0, y0, s=60, c='red')

            plt.show()
            continue

        """ Build error matrix. """

        # Uncorrelated errors
        # (diagonal -> variance of uncorrelated white noise)
        Nij = np.diag(ec*ec)  

        # Matrices with track id for each data point
        Kx, Ky = np.meshgrid(kc, kc)
        kuni = np.unique(Kx)

        if 0:
            # Plot error matrix w/diagonal only
            plt.matshow(Nij)
            plt.colorbar(shrink=.65, location='bottom', label='sigma^2')

        # Correlated errors
        # (off-diagonal => variance of along-track long-wavelength error)
        add_off_diag_err(Kx, Ky, Nij, sigma_corr**2)

        if 0:
            # Plot error matrix w/off-diagonal entries
            plt.matshow(Nij)
            plt.colorbar(shrink=.65, location='bottom', label='sigma^2')
            plt.show()
            continue

        """ Solve the Least-Squares system for the inversion cell. """

        if len(zc) < min_obs or len(Cxj) != len(zc): continue

        # Augmented data-cov matrix w/errors
        Aij = Cij + Nij

        # Matrix inversion of: Cxj * Aij^(-1)
        CxjAiji = np.linalg.solve(Aij.T, Cxj.T)

        # Predicted value
        zi[i_node] = np.dot(CxjAiji, zc) + (1 - np.sum(CxjAiji)) * m0
        
        # Predicted error -> std
        ei[i_node] = np.sqrt(np.abs(c0 - np.dot(CxjAiji, Cxj.T)))
        
        # Number of obs used for prediction    
        ni[i_node] = len(zc)

        # Reference time of prediction
        ti[i_node] = tc.mean()

        # Time span of obs used for prediction 
        si[i_node] = tc.max() - tc.min()

        # Mean distance to obs
        di[i_node] = 1e-3 * Dxj.mean()

        # Print progress to terminal
        if (i_node % 500) == 0:
            print 'node:', i_node, '/', len(xi)
            print 'pred:', round(zi[i_node], 2)
            print 'pstd:', round(ei[i_node], 4)
            print 'time:', round(ti[i_node], 2)
            print 'span:', round(si[i_node], 2)
            print 'davg:', round(di[i_node], 2) 
            print 'nobs:', ni[i_node]
            print ''

    return zi, ei, ni, ti, si, di



#def to_grid(arrs, shape):
#    return [np.flipud(a.reshape(shape)) for a in arrs]          ##FIXME: Check: I think the 'flipud' is not needed... it is messing things up!

def to_grid(arrs, shape):
    return [a.reshape(shape) for a in arrs]  # 1d -> 2d


def crop_tile(arrs, x, y, ifile):
    """Crop tile according bbox in file name."""
    bbox = get_bbox(ifile)
    (i1,i2,j1,j2) = get_limits(x, y, bbox)
    return [arr[i1:i2,j1:j2] for arr in arrs]


def save_data(ofile, variables, names):
    with h5py.File(ofile, 'w') as f:
        for var, name in zip(variables, names): f[name] = var


def main(ifile, args):

    print ifile

    #ifile = args.ifile[0]
    ofile = args.ofile[0]
    suffix = args.suffix[0]
    bbox = args.bbox[:]
    vnames = args.vnames[:]
    tvar = args.tvar[0]
    kvar = args.kvar[0]
    sigma_corr = args.sigmacorr[0]
    dx = args.dxy[0] * 1e3
    dy = args.dxy[1] * 1e3
    radius = args.radius[0] * 1e3
    tlim = args.tlim[:]
    xlim = args.xlim[:]
    ylim = args.ylim[:]
    proj = args.proj[0]


    ##FIXME: Remove, this is temporary
    if os.path.exists(ifile + suffix):
        print 'FILE EXISTS... skipping!'
        return


    min_obs = MIN_OBS

    print_args(args)

    startTime = datetime.now()

    xvar, yvar, zvar, evar = vnames

    # Load data in-memory
    lon, lat, obs, err, time, trk = \
            load_data(ifile, xvar, yvar, zvar, evar, tvar, kvar, step=1)

    # If no corr error given, uses half the (median) random error**2 (variance)
    if not sigma_corr: sigma_corr = np.sqrt(0.5 * np.nanmedian(err)**2)
         
    ##FIXME: Only use this for DEM interpolation using RA data
    if DEM_RA:
        ii = (lat >= -81.5)
        time = time[ii]
        lon = lon[ii]
        lat = lat[ii]
        obs = obs[ii]
        err = err[ii]
        trk = trk[ii]

    if None in tlim: tlim = [np.nanmin(time), np.nanmax(time)]
    if None in xlim: xlim = [np.nanmin(lon), np.nanmax(lon)]
    if None in ylim: ylim = [np.nanmin(lat), np.nanmax(lat)]

    if ofile is None: ofile = ifile + suffix

    # Remove NaNs
    lon, lat, obs, err, time, trk = \
            remove_invalid(obs, [lon, lat, obs, err, time, trk])

    if len(obs) < MIN_OBS:
        print 'no sufficient data points!'
        return

    # Convert to stereo coordinates
    x, y = transform_coord(4326, proj, lon, lat)

    # Assign a track ID to each data point
    if np.sum(trk) == 0: trk = get_track_id(time, tmax=100, years=True)  ##TODO: User should set tmax and years!!!

    #--- Plot ------------------------------------
    if 0:
        # Test track separation
        plt.figure()
        std = .15 #np.nanstd(obs)/2.
        plt.scatter(x, y, c=obs, s=5, rasterized=True,
                vmin=-std, vmax=std, cmap=plt.cm.RdBu)
        plt.colorbar()

        plt.figure()
        trk_unique = np.unique(trk)
        for k in trk_unique:
            ii, = np.where(k == trk)
            x_ = x[ii]
            y_ = y[ii]

            # Plot all tracks
            plt.plot(x_, y_, '.', rasterized=True)

        plt.show()
        sys.exit()
    #---------------------------------------------

    """ Set prediction grid. """

    # Set spatial limits of prediction grid 
    if bbox:
        xmin, xmax, ymin, ymax = bbox
    elif 'bbox' in ifile:
        xmin, xmax, ymin, ymax = get_bbox(ifile, key='bbox')
    else:
        xmin, xmax, ymin, ymax = (x.min() + radius), (x.max() - radius), \
                                 (y.min() + radius), (y.max() - radius)

    ##TODO: In the future, pass the prediction grid here
    ## and alow for variable search radius?

    # Generate 2D prediction grid     
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)
    xi, yi = Xi.ravel(), Yi.ravel()

    # Convert to stereographic coord.
    #if np.abs(ymax) < 100: xi, yi = transform_coord(projGeo, projGrd, xi, yi)  ##FIXME: Check why this is being triggered on x/y data?!

    """ Interpolate data. """

    zi, ei, ni, ti, si, di = \
            ointerp2d([time, x, y, obs, err, trk], (xi,yi), t_ref=None,
                      radius=radius, sigma_corr=sigma_corr, min_obs=min_obs,
                      s=s, R=R, tau=0, n_samples=N_SAMPLES)

    Xi, Yi, Zi, Ei, Ni, Ti, Si, Di = \
            to_grid([xi, yi, zi, ei, ni, ti, si, di], Xi.shape)  # 1d -> 2d


    try:
        Xi, Yi, Zi, Ei, Ni, Ti, Si, Di = \
                crop_tile([Xi, Yi, Zi, Ei, Ni, Ti, Si, Di], 
                          Xi[0,:], Yi[:,0], ifile)  # grid -> tile
    except:
        print 'No BBOX in file name... skipping cropping!'
        pass

    #--- Plot ------------------------------------
    if 0:
        # Plot interpolated grid
        from scipy import ndimage as ndi
        #vmin, vmax, cmap = -40, 40, 'terrain'
        vmin, vmax, cmap = -.5, .5, 'RdBu'

        plt.figure()
        plt.scatter(x, y, c=obs, s=10, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Original points')
        plt.figure()
        plt.scatter(xi, yi, c=zi, s=5, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Interpolated points')
        plt.figure()
        plt.scatter(xi, yi, c=ei, s=5, vmin=0, vmax=1, cmap='inferno_r')
        plt.colorbar()
        plt.title('Interpolation error')
        plt.figure()
        plt.scatter(xi, yi, c=ni, s=5, vmin=None, vmax=None, cmap='Blues')
        plt.colorbar()
        plt.title('Number of observations')
        
        plt.figure()
        Zi = ndi.median_filter(Zi, 3)
        plt.imshow(Zi, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Interpolated grid')

        plt.show()
        #sys.exit()
    #---------------------------------------------

    """ Save interpolated fields. """

    ti, xi, yi = [np.nanmean(time)], Xi[0,:], Yi[:,0] 

    save_data(ofile, [ti,xi,yi,Zi,Ei,Ni,Ti,Si,Di],
              ['t_year', 'x','y',zvar,zvar+'_err',
               'n_obs','t_ref','t_span','d_mean'])

    print 'Mean time to interpolate field:', ti
    print 'Execution time: '+ str(datetime.now()-startTime)
    print 'output ->', ofile


# Get command line args
args = get_args() 
files = args.ifile[:]
njobs = args.njobs[0]

if njobs == 1:
    print 'running serial code ...'
    [main(f, args) for f in files]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=1)(delayed(main)(f, args) for f in files)
            

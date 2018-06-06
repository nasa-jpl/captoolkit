#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial Optimal Interpolation using modeled covariance function.

Example:
    python ointerp3d.py ~/data/ers1/floating/filt_scat_det/joined_pts_a.h5_ross -d 3 3 0.25 -r 15 -k 0.125 -e .2 -v t_year lon lat h_res None -x -152 0 -t 1995.0 1995.25

"""

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

#-----------------------------------------------------------

""" Covariance models. """

# Modeled parameters
#s, R = [0.831433532, 1536.21967]  # gauss
s, R = [0.85417087, 655.42437697]  # markov  ##NOTE: Seems to be the best performing
#s, R = [0.720520448, 1084.11029]  # generic

# Half the time-range of horizontal layers
tau = 1.5/12.

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

#covmodel = markov
covmodel = covxt

#----------------------------------------------------------

def get_args():
    """ Get command-line arguments. """

    des = 'Optimal Interpolation of space-time data'
    parser = argparse.ArgumentParser(description=des)

    parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='name of i-file, numpy binary or ascii (for binary ".npy")')

    parser.add_argument(
            '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
            help='name of o-file, numpy binary or ascii (for binary ".npy")',
            default=[None])

    parser.add_argument(
            '-i', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geograph. region (deg or m), optional'),
            default=[],)

    parser.add_argument(
            '-x', metavar=('x1', 'x2'), dest='xlim', type=float, nargs=2,
            help=('x/lon span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-y', metavar=('y1', 'y2'), dest='ylim', type=float, nargs=2,
            help=('y/lat span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-t', metavar=('t1', 't2'), dest='tlim', type=float, nargs=2,
            help=('time span to subest for covariance calc'),
            default=[None],)

    parser.add_argument(
            '-d', metavar=('dx','dy', 'dt'), dest='dxyt', type=float, nargs=3,
            help=('grid resolution in space and time (km km yr)'),
            default=[1, 1, 1],)

    parser.add_argument(
            '-n', metavar='nobs', dest='nobs', type=int, nargs=1,
            help=('number of obs. for each quadrant'),
            default=[100],)

    parser.add_argument(
            '-r', metavar='radius', dest='radius', type=float, nargs=1,
            help=('spatial search radius for each inversion cell (km)'),
            default=[1],)

    parser.add_argument(
            '-k', metavar='delta', dest='delta', type=float, nargs=1,
            help=('time range of data for each horizontal layer (yr)'),
            default=[1],)

    parser.add_argument(
            '-e', metavar='sigma', dest='sigma', type=float, nargs=1,
            help=('rms noise of obs. (m)'),
            default=[0],)

    parser.add_argument(
            '-v', metavar=('time', 'lon','lat', 'obs', 'err'), dest='vnames',
            type=str, nargs=5,
            help=('name of t/x/y/z/e variables in the HDF5. If err=None, skip'),
            default=[None], required=True)

    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

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


def spatial_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Generate a regular grid. """

    # Setup grid dimensions
    Nx = int((np.abs(xmax - xmin)) / dx) + 1
    Ny = int((np.abs(ymax - ymin)) / dy) + 1

    # Initiate lat/lon vectors for grid
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)

    # Construct output grid-coordinates
    return np.meshgrid(x, y)


def time_grid(tmin, tmax, dt):
    Nt = int((np.abs(tmax - tmin)) / dt) + 1
    return np.linspace(tmin, tmax, Nt)


def get_track_id(time_, tmax=1, years=False):
    """
    Partition time array into segments with breaks > tmax.

    Returns an array w/unique identifiers for each segment.
    """
    time = time_.copy()
    if years:
        time *= 3.154e7  # year -> sec
    n = 0
    trk = np.zeros(time.shape)
    for k in xrange(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax:
            n += 1
        trk[k] = n
    return trk


def adjust_tracks(z, trk, median=False):
    """ Remove offset from each individual track. """
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
def add_offdiag_err(A, B, C, err):
    """
    Add correlated (off-diagonal) errors to C.

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

def subset_data(t, x, y, z, e,
        tlim=(1995.25, 1995.5), xlim=(-1, 1), ylim=(-1, 1)):
    """ Subset data domain (add NaNs). """
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    ii, = np.where(tt & xx & yy)
    return t[ii], x[ii], y[ii], z[ii], e[ii]


def remove_invalid(t, x, y, z, e):
    """ Remove NaNs and Zeros. """
    #ii, = np.where((z != 0) & ~np.isnan(z))
    ii, = np.where(~np.isnan(z))
    return t[ii], x[ii], y[ii], z[ii], e[ii]


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def iterfilt(x, xmin, xmax, tol, alpha):
    """ Iterative outlier filter """
    
    # Set default value
    tau = 100.0
    
    # Remove data outside selected range
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan
    
    # Initiate counter
    k = 0
    
    # Outlier rejection loop
    while tau > tol:
        
        # Compute initial rms
        rmse_b = mad_std(x)
        
        # Compute residuals
        dh_abs = np.abs(x - np.nanmedian(x))
        
        # Index of outliers
        io = dh_abs > alpha * rmse_b
        
        # Compute edited rms
        rmse_a = mad_std(x[~io])
        
        # Determine rms reduction
        tau = 100.0 * (rmse_b - rmse_a) / rmse_a
        
        # Remove data if true
        if tau > tol or k == 0:
            
            # Set outliers to NaN
            x[io] = np.nan
            
            # Update counter
            k += 1
    
    return x


def load_data(ifile, vnames, step=2):

    tvar = vnames[0]
    xvar = vnames[1]
    yvar = vnames[2]
    zvar = vnames[3]
    evar = vnames[4]

    with h5py.File(ifile, 'r') as f:

        time = f[tvar][::step]
        lon = f[xvar][::step]
        lat = f[yvar][::step]
        obs = f[zvar][::step]
        err = f[evar][::step] if evar != 'None' else np.full_like(obs, np.nan)

        # Remove uncorrected data (this should be done before applying this code)
        if 1:
            b = f['h_bs'][::step] 
            obs[b==0] = np.nan
            obs[np.isnan(b)] = np.nan

    return time, lon, lat, obs, err


#---------------------------------------------------

""" Main interpolation functions. """

def get_data_cell(data, Tree, x0, y0, radius):
    """ Get data within search radius. """
    i_cell = Tree.query_ball_point((x0, y0), radius)
    data_cell = [var[i_cell] for var in data]
    return data_cell


def ointerp2d(data, (xi,yi), t0=None, radius=1,
        sigma2=None, sigma2c=None, min_obs=10):
    """ Optimal Interpolation of spatial data on a x/y grid. """

    # t/x/y coords of data
    t, x, y = data[0], data[1], data[2]

    # Construct cKDTree with all data available
    Tree = cKDTree(np.column_stack((x, y)))

    if t0 is None:
        # Use mean time as the time ref 
        t0 = t.min() + (t.max()-t.min())/2.

    # Create output containers for predictions
    zi = np.full_like(xi, np.nan)
    ei = np.full_like(xi, np.nan)
    ni = np.full_like(xi, np.nan)

    # Enter prediction loop
    for i_node in xrange(xi.shape[0]):

        # Get prediction pt (grid node)
        x0 = xi[i_node]
        y0 = yi[i_node]

        # Get data within inversion cell
        tc, xc, yc, zc, ec, kc = get_data_cell(data, Tree, x0, y0, radius)

        # Quick outlier editing
        zmin, zmax, tol, thres = [-3, 3, 5, 3]
        i_filt = ~np.isnan(iterfilt(zc.copy(), zmin, zmax, tol, thres))
        tc = tc[i_filt]
        xc = xc[i_filt]
        yc = yc[i_filt]
        zc = zc[i_filt]
        ec = ec[i_filt]
        kc = kc[i_filt]

        # Test for empty cell
        if len(zc) < min_obs:
            continue

        #zc = adjust_tracks(zc, kc, median=False)  ##FIXME: Adjust at cell level

        # Plot data within search radius
        if 0:

            plt.figure()
            plt.scatter(xc, yc, c=zc, s=50, cmap=plt.cm.RdBu)

            plt.figure()
            k_unique = np.unique(kc)

            for ki in k_unique:

                i_trk, = np.where(ki == kc)

                plt.plot(np.hypot(xc[i_trk], yc[i_trk]), zc[i_trk], 'o')
                plt.xlabel('Distance along track (m)')

                zc[i_trk] -= np.nanmean(zc[i_trk])

                plt.plot(np.hypot(xc[i_trk], yc[i_trk]), zc[i_trk], 'x')
                plt.xlabel('Distance along track (m)')

            plt.figure()
            plt.scatter(xc, yc, c=zc, s=50, cmap=plt.cm.RdBu)

            plt.show()
            continue

        """ Compute space-time distances. """

        # Compute spatial distance between prediction pt and obs in search radius
        Dxj = space_dist_grid_data(x0, y0, xc, yc)  # -> vec

        # Compute time distance between prediction pt and obs in search radius
        Dxk = time_dist_grid_data(t0, tc)  # -> vec

        # Compute spatial distances between obs in search radius 
        Dij = space_dist_data_data(xc, yc)  # -> mat

        # Compute time distances between obs in search radius 
        Dik = time_dist_data_data(tc)  # -> mat
        
        """ Build covariance matrices. """

        # Estimate local median (robust) and local variance of data
        m0 = np.nanmedian(zc)
        c0 = np.nanvar(zc) 

        # Scaling factor to convert: global cov -> local cov
        scale = c0/covmodel(0, 0, s, R, tau)
        
        # Covariance vector: model-data 
        Cxj = covmodel(Dxj, Dxk, s, R, tau) * scale  ##FIXME
        #Cxj = covmodel(Dxj, 0, s, R, tau) * scale
        
        # Covariance matrix: data-data 
        #Cij = covmodel(Dij, Dik, s, R, tau) * scale  ##FIXME
        Cij = covmodel(Dij, 0., s, R, tau) * scale

        ######

        if 0:
            # Plot cov values vs space-time distance
            plt.figure()
            plt.scatter(Dxj, Cxj, c=Dxk, s=20, cmap=plt.cm.hot)
            plt.colorbar()
            plt.title('Cov(grid,data) vs space-time dist')
            plt.xlabel('Spatial distance')
            plt.ylabel('Covariance')
            plt.figure()
            plt.scatter(Dij, Cij, c=Dik, s=20, cmap=plt.cm.hot)
            plt.colorbar()
            plt.title('Cov(data,data) vs space-time dist')
            plt.xlabel('Spatial distance')
            plt.ylabel('Covariance')
            plt.show()
            continue

        """ Build error matrix. """

        # Uncorrelated errors
        # (diagonal => variance of uncorrelated white noise)
        Nij = np.diag(ec)  

        # Matrices with track id for each data point
        Kx, Ky = np.meshgrid(kc, kc)

        # Plot error matrix w/diagonal only
        if 0:
            plt.figure()
            plt.imshow(Nij)

        # Correlated errors
        # (off-diagonal => variance of along-track long-wavelength error)
        add_offdiag_err(Kx, Ky, Nij, sigma2c)

        # Plot error matrix w/off-diagonal entries
        if 0:
            plt.figure()
            plt.imshow(Nij)
            plt.show()
            continue

        """ Solve the Least-Squares system for the inversion cell. """

        # Augmented data-cov matrix w/errors
        Aij = Cij + Nij

        # Matrix inversion of: Cxj * Aij^(-1)
        CxjAiji = np.linalg.solve(Aij.T, Cxj.T)

        # Predicted value
        zi[i_node] = np.dot(CxjAiji, zc) + (1 - np.sum(CxjAiji)) * m0
        
        # Predicted error
        ei[i_node] = np.sqrt(np.abs(c0 - np.dot(CxjAiji, Cxj.T)))
        
        # Number of data used for prediction    
        ni[i_node] = len(zc)

        # Print progress to terminal
        if (i_node % 1000) == 0:
            print 'node: ', i_node, '/', len(xi)
            print 'pred: ', round(zi[i_node], 2)
            print 'npts: ', ni[i_node]
            print 'dmax: ', round(1e-3 * Dxj.max(), 2)
            print 'tlim: ', round(tc.min(), 2), round(tc.max(), 2)
            print ''

    return zi, ei, ni


def interp_slice(data_slice, args):
    """
    Wrapper around ointerp2d to reduce number of args -> dict(dict).
    
    data_slice: {t_slice: {'x': x, 'y': y, 'obs': obs, ..}}
    args: {'grid': grid, 'radius': r, 'sigma2': sigma2, ..}
    """
    t_slice = data_slice.keys()[0]    # float
    d_slice = data_slice.values()[0]  # dict

    time = d_slice['time']
    x = d_slice['x']
    y = d_slice['y']
    obs = d_slice['obs']
    err = d_slice['err']
    trk = d_slice['trk']

    xi, yi = args['grid']
    r = args['radius']
    s2 = args['sigma2']
    s2c = args['sigma2c']
    n = args['min_obs']

    obs -= np.nanmean(obs)                       ##FIXME: Remove this
    obs = adjust_tracks(obs, trk, median=False)  ##FIXME: Remove this

    interped_slice = {}

    # 2D Optimal Interpolation (loops through each grid node)
    if len(obs) >= n:

        zi, ei, ni = ointerp2d([time,x,y,obs,err,trk], [xi,yi],
                t0=t_slice, radius=r, sigma2=s2, sigma2c=s2c, min_obs=n)

        interped_slice = {t_slice: {'zi': zi.copy(),
                                    'ei': ei.copy(),
                                    'ni': ni.copy(),}}
    return interped_slice


def get_slice(data, t_slice, length, min_obs=1000):
    """ Get a slice of data points in time -> {t_slice: {d_slice}}. """
    time = data['time']
    t1, t2 = t_slice-length/2., t_slice+length/2.
    i_slice, = np.where( (time >= t1) & (time <= t2) )
    d_slice = {key:value[i_slice] for (key,value) in data.items()}
    return {t_slice: d_slice}


def get_slices(data, t_slices, length, min_obs=10):
    """ Run get_slice() for a sequence of times -> [dict1, dict2...]. """ 
    return [get_slice(data, t_i, length, min_obs) for t_i in t_slices]


def store_1d_to_3d(arr1d, arr3d, i_3d):
    """ Reshape 1d to 2d and store as slice in 3d array. """
    arr2d = np.flipud(arr1d.reshape(arr3d[:,:,0].shape))
    arr3d[:,:,i_3d] = arr2d


def store_slice(arrs1d, arrs3d, i_3d):
    """ Store a list w/1d arrays into a list w/3d arrays. """
    [store_1d_to_3d(a1d, a3d, i_3d) for (a1d,a3d) in zip(arrs1d, arrs3d)]


def store_slices(interped_slices, arrs3d, ti):
    """
    Store 1d arrs [zi,ei,ni] into 3d arrs [Z,E,N].
    
    interped_slices: [{t_slice: {'zi': zi, 'ei': ei, 'ni': ni}}, ..]
    arrs3d: [Z, E, N]
    ti: [t0, t1, t2, ..]
    """
    for intslice in interped_slices:

        if len(intslice) == 0:
            continue

        t_slice = intslice.keys()[0]
        d_slice = intslice.values()[0]

        arrs1d = [d_slice['zi'], d_slice['ei'], d_slice['ni']]

        i_3d = np.argmin(np.abs(ti-t_slice))

        store_slice(arrs1d, arrs3d, i_3d)


def save_data(ofile, variables, names):
    with h5py.File(ofile, 'w') as f:
        for var,name in zip(variables, names):
            f[name] = var


def main():

    # Parser argument to variable
    args = get_args() 

    # Read input from terminal
    ifile = args.ifile[0]
    ofile = args.ofile[0]
    bbox = args.bbox[:]
    vnames = args.vnames[:]
    tlim = args.tlim[:]
    xlim = args.xlim[:]
    ylim = args.ylim[:]
    dx = args.dxyt[0] * 1e3
    dy = args.dxyt[1] * 1e3
    dt = args.dxyt[2]
    radius = args.radius[0] * 1e3
    delta = args.delta[0]
    sigma = args.sigma[0]
    proj = args.proj[0]

    # Minimum number of pts per cell
    min_obs = 10

    # Print parameters to screen
    print_args(args)

    # Start timing of script
    startTime = datetime.now()

    # Load data into memory
    time, lon, lat, obs, err = load_data(ifile, vnames, step=2)

    if None in tlim:
        tlim = [np.nanmin(time), np.nanmax(time)]

    if None in xlim:
        xlim = [np.nanmin(lon), np.nanmax(lon)]

    if None in ylim:
        ylim = [np.nanmin(lat), np.nanmax(lat)]

    if ofile is None:
        ofile = ifile + '_interp'

    # Include the 'delta time' and 'search radius' for subsetting the data 
    tlim_ = tlim[:]
    xlim_ = xlim[:]
    ylim_ = ylim[:]
    tlim_[0] -= delta/2.
    tlim_[1] += delta/2.
    xlim_[0] -= radius/100000.
    xlim_[1] += radius/100000.
    ylim_[0] -= radius/100000.
    ylim_[1] += radius/100000.

    # Subset data in space and time
    time, lon, lat, obs, err = subset_data(
            time, lon, lat, obs, err, tlim=tlim_, xlim=xlim_, ylim=ylim_)

    # Remove NaNs
    time, lon, lat, obs, err = remove_invalid(time, lon, lat, obs, err)

    if len(obs) < 100:
        print 'not sufficient data points!'
        sys.exit()

    # Convert to stereo coordinates
    x, y = transform_coord(4326, proj, lon, lat)

    # Assign a track ID to each data point
    trk = get_track_id(time, tmax=100, years=True)

    #obs -= np.nanmean(obs)
    #obs = adjust_tracks(obs, trk, median=False)  ##FIXME

    #--- Plot ------------------------------------
    if 0:
        plt.figure()
        std = np.nanstd(obs)/2.
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
            '''
            # Plot individual track profiles
            plt.plot(np.hypot(x, y), obs, '.')
            plt.plot(np.hypot(xsub_, ysub_), obssub_, '.')
            plt.show()
            '''
        plt.show()
        print time
        sys.exit()
    #---------------------------------------------

    """ Set prediction grid. """

    # Set spatial limits of prediction grid 
    if len(bbox) == 6:
        # Extract bounding box elements
        xmin, xmax, ymin, ymax = bbox
    else:
        # Create bounding box limits
        xmin, xmax, ymin, ymax = (x.min() - 10.*dx), (x.max() + 10.*dx), \
                                 (y.min() - 10.*dy), (y.max() + 10.*dy)

    # Set time limits of prediction grid
    tmin, tmax = tlim

    ##TODO: In the future, pass the prediction grid here

    # Generate 2D prediction grid     
    Xi, Yi = spatial_grid(xmin, xmax, ymin, ymax, dx, dy)

    # Flatten prediction grid
    xi, yi = Xi.ravel(), Yi.ravel()

    # Generate time steps for each horizontal slice 
    ti = time_grid(tmin, tmax, dt)

    # Geographical projection
    if np.abs(ymax) < 100:
        
        # Convert to stereographic coord.
        (xi, yi) = pyproj.transform(projGeo, projGrd, xi, yi)

    """ Set errors. """

    # Compute noise variance
    sigma2 = sigma * sigma

    # Compute Long-wavelength (along-track) error  ##FIXME: What's a good error?!!!
    sigma2c = sigma2 * 0.75

    # Build error array if not provided (all obs w/the same sigma) 
    if np.isnan(err).all():
        err[:] = sigma2

    """ Define data structures to be passed to each core. """

    # Full data to be sliced
    data = {}
    data['time'] = time
    data['x'] = x
    data['y'] = y
    data['obs'] = obs
    data['err'] = err
    data['trk'] = trk

    # Arguments passed to _all_ cores
    args = {}
    args['grid'] = (xi,yi)
    args['radius'] = radius
    args['sigma2'] = sigma2
    args['sigma2c'] = sigma2c
    args['min_obs'] = min_obs

    """ Create output containers. """

    # Output 3D arrays
    Z = np.full((Xi.shape[0],Xi.shape[1],ti.shape[0]), np.nan)
    E = np.full((Xi.shape[0],Xi.shape[1],ti.shape[0]), np.nan)
    N = np.full((Xi.shape[0],Xi.shape[1],ti.shape[0]), np.nan)

    """ Main processing. """

    # Get slices of data from point cloud 
    data_slices = get_slices(data, ti, delta, min_obs)

    # Interpolate slices into vertical layers
    if 0: 

        """ Sequetial """
        interped_slices = [interp_slice(d_i, args) for d_i in data_slices]

    else:

        """ Parallel """
        from dask import compute
        from dask.distributed import Client, LocalCluster

        # Create a local cluster for testing
        cluster = LocalCluster(n_workers=4, threads_per_worker=None, 
                scheduler_port=8002, diagnostics_port=8003)

        print 'scheduler address:', cluster

        # Connect to cluster
        client = Client(cluster)                    # local cluster
        #client = Client('scheduler-address:8786')  # remote cluster

        # Submit each process to each core
        future = [client.submit(interp_slice, d_i, args) for d_i in data_slices]

        # Gather resutls from each core
        interped_slices = client.gather(future)

    # Store vertical layers into 3d arrays 
    store_slices(interped_slices, [Z,E,N], ti)

    """ Save interpolated fields. """

    xx = np.flipud(Xi)
    yy = np.flipud(Yi)
    xi = xx[0,:]
    yi = yy[:,0]

    # Save prediction
    save_data(ofile, [ti,xi,yi,Z,E,N], ['t','x','y','z','e','n'])

    print 'output ->', ofile

    """ Plot interpolated fields. """

    if 0:

        from scipy import ndimage as ndi

        h = delta/2.

        for k in range(Z.shape[2]):

            print k 

            t_k = ti[k]
            print t_k
            i_slice, = np.where( (time >= t_k-h) & (time <= t_k+h) )
            tp = time[i_slice]
            xp = x[i_slice]
            yp = y[i_slice]
            zp = obs[i_slice]
            zp -= np.nanmean(zp)

            zz = Z[:,:,k]
            zz -= np.nanmean(zz)

            # Filter spatial field
            if 1:
                zz = ndi.median_filter(zz, 3)

            vmin = -.3
            vmax = .3

            plt.figure()
            plt.pcolormesh(xx, yy, zz, vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)
            plt.colorbar()

            plt.figure()
            plt.scatter(xp, yp, c=zp, s=1, vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)
            plt.colorbar()

        plt.show()


    # Print execution time of script
    print 'Execution time: '+ str(datetime.now()-startTime)


if __name__ == '__main__':
    main()

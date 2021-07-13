# -*- coding: utf-8 -*-
"""
Generate time-evolving DEMs.

Two ways:
    1. Projecting a ref DEM using fitted parameters: trend, accel, seasonal.
    2. Projecting a ref DEM using smoothed time series of residuals.

Input:
    - 2d DEM (any res)
    - 3d time series cube

Output:
    - 3d DEM

"""
import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
import pyresample as pr
import matplotlib.pyplot as plt

import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

#=== Edit =============================================

t_ref = 1994.0   # ref time of 2d DEM (see TIMESPANS.txt)

# DEM vars
d_xvar = 'x'
d_yvar = 'y'
d_hvar = 'height'
d_evar = 'height_err'
d_nvar = 'height_nobs'
d_dvar = None
d_avar = None

# Time series vars
s_xvar = 'x'
s_yvar = 'y'
s_tvar = 't_year'
s_hvar = 'h_res_filt'

# 2d DEM
fdem = ('/Users/paolofer/data/ers1/floating/latest/'
        'SECFIT_ALL_AD_PTS_d22_r1535_q2_fvar_rls.h5_interp_height')

# 3d height time series
fcube = '/Users/paolofer/data/ers1/floating/ointerp/ER1_CUBE.h5'

# 2d ice shelf mask
fmask = '/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5'

ofile = None

#======================================================

def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Generate DEM(t) from static height/trend/accel.')
    parser.add_argument(
            'files', metavar='files', type=str, nargs='+',
            help='file(s) containing height, trend, accel fields')
    parser.add_argument(
            '-o', metavar=('ofile'), dest='ofile', type=str, nargs=1,
            help=('output file name'),
            default=[None],)
    parser.add_argument(
            '-f', metavar=('fmask'), dest='fmask', type=str, nargs=1,
            help=('ice-shelf mask file name'),
            default=[fmask],)
    parser.add_argument(
            '-t', metavar=('t1','t2','tr'), dest='tspan', type=float, nargs=3,
            help=('min obs for filtering'),
            default=[t_beg, t_end, t_ref],)
    parser.add_argument(
            '-d', metavar=('dt'), dest='dt', type=float, nargs=1,
            help=('time step for DEM time series'),
            default=[dt],)
    parser.add_argument(
            '-m', metavar=('min_obs'), dest='minobs', type=int, nargs=1,
            help=('min obs for filtering'),
            default=[dt],)
    parser.add_argument(
            '-c', dest='cube', action='store_true', 
            help=('save results to a 3D cube -> single file'),
            default=False)
    parser.add_argument(
            '-a', dest='apply', action='store_true', 
            help=('cut-off data at lat > 81.5 (for standard RA) '),
            default=False)
    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def mad_std(x, axis=None):
    """Robust standard deviation (using MAD)."""
    return 1.4826 * np.nanmedian(np.abs(x-np.nanmedian(x, axis)), axis)


def model_mean_height(t_k, tref, height, trend,
                      accel=None, ampli=None, phase=None):
    """Reconstruct h(t) = h0 + h' dt + 0.5 h'' dt^2."""
    dt = t_k - tref
    if accel is None:
        return height + trend * dt + 0.5
    elif ampli is None:
        return height + trend * dt + 0.5 * accel * dt**2
    else:
        return height + trend * dt + 0.5 * accel * dt**2 \
                + ampli * np.sin(2*np.pi * dt + phase)


def model_inst_rate(t_k, tref, trend, accel):
    """Reconstruct dh/dt(t) = h' + h'' dt."""
    return trend + accel * (t_k - tref)


def geotiff_read(ifile, metaData):
    """Read raster from file."""
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
        X = trans[0] + (Xp+0.5)*trans[1] + (Yp+0.5)*trans[2]  #FIXME: bottleneck!
        Y = trans[3] + (Xp+0.5)*trans[4] + (Yp+0.5)*trans[5]
    if metaData == "P":
        xp = np.arange(Nx)
        yp = np.arange(Ny)
        (Xp, Yp) = np.meshgrid(xp,yp)
        X = trans[0] + Xp*trans[1] + Yp*trans[2]  #FIXME: bottleneck!
        Y = trans[3] + Xp*trans[4] + Yp*trans[5]
    band = file.GetRasterBand(1)
    Z = band.ReadAsArray()
    dx = np.abs(dx)
    dy = np.abs(dy)
    return X, Y, Z, dx, dy, proj


def interp2d(xd, yd, data, xq, yq, **kwargs):
    """Bilinear interpolation from grid."""
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


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Stereo AnIS (x/y):  3031
        Stereo GrIS (x/y):  3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    return pyproj.transform(proj1, proj2, x, y)


def sgolay1d(h, window=3, order=1, deriv=0, dt=1.0, mode='nearest', time=None):
    """Savitztky-Golay filter with support for NaNs

    If time is given, interpolate NaNs otherwise pad w/zeros.

    dt is spacing between samples.
    """
    h2 = h.copy()
    ii, = np.where(np.isnan(h2))
    jj, = np.where(np.isfinite(h2))
    if len(ii) > 0 and time is not None:
        h2[ii] = np.interp(time[ii], time[jj], h2[jj])
    elif len(ii) > 0:
        h2[ii] = 0
    else:
        pass
    h2 = savgol_filter(h2, window, order, deriv, delta=dt, mode=mode)
    return h2 


def filter_cube(t_cube, h_cube, window=3):
    for i in range(h_cube.shape[0]):
        for j in range(h_cube.shape[1]):
            y = h_cube[i,j,:]
            if sum(np.isfinite(y)) == 0: continue
            y_orig = y.copy()
            y = sgolay1d(y, window=window, order=1, deriv=0, time=t_cube)
            """
            plt.plot(t_cube, y_orig)
            plt.plot(t_cube, y)
            plt.show()
            """
            h_cube[i,j,: ] = y
    return h_cube


def regrid_dem(x_dem, y_dem, h_dem, x_cube, y_cube):
    """ Regrid height field (low res) onto velocity field (high res). """

    # Generate 2d coordinate grids
    X_dem, Y_dem = np.meshgrid(x_dem, y_dem)
    X_cube, Y_cube = np.meshgrid(x_cube, y_cube)

    # x/y -> lon/lat 
    lon2d_dem, lat2d_dem = transform_coord(3031, 4326, X_dem, Y_dem)
    lon2d_cube, lat2d_cube = transform_coord(3031, 4326, X_cube, Y_cube)

    orig_grid = pr.geometry.SwathDefinition(lons=lon2d_dem, lats=lat2d_dem)
    targ_grid = pr.geometry.SwathDefinition(lons=lon2d_cube, lats=lat2d_cube)

    h_dem[np.isnan(h_dem)] = 0.

    ##NOTE: Interp using inverse-distance weighting
    wf = lambda r: 1/r**2
    h_interp = pr.kd_tree.resample_custom(orig_grid, h_dem,
            targ_grid, radius_of_influence=10000, neighbours=10,
            weight_funcs=wf, fill_value=0.)

    return h_interp


def get_fields(fname, vnames):
    with h5py.File(fname, 'r') as f:
        fields = [f[k][:] for k in vnames if k in f]
    return fields


# Pass arguments 
'''
args = get_args()
#ifiles = args.files     
ifiles = [fdem]
ofile = args.ofile[0]  
fmask = args.fmask[0]  
#vnames = args.vnames[:]
t_beg = args.tspan[0] 
t_end = args.tspan[1] 
t_ref = args.tspan[2] 
#min_obs = args.minobs[0]
#dt = args.dt[0] 
cube = args.cube
RA = args.apply

print_args(args)
'''

if not ofile: ofile = fcube + '_DEM'

x_dem, y_dem, h_dem, e_dem, n_dem = get_fields(fdem, [d_xvar, d_yvar, d_hvar, d_evar, d_nvar])

x_cube, y_cube, t_cube, h_cube = get_fields(fcube, [s_xvar, s_yvar, s_tvar, s_hvar])

if 1:
    # Fill in NaN values w/Gaussian interpolation 
    kernel = Gaussian2DKernel(2)
    h_dem = interpolate_replace_nans(h_dem, kernel, boundary='fill', fill_value=np.nan)

if 1:
    h_dem = ndi.median_filter(h_dem, 3)

# Plot
if 0:
    plt.figure()
    plt.pcolormesh(x_dem, y_dem, h_dem, vmin=-20, vmax=200, cmap='RdBu')
    plt.title('Height (m)')
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(x_dem, y_dem, e_dem, vmin=0, vmax=25, cmap='RdBu')
    plt.title('Error (m)')
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(x_dem, y_dem, n_dem, vmin=0, vmax=500, cmap='RdBu')
    plt.title('N obs')
    plt.colorbar()
    plt.show()
    sys.exit()


print 'regridding dem ...'
if h_dem.shape != h_cube[:,:,0].shape:
    h_dem = regrid_dem(x_dem, y_dem, h_dem, x_cube, y_cube)

x_dem, y_dem = x_cube, y_cube

print 'filtering cube ...'
h_cube = filter_cube(t_cube, h_cube, window=5)

##NOTE: Ref filtered cube?

##NOTE: Replace cube NaNs for zeros?

h_dem = h_dem[:,:,None] + h_cube

print h_dem.shape
plt.figure()
plt.pcolormesh(x_dem, y_dem, h_dem[:,:,0], vmin=-20, vmax=200)

plt.figure()
plt.pcolormesh(x_dem, y_dem, h_dem[:,:,-1], vmin=-20, vmax=200)

dhdt = (h_dem[:,:,-1] - h_dem[:,:,0]) / (t_cube[-1]-t_cube[0])

plt.figure()
plt.pcolormesh(x_dem, y_dem, dhdt, vmin=-.5, vmax=.5, cmap='RdBu')

plt.show()


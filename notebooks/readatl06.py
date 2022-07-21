#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import pyproj
#import warnings
import argparse
#warnings.filterwarnings('ignore')
import h5py
import numpy as np
from astropy.time import Time
from scipy.ndimage import map_coordinates

try:
    from gdalconst import *
except ImportError as e:
    pass
try:
    from osgeo.gdalconst import *
except ImportError as e:
    pass
try:
    from osgeo import gdal, osr
except:
    print('No GDAL, proceeding without it!')


def segDifferenceFilter(dh_fit_dx, h_li, tol=2):

    dAT=20.

    if h_li.shape[0] < 3:
        mask = np.ones_like(h_li, dtype=bool)
        return mask

    EPplus  = h_li + dAT * dh_fit_dx
    EPminus = h_li - dAT * dh_fit_dx

    segDiff       = np.zeros_like(h_li)
    segDiff[0:-1] = np.abs(EPplus[0:-1] - h_li[1:])
    segDiff[1:]   = np.maximum(segDiff[1:], np.abs(h_li[0:-1] - EPminus[1:]))

    mask = segDiff < tol

    return mask

def gps2dyr(time):
    """ Converte from GPS time to decimal years. """
    time = Time(time, format='gps')
    time = Time(time, format='decimalyear').value

    return time

def list_files(path, endswith='.h5'):
    """ List files in dir recursively."""
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if f.endswith(endswith)]

def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def track_type(time, lat, tmax=1):
    """
        Determines ascending and descending tracks.
        Defines unique tracks as segments with time breaks > tmax,
        and tests whether lat increases or decreases w/time.
    """

    # Generate track segment
    tracks = np.zeros(lat.shape)

    # Set values for segment
    tracks[0:np.argmax(np.abs(lat))] = 1

    # Output index array
    i_asc = np.zeros(tracks.shape, dtype=bool)

    # Loop trough individual tracks
    for track in np.unique(tracks):

        # Get all points from an individual track
        i_track, = np.where(track == tracks)

        # Test tracks length
        if len(i_track) < 2:
            continue

        # Test if lat increases (asc) or decreases (des) w/time
        i_min = time[i_track].argmin()
        i_max = time[i_track].argmax()
        lat_diff = lat[i_track][i_max] - lat[i_track][i_min]

        # Determine track type
        if lat_diff > 0:
            i_asc[i_track] = True

    # Output index vector's
    return i_asc, np.invert(i_asc)


def geotiffread(ifile,metaData):
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


def bilinear2d(xd,yd,data,xq,yq, **kwargs):
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


# Output description of solution
description = ('Program for reading ICESat ATL06 data.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        'ifiles', metavar='ifile', type=str, nargs='+',
        help='path for ifile(s) to read (.h5)')

parser.add_argument(
        '-o', metavar=('odir'), dest='odir', type=str, nargs=1,
        help='path to output folder',
        default=[None])

parser.add_argument(
        '-f', metavar=('fmask'), dest='fmask', type=str, nargs=1,
        help="name of raster file mask",
        default=[None],)

parser.add_argument(
        '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
        help=('bounding box for geographical region (deg)'),
        default=[None],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="number of cores to use for parallel processing",
        default=[1],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-i', metavar=('index'), dest='index', type=int, nargs=1,
        help=('provide mission index'),
        default=[None],)

parser.add_argument(
        '-g', metavar=('gran'), dest='gran', type=str, nargs='+',
        help='only select specific granule(s)',
        default=[None],)

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifiles = args.ifiles
opath_ = args.odir[0]
bbox   = args.bbox
njobs  = args.njobs[0]
fmask  = args.fmask[0]
proj   = args.proj[0]
index  = args.index[0]
gran   = args.gran

# Get filelist of data to process
#ifiles = list_files(ipath,endswith='.h5')

# Beam namnes
group = ['/gt1l','/gt1r','/gt2l','/gt2r','/gt3l','/gt3r']

# Beam indicies
beams = [1, 2, 3, 4, 5, 6]

# Create beam index container
orb_i = 0

# Select the granules
if gran[0]: gran = np.asarray(gran).astype(int)

# Raster mask
if fmask is not None:

    print('Reading raster mask ....')
    if fmask.endswith('.tif'):

        # Read in masking grid
        (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(fmask, "A")

    else:

        # Read Hdf5 from memory
        Fmask = h5py.File(fmask, 'r')

        # Get variables
        Xm = Fmask['X'][:]
        Ym = Fmask['Y'][:]
        Zm = Fmask['Z'][:]

# Loop trough and open files
def main(ifile, n=''):

    if gran[0] is not None:
        gran_str = ifile.split('_')[-3]

        n_str = len(gran_str)

        gran_num = int(gran_str[-2:n_str])

        if np.all(gran_num != gran):
            return

    # Access global variable
    global orb_i

    # Check if we already processed the file
    if ifile.endswith('_A.h5') or ifile.endswith('_D.h5'):
        return

    # Beam flag error
    flg_read_err = False

    # Loop trough beams
    for k in range(len(group)):

        # Load full data into memory (only once)
        with h5py.File(ifile, 'r') as fi:

            # Try to read vars
            try:

                # Read in varibales of interest (more can be added!)
                dac  = fi[group[k]+'/land_ice_segments/geophysical/dac'][:]
                lat  = fi[group[k]+'/land_ice_segments/latitude'][:]
                lon  = fi[group[k]+'/land_ice_segments/longitude'][:]
                h_li = fi[group[k]+'/land_ice_segments/h_li'][:]
                s_li = fi[group[k]+'/land_ice_segments/h_li_sigma'][:]
                t_dt = fi[group[k]+'/land_ice_segments/delta_time'][:]
                flag = fi[group[k]+'/land_ice_segments/atl06_quality_summary'][:]
                s_fg = fi[group[k]+'/land_ice_segments/fit_statistics/signal_selection_source'][:]
                snr  = fi[group[k]+'/land_ice_segments/fit_statistics/snr_significance'][:]
                h_rb = fi[group[k]+'/land_ice_segments/fit_statistics/h_robust_sprd'][:]
                f_sn = fi[group[k]+'/land_ice_segments/geophysical/bsnow_conf'][:]
                tref = fi['/ancillary_data/atlas_sdp_gps_epoch'][:]
                rgt  = fi['/orbit_info/rgt'][:]*np.ones(len(lat))
                dh_fit_dx = fi[group[k]+'/land_ice_segments/fit_statistics/dh_fit_dx'][:]

                # Tide corrections
                tide_earth = fi[group[k]+'/land_ice_segments/geophysical/tide_earth'][:]
                tide_load  = fi[group[k]+'/land_ice_segments/geophysical/tide_load'][:]
                tide_ocean = fi[group[k]+'/land_ice_segments/geophysical/tide_ocean'][:]
                tide_pole  = fi[group[k]+'/land_ice_segments/geophysical/tide_pole'][:]

            except:

                # Set error flag
                flg_read_err = True
                pass

        # Continue to next beam
        if flg_read_err: return

        # Remove DAC
        #h_li = h_li + np.float64(dac)     ####### NOTE WE ARE REMOVING DAC FOR R205

        # Make sure its a 64 bit float
        dh_fit_dx = np.float64(dh_fit_dx)

        # Apply bounding box
        if bbox[0]:

            # Extract bounding box
            (lonmin, lonmax, latmin, latmax) = bbox

            # Select data inside bounding box
            ibox = (lon >= lonmin) & (lon <= lonmax) & (lat >= latmin) & (lat <= latmax)

            # Set mask container
            i_m  = np.ones(lat.shape)

        # Raster mask
        elif fmask is not None:


            # Determine projection
            if proj != '4326':

                # Reproject coordinates
                (x, y) = transform_coord('4326', proj, lon, lat)

            else:

                # Keep lon/lat
                x, y = lon, lat

            # Interpolation of grid to points for masking
            i_m = bilinear2d(Xm, Ym, Zm, x.T, y.T, order=1)

            # Set all NaN's to zero
            i_m[np.isnan(i_m)] = 0

        else:

            # Select all boolean
            ibox = np.ones(lat.shape,dtype=bool)

            # Set mask container
            i_m  = np.ones(lat.shape)

        # Compute segment difference mask
        mask = segDifferenceFilter(dh_fit_dx, h_li, tol=2)

        # Copy original flag
        q_flag = flag.copy()

        # Quality flag, only keep good data and data inside box
        # flag = (s_fg == 0) & (s_li < 1) & (h_rb < 1) & (np.abs(h_li) < 10e3) & (i_m > 0)
        flag = (flag == 0) & (np.abs(h_li) < 10e3) & (i_m > 0) & mask

        # Only keep good data
        lat, lon, h_li, s_li, t_dt, h_rb, s_fg, snr, q_flag, f_sn, tide_earth, tide_load, tide_ocean, tide_pole, dac,\
            rgt = lat[flag], lon[flag], h_li[flag],s_li[flag], t_dt[flag], h_rb[flag], s_fg[flag], \
                snr[flag], q_flag[flag], f_sn[flag], tide_earth[flag], tide_load[flag], \
                    tide_ocean[flag], tide_pole[flag], dac[flag], rgt[flag]

        # Test for no data
        if len(h_li) == 0: return

        # Time in decimal years
        t_li = gps2dyr(t_dt + tref)

        # Time in GPS seconds
        t_gps = t_dt + tref

        # Determine track type
        (i_asc, i_des) = track_type(t_li, lat)

        # Determine if to use the index
        if index is not None:

            # Add unique mission identifier
            orb_unique = np.char.add(str(index), str(orb_i)).astype('int')

            # Create orbit number
            orb = np.full(t_gps.shape, orb_unique)

        else:

            # Create orbit number
            orb = np.full(t_gps.shape, orb_i)

        # Construct output name and path
        ipath, fname = os.path.split(ifile)
        name, ext = os.path.splitext(fname)
        opath = opath_ if opath_ else ipath
        ofile = os.path.join(opath, name +'_'+group[k][1:]+ ext)

        with h5py.File(ofile, 'w') as fa:

            # Save ascending vars
            fa['orbit'] = orb
            fa['lon'] = lon
            fa['lat'] = lat
            fa['h_elv'] = h_li
            fa['s_elv'] = s_li
            fa['t_year'] = t_li
            fa['h_rb'] = h_rb
            fa['s_fg'] = s_fg
            fa['snr'] = snr
            fa['q_flg'] = q_flag
            fa['f_sn'] = f_sn
            fa['t_sec'] = t_gps
            fa['tide_load'] = tide_load
            fa['tide_ocean'] = tide_ocean
            fa['tide_pole'] = tide_pole
            fa['tide_earth'] = tide_earth
            fa['dac'] = dac
            fa['rgt'] = rgt
            fa['trk_type'] = i_asc

        # Name of file
        try:
            print('out ->', ofile)
        except:
            print('Not processed!')

        # Update orbit number
        orb_i += 1

# Run main program
if njobs == 1:

    print('running sequential processing ...')
    [main(f) for f in ifiles]

else:

    print('running parallel processing (%d jobs) ...' % njobs)
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f, n) for n, f in enumerate(ifiles))

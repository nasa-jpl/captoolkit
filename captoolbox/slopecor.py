#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:20:19 2015

@author: nilssonj


Change log:

    - added imports
    - added HDF5 I/O
    - added parallelization
    - added argparse (command-line args)
    - added auto-detection for aspect cor (south hemi)
    - added optional input string for glob (when too many files)
    - added optional column or variable names as input
    - if HDF5, only loads x,y,z,r variables, and keeps the original

Example:

    python slopecor.py '/mnt/devon-r0/shared_data/envisat/grounded/*.h5' -s /mnt/devon-r0/shared_data/DEM/bedmap2/bedmap2_surface_wgs84_2km_slope.tif -a /mnt/devon-r0/shared_data/DEM/bedmap2/bedmap2_surface_wgs84_2km_aspect.tif -u /mnt/devon-r0/shared_data/DEM/bedmap2/bedmap2_surface_wgs84_2km_curve.tif -m RM -v lon lat h_cor r_ice1_cor -l 1.5 -d -g A -n 16

"""
__version__ = 0.2

import os
import sys
import glob
import h5py
import pyproj
import argparse
import pandas as pd
import numpy as np
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates
from scipy.ndimage import generic_filter

"""

Program for correcting satellite derived surface elevation for slope induced range errors, either by (1) Direct Method
(DM) (Brenner et al., 1983) or (2) Relocation Method (RM) (Bamber et al., 1994). The algorithm uses pre-defined rasters
of surface slope, curvature and aspect to correct the measured surface range to either nadir (1) or to the echolocation
(2). These parameters are extraxted for each nadir location by the means of bilinear-interpolation and the elevation
correction computed and applied. The algorithm corrects for both the local topography and for Earth's curvature. Each
topographical parameter can also be low-pass filtered using a NxN averaging kernel to provider smoother estimates. User
also provides a maximum allowed surface slope, where values larger than the provided slope are set to the maximum slope.
If range not provided the user can set a default range by setting the "cr" column to a values less than zero, with the
value corresponding to the altitude in km.

INPUT:
----------

ifilePath   :    Path to directory of data
ofilePath   :    Path to directroy of output
slopefile   :    Name (and path) of slope grid (rad or deg)
aspecfile   :    Name (and path) of aspect grid (rad or deg)
curvefile   :    Name (and path) of curvature grid (1/m)
mode        :    Method ("DM") Direct method ("RM") Relocation method
cx          :    Column of x-data (lon)
cy          :    Column of y-data (lat)
cz          :    Column of z-data (elev)
cr          :    Column of r-data (range), if cr < 0 then cr is altitude in (km)
proj        :    Grid projection EPGS number
filt        :    Filter the input DEM "on" "off"
kern        :    Size of filter kernel (int)
smax        :    Maximum allowed surface slope (deg)
degrad      :    Rasters are given in radians (0) or degrees (1)
meta        :    Raster coordinate type

OUTPUT:
----------

The output files are saved wiht the same name as the input, but with either (RM) or (DM) added to the end to indicate
which slope correction method has been used. Data in the columns (cx,cy,cz) or (cz) are replaced with the new and
corrected value.

"""

# Define command-line arguments
parser = argparse.ArgumentParser(description='Computes slope correction')

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (ASCII, HDF5 or Numpy)')

parser.add_argument(
        '-o', metavar=('outdir'), dest='outdir', type=str, nargs=1,
        help='output dir, default same as input',
        default=[None],)

parser.add_argument(
        '-s', metavar=('slope.tif'), dest='slope', type=str, nargs=1,
        help='raster file containing slope (deg or rad)',
        default=[None],)

parser.add_argument(
        '-a', metavar=('aspect.tif'), dest='aspect', type=str, nargs=1,
        help='raster file containing aspect (deg or rad)',
        default=[None],)

parser.add_argument(
        '-u', metavar=('curve.tif'), dest='curve', type=str, nargs=1,
        help='raster file containing curvature (1/m)',
        default=[None],)

parser.add_argument(
        '-m', metavar=None, dest='mode', type=str, nargs=1,
        help=('correction mode: direct (DM) or relocation (RM)'),
        choices=('DM', 'RM'), default=['DM'],)

parser.add_argument(
        '-c', metavar=('0','1','2','3'), dest='cols', type=int, nargs=4,
        help='lon,lat,height,range columns, if -1 for range, use alt. (km)',
        default=[0,1,2,3],)  # <- default is used for HDF5

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-k', metavar=('kern_size'), dest='kern', type=int, nargs=1,
        help=('if provided, smooth fields using N by N mean kernel (pxl)'),
        default=[None],)

parser.add_argument(
        '-l', metavar=('max_slope'), dest='smax', type=float, nargs=1,
        help=('max value allowed for slope (deg)'),
        default=[1.5],)

parser.add_argument(
        '-d', dest='degrad', action='store_true',
        help=('rasters are in degrees -> convert to radians'),
        default=False)

parser.add_argument(
        '-g', metavar=None, dest='meta', type=str, nargs=1,
        help=('rasters are cell-centered (P) or node-centered (A)'),
        choices=('P', 'A'), default=['A'],)

parser.add_argument(
        '-v', metavar=('x','y','h','r'), dest='vnames', type=str, nargs=4,
        help='lon/lat/height/range variable names in HDF5',
        default=['lon','lat','height','range'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple files",
        default=[1],)

args = parser.parse_args()

# Data input
ifilePath = args.files
ofilePath = args.outdir[0]
slopeFile = args.slope[0]
aspecFile = args.aspect[0]
curveFile = args.curve[0]
mode = args.mode[0]
cx = args.cols[0]
cy = args.cols[1]
cz = args.cols[2]
cr = args.cols[3]
proj  = args.proj[0]
kern = args.kern[0]
filt = 'on' if kern else 'off'
smax = args.smax[0] 
degrad = args.degrad
meta = args.meta[0] 

vnames = args.vnames
njobs = args.njobs[0]

print 'parameters:'
for arg in vars(args).iteritems(): print arg


def bilinear2d(xd,yd,data,xq,yq, **kwargs):
    
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


def geotiffread(ifile,metaData):
    
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
        
        Xp = np.arange(Nx)
        Yp = np.arange(Ny)
        
        (Xp, Yp) = np.meshgrid(Xp,Yp)
        
        X = trans[0] + (Xp+0.5)*trans[1] + (Yp+0.5)*trans[2]
        Y = trans[3] + (Xp+0.5)*trans[4] + (Yp+0.5)*trans[5]
    
    if metaData == "P":
        
        Xp = np.arange(Nx+1)
        Yp = np.arange(Ny+1)
        
        (Xp, Yp) = np.meshgrid(Xp,Yp)
        
        X = trans[0] + Xp*trans[1] + Yp*trans[2]
        Y = trans[3] + Xp*trans[4] + Yp*trans[5]
    
    band = file.GetRasterBand(1)
    
    Z = band.ReadAsArray()
    
    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def wrapTo2Pi(radians):
    
    positiveInput = (radians > 0)
    
    radians = np.mod(radians, 2*np.pi)
    
    radians[(radians == 0) & positiveInput] = 2*np.pi
    
    return radians


def azimuth(lat1,lon1,lat2,lon2):
    
    dlong = np.deg2rad(lon2-lon1)
    
    x = np.sin(dlong)*np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2)-(np.sin(lat1)*np.cos(lat2)*np.cos(dlong))
    
    azimuth = np.arctan2(x, y) # Depend on hemispehere?
    
    azimuth[lat1 <= -np.pi/2.] = 0
    azimuth[lat2 >=  np.pi/2.] = 0
    azimuth[lat2 <= -np.pi/2.] = np.pi
    azimuth[lat1 >=  np.pi/2.] = np.pi
    
    azimuth = np.mod(azimuth, 2*np.pi);
    
    return azimuth.reshape((len(azimuth),1))


def track_azimuth(lat,lon):
    
    N = len(lat)
    
    az = azimuth(lat[0:N-2],lon[0:N-2],lat[2:N],lon[2:N])

    Az = np.vstack((az[0]-(az[1]-az[0]),az,az[-1]+(az[-1]-az[-2])))
    
    p = np.polyfit(np.arange(0,len(lat)),Az,3)
    
    Az = np.polyval(p,np.arange(0,len(lat)))

    return Az


# Get file list from directory
if len(ifilePath) == 1:
    files = glob.glob(ifilePath[0])
else:
    files = ifilePath

# Change to radians
smax *= np.pi / 180.0

# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")
projGrd = pyproj.Proj("+init=EPSG:"+proj)

# Ellipsoid parameters - WGS84
a = 6378137.0
f = 1.0 / 298.2572235630
b = (1 - f) * a
e2 = (a * a - b * b) / (a * a)

# Load DEM from .tif
(Xs, Ys, Zs, dX, dY, PROJ) = geotiffread(slopeFile, meta)
(Xa, Ya, Za, dX, dY, PROJ) = geotiffread(aspecFile, meta)
(Xc, Yc, Zc, dX, dY, PROJ) = geotiffread(curveFile, meta)

# Converte from degrees to radians
if degrad == 1:

    # Degrees to radians
    Za *= np.pi / 180
    Zs *= np.pi / 180

# Filter topological parameters
if filt == "on":

    # Filter the input grids
    Zs = generic_filter(Zs, np.mean, kern)
    Za = generic_filter(Za, np.mean, kern)
    Zc = generic_filter(Zc, np.mean, kern)


def main(ifile):

    print 'input file:', ifile, '...'
    
    # Determine file type
    if ifile.endswith('.npy'):

        # Load data points - Binary
        Points = np.load(ifile)

    elif ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
        
        # Load data points - HDF5
        with h5py.File(ifile) as f:
            if len(f.keys()) == 0: return
            Points = np.column_stack([f[k][:] for k in vnames])

    else:

        # Load data points - ASCII
        Points = pd.read_csv(ifile, header=None, delim_whitespace=True)
    
        # Converte to numpy array
        Points = pd.DataFrame.as_matrix(Points)

    # Check if empty file
    if len(Points) == 0: return

    # Define output array
    OFILE = np.copy(Points)
    
    # Satellite elevation
    H = Points[:,cz]

    # Check if range is available
    if cr < 0:
        
        # Set altitude from input
        A = np.abs(cr)
        
        # Calulate range
        R = A - H
    
    else:
        
        # Get range estimates
        R = Points[:, cr]

    # Compute satellite altitude
    A = H + R

    # Satellite coordinates
    lon = Points[:,cx]
    lat = Points[:,cy]

    # Reproject coordinates to conform with grid
    (x, y) = pyproj.transform(projGeo, projGrd, lon, lat)

    # Satellite coordinates in radians
    lon *= np.pi / 180
    lat *= np.pi / 180
    
    # Interpolate grid-values to location of point data
    slope  = bilinear2d(Xs, Ys, Zs, x, y, order=1)
    curve  = bilinear2d(Xc, Yc, Zc, x, y, order=1)
    aspect = bilinear2d(Xa, Ya, Za, x, y, order=1)
    
    # Edit slope magnitude
    slope[slope > smax] = smax
    
    # Correct to north using longitude (only south hemisphere)
    lat2 = lat[(np.abs(lat)<90)&(~np.isnan(lat))]
    if len(lat2) > 0 and lat2[0] < 0:
        aspect -= wrapTo2Pi(lon.copy())
        del lat2
    
    # Compute correct slope aspect
    aspect -= np.pi

    # Curvature correction - topography
    nabla = 1 + R * curve
        
    # Curvature parameters in lon/lat directions
    rho_lat = (a * (1 - e2)) / ((1 - e2 * (np.sin(lat) ** 2)) ** (1.5))
    rho_lon = (a * np.cos(lat)) / (np.sqrt(1 - e2 * np.sin(lat) ** 2))
        
    # Combined parameters in range direction
    rho = (rho_lat * rho_lon) / (rho_lat * np.cos(lat) * \
            np.sin(aspect) ** 2 + rho_lon * np.cos(aspect) ** 2)
    
    # Direct method (DM)
    if mode == "DM":
        
        # Slope and curvature corrected range - Direct method
        h_echo = A - (R / np.cos(slope) + (nabla * (R * np.sin(slope)) ** 2) / (2 * rho))
        
        # Save corrected data
        OFILE[:,cz] = h_echo

        # Dictionary to save data into HDF5
        OFILEd = {vnames[2]: h_echo}
    
    # Relocation method (RM)
    if mode == "RM":
        
        # Slope and curvature corrected range - Relocation method
        h_echo = A - R * np.cos(slope) + (nabla * (R * np.sin(slope)) ** 2) / (2.0 * rho)
        
        # Postion migrated to approximate echolocation
        lat_echo = lat + R * np.sin(slope) * np.cos(aspect) / rho_lat
        lon_echo = lon + R * np.sin(slope) * np.sin(aspect) / rho_lon
            
        # Converte to degrees
        lat_echo *= 180 / np.pi
        lon_echo *= 180 / np.pi
        
        # Save corrected data
        OFILE[:,cx] = lon_echo
        OFILE[:,cy] = lat_echo
        OFILE[:,cz] = h_echo

        # Dictionary to save data into HDF5
        OFILEd = {vnames[0]: lon_echo, vnames[1]: lat_echo, vnames[2]: h_echo}

    # Get output file name (to replace input name)
    path, fname = os.path.split(ifile)
    name, ext = os.path.splitext(fname)
    suffix = '_DM' if mode == 'DM' else '_RM'
    path = ofilePath if ofilePath else path
    ofile = os.path.join(path, name + suffix + ext)

    # Save data
    if ifile.endswith('.npy'):
        # Save to binary file
        np.save(ofile,OFILE)

    elif ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):

        with h5py.File(ifile, 'a') as f:
            for k,v in OFILEd.items():
                f[k+'_orig'] = f[k]  # rename original vars
                del f[k]
                f[k] = v

        os.rename(ifile, ofile)

    else:
        # Save to ascii file
        np.savetxt(ofile, OFILE, delimiter=' ',fmt="%8.5f")

    print 'output file:', ofile
        

if njobs == 1:
    print 'running sequential code ...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)


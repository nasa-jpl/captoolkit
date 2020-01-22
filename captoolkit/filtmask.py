#!/usr/bin/env python

import sys
import h5py
import pyproj
import pandas as pd
import numpy as np
import argparse
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates
from scipy.io import savemat
from scipy.ndimage.morphology import distance_transform_edt
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator as LNDI

"""
Program for masking point data using binary masking rasters or bounding box. The program takes as input a point file
and a raster mask containing specific values. The raster values are interpolated to each record by the means of
bilinear interpolation and the values selected for output is provided by the user as input. The ouput data can be read
and saved to several types of data types, depending on the file ending "i.e. .xyz", where three types are supported (1)
python binary format ".npy", (2) MATLAB binary format ".mat" and (3) everything else is interpreted as ascii format.
"""

# Read raster from file
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
        
        xp = np.arange(Nx)
        yp = np.arange(Ny)
        
        (Xp, Yp) = np.meshgrid(xp,yp)
        
        X = trans[0] + (Xp+0.5)*trans[1] + (Yp+0.5)*trans[2]
        Y = trans[3] + (Xp+0.5)*trans[4] + (Yp+0.5)*trans[5]
    
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

# Bilinear interpolation from grid
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

# Output description of solution
description = ('Masking of h5 files using raster, polygon, bounding box and time.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='name of input file')

parser.add_argument(
            'ofile', metavar='ofile', type=str, nargs='+',
            help='name of output file')

parser.add_argument(
            '-f', metavar='mfile', dest='mfile', type=str, nargs=1,
            help=('name of raster mask (".tif").'),
            default=[False],)

parser.add_argument(
            '-v', metavar='var', dest='vnames', type=str, nargs='+',
            help=('name of longitude and latitude variables'),
            default=[],)

parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
            default=[4326],)

parser.add_argument(
            '-k', metavar=('keep'), dest='keep', type=int, nargs=1,
            help=('selection index'),
            default=[1],)

parser.add_argument(
            '-r', metavar=('buffer'), dest='buffer', type=float, nargs=1,
            help=('buffer distance - buffer removed (km).'),
            default=[0],)

parser.add_argument(
            '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geograph. region (deg or m), optional'),
            default=[False],)

parser.add_argument(
            '-g', metavar=None, dest='meta', type=str, nargs=1,
            help=('coordinate reference for grid'),
            choices=('A', 'P'), default=['A'],)

parser.add_argument(
            '-t', metavar=('tmin','tmax'), dest='time', type=float, nargs=2,
            help=('time selection of data'),
            default=[-9999,9999],)

parser.add_argument(
            '-c', metavar=None, dest='comp', type=bool, nargs=1,
            help=('compress output file'),
            choices=(True, False), default=[False],)

# Add argument to parser string
args = parser.parse_args()

# Get the input
ifile  = args.ifile[0]
ofile  = args.ofile[0]
imask  = args.mfile[0]
icol   = args.vnames[:]
proj   = args.proj[0]
gmeta  = args.meta[0]
keep   = args.keep[0]
buffer = args.buffer[0]
bbox   = args.bbox[:]
tmin   = args.time[0]
tmax   = args.time[1]
comp   = args.comp[0]

# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")

# Make pyproj format
projection = '+init=EPSG:' + proj

# Projection - prediction grid
projGrd = pyproj.Proj(projection)

# Check for compression
if comp:

    comp='lzf'

else:
    
    comp=''

# Check mask option
if imask:
    
    # Read in masking grid
    (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(imask, gmeta)
    
    # Set NaN to zero
    Zm[np.isnan(Zm)] = 0

elif bbox:

    # Provide Bounding box
    (xmin, xmax, ymin, ymax) = bbox

else:

    # Print message to terminal
    print("*** Warning - Provide raster or bounding box! ***")

    # Exit program
    sys.exit()

# Create buffer - for grid
if buffer > 0:

    # Compute distances in masked array
    Zm = distance_transform_edt(Zm)

    # Create buffer (km)
    Zm[Zm <= buffer] = 1

else:

    # Buffer to add - deg or meter
    dxy = np.abs(buffer)

print('loading data ...')
    
# Determine input file type
if not ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):
    
    # Print message
    print("*** Warning - input file must be in hdf5-format! ***")
    
    # Exit program
    sys.exit()

# Input variables
xvar, yvar, tvar = icol

# Open pointer to file
fi = h5py.File(ifile, 'r')
    
# Load latitude and longitude
lon = fi[xvar][:]
lat = fi[yvar][:]

# Load time variable
time = fi[tvar][:]

# Check for coordinate type
if proj != "4326":
    
    # Change projection
    (x, y) = pyproj.transform(projGeo, projGrd, lon, lat)

else:
    
    # Output geographical coordinates
    x = lon
    y = lat

print("editing data ...")

# Raster masking mode
if imask:

    # Interpolation of grid to points for masking
    Ii = bilinear2d(Xm, Ym, Zm, x, y, order=0)
    
    # Set all NaN's to zero
    Ii[np.isnan(Ii)] = 0

    # Get index for data inside mask
    Io = (Ii == keep) #& ((time >= tmin) & (time <= tmax))

# Bounding-box mode
elif bbox:

    # Get index of wanted bounding box
    Io = (x >= (xmin - dxy)) & (x <= (xmax + dxy)) & (y >= (ymin - dxy)) & (y <= (ymax + dxy)) & \
         ((time >= tmin) & (time <= tmax))

# Otherwise exit program
else:

    # Print message to terminal
    print("*** Warning - Provide raster or bounding box! ***")
    
    # Exit program
    sys.exit()

print("saveing data ...")

# Open ouput file and save data
with h5py.File(ofile, 'w') as fout:
    
    # Create output file and loop trough varibales
    [fout.create_dataset(k, data=d[:][Io], dtype='float64', compression=comp) for k,d in list(fi.items())]

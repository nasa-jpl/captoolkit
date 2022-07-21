"""
Regrid mask (from input file) onto grid (from output file).

"""
import sys
import pyproj
import numpy as np

import h5py
from netCDF4 import Dataset

try:
    from gdalconst import *
except ImportError as e:
    from osgeo.gdalconst import *
from osgeo import gdal, osr

from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve

#file_mask = '/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5'
file_mask = '/Users/paolofer/data/masks/scripps_new/ICE1_ICE2_AnIS_dHdt_FLOAT_mask.h5'
xvar1 = 'x'
yvar1 = 'y'
zvar1 = 'mask'

file_out = '/Users/paolofer/code/captoolkit/captoolkit/work/cube_full/FULL_CUBE_v2.h5'
xvar2 = 'x'
yvar2 = 'y'


def h5read(ifile, vnames):
    with h5py.File(ifile, 'r') as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode='a'):
    with h5py.File(fname, mode) as f:
        for k, v in list(vardict.items()):
            try:
                f[k] = np.squeeze(v)
            except:
                f[k][:] = np.squeeze(v)


def ncread(ifile, vnames):
    ds = Dataset(ifile, "r")   # NetCDF4
    d = ds.variables
    return [d[v][:] for v in vnames]


def tifread(ifile, metaData):
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
    #return X, Y, Z, dx, dy, proj
    return X, Y, Z


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


# Get grid 1
if '.h5' in file_mask:
    print('getting grid from hdf5 ...')
    x1, y1, Z1 = h5read(file_mask, [xvar1,yvar1,zvar1])
elif '.nc' in file_mask:
    print('getting grid from netcdf4 ...')
    x1, y1, Z1 = ncread(file_mask, [xvar1,yvar1,zvar1])
else:
    print('getting grid from geotiff ...')
    x1, y1, Z1 = tifread(file_mask, 'A')

if 1:
    # For Susheel's Calving mask
    Z1 = Z1.T
    x1 = x1.T
    y1 = y1.T

x1, y1 = x1[0,:], y1[:,0]


# Get grid 2
if '.h5' in file_out:
    print('getting grid from hdf5 ...')
    x2, y2 = h5read(file_out, [xvar2,yvar2])
elif '.nc' in file_out:
    print('getting grid from netcdf4 ...')
    x2, y2 = ncread(file_out, [xvar2,yvar2])
else:
    print('getting grid from geotiff ...')
    x2, y2, _ = tifread(file_out, 'A')


#Z1 = Z1.filled(np.nan)  # masked -> ndarray
X1, Y1 = np.meshgrid(x1, y1)  # 1d -> 2d

X2, Y2 = np.meshgrid(x2, y2)
xx2, yy2 = X2.ravel(), Y2.ravel()


try:
    zz2 = interp2d(X1, Y1, Z1, xx2, yy2, order=1)
except:
    zz2 = interp2d(X1, np.flipud(Y1), np.flipud(Z1), xx2, yy2, order=1)

Z2 = zz2.reshape(X2.shape)

# Mask ice shelves

plt.matshow(Z2)
plt.figure()
plt.pcolormesh(x2, y2, Z2, cmap='RdBu')
plt.colorbar()
plt.show()


if 0:
    h5save(file_out, {'mask_calving': Z2}, 'a')
    print(('out ->', file_out))

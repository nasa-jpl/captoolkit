"""
Filter and re-grid Mean Height field (DEM) from grid-1 onto grid-2.

Notes:
    Interpolate coordinates of grid-2 (x2/y2) onto grid-1

"""
import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates
from scipy.ndimage import median_filter
from scipy.interpolate import griddata
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

# === EDIT ======================================================

'''
FILE1 = ('/Users/paolofer/code/captoolkit/captoolkit/work/'
         'cube_full/FULL_DEM.h5')
'''
FILE1 = ('/Users/paolofer/code/captoolkit/captoolkit/work/data/'
                    'CS2_AD_OINTERP_HEIGHT.h5')

xvar1 = 'x'
yvar1 = 'y'
tvar1 = 't_ref'
zvar1 = 'height'

FILE2 = ('/Users/paolofer/code/captoolkit/captoolkit/work/'
         'cube_full/FULL_CUBE_v2.h5')

xvar2 = 'x'
yvar2 = 'y'
tvar2 = 't_mean_cs2'  # output var
zvar2 = 'h_mean_cs2'  # output var

FILE_MASK = '/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5'  # floating
#FILE_MASK = '/Users/paolofer/data/masks/jpl/ANT_groundedice_240m.tif.h5'  # grounded

#================================================================

if len(sys.argv) > 1:
    FILE1 = sys.argv[1]
    FILE2 = sys.argv[2]


def h5read(ifile, vnames):
    with h5py.File(ifile, 'r') as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode='a'):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items(): f[k] = np.squeeze(v)


def get_mask(file_msk, X, Y):
    """Given mask file and x/y grid coord return boolean mask.
    
    X/Y can be either 1D or 2D.
    """
    try:
        Xm, Ym, Zm = tifread(file_msk)
    except:
        Xm, Ym, Zm = h5read(file_msk, ['x', 'y', 'mask'])

    # Interpolation of grid to points for masking
    mask = interp2d(Xm, Ym, Zm, X.ravel(), Y.ravel(), order=1)
    mask = mask.reshape(X.shape)

    # Set all NaN's to zero
    mask[np.isnan(mask)] = 0

    # Convert to boolean
    return mask == 1


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
    return pyproj.transform(proj1, proj2, x, y)


def regrid2d(x1, y1, z1, x2, y2, method='linear'):
    """Regrid z1(x1,y1) onto z2(x2,y2).

    Args:
        z1 is a 2D array.
        x1/y1/x2/y2 can be either 1D or 2D arrays.
    """
    if np.ndim(x1) == 1:
        X1, Y1 = np.meshgrid(x1, y1)
    else:
        X1, Y1 = x1, y1
    if np.ndim(x2) == 1:
        X2, Y2 = np.meshgrid(x2, y2)
    else:
        X2, Y2 = x2, y2

    Z2 = griddata((X1.ravel(),Y1.ravel()), z1.ravel(), (X2,Y2), method=method)
    return Z2.reshape(X2.shape)


def xregrid2d(x1, y1, z1, x2, y2):
    import xarray as xr
    da = xr.DataArray(z1, [('y', y1), ('x', x1)])
    return da.interp(x=x2, y=y2).values


print('loading grids ...')
x1, y1, t1, Z1 = h5read(FILE1, [xvar1,yvar1,tvar1,zvar1])
x2, y2 = h5read(FILE2, [xvar2,yvar2])

try:
    t1 = np.nanmean(t1)
except:
    pass

#NOTE: Always check this <<<<<<
y1 = y1[::-1]  

X1, Y1 = np.meshgrid(x1, y1)  # 1d -> 2d
X2, Y2 = np.meshgrid(x2, y2)

# Get mask
if 0:
    print('generating mask ...')
    mask = get_mask(FILE_MASK, X2, Y2)
    h5save(FILE2, {'mask_floating': mask}, 'a')
else:
    mask, = h5read(FILE2, ['mask_floating'])

# Fillin/Interp before regridding
if 1:
    print('filling and filtering ...')
    kernel = Gaussian2DKernel(1)
    Z1 = interpolate_replace_nans(Z1, kernel)
    Z1 = median_filter(Z1, 3)

print('re-gridding ...')
#Z2 = regrid2d(x1, y1, Z1, x2, y2)
Z2 = xregrid2d(x1, y1, Z1, x2, y2)

Z2[mask==0] = np.nan

if 0:
    h5save(FILE2, {zvar2: Z2, tvar2: t1}, 'a')
    print('saved.')

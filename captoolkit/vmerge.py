import sys
import h5py
import pyproj
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from netCDF4 import Dataset

import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

warnings.filterwarnings("ignore")

f1 = 'gardner/ANT_G0240_0000.nc'
f2 = 'rignot/antarctica_ice_velocity_450m_v2.nc'
f3 = 'merged/ANT_G0240_0000_PLUS_450m_v2_c.h5'

bbox = (-600000, 410000, -1400000, -400000)  # Ross


def load_fields(fname, variables=['x', 'y', 'vx', 'vy']):
    """Load velocity grids from NetCDF4."""
    ds = Dataset(fname, "r")
    s = [np.ma.filled(ds.variables[v][:], fill_value=np.nan) for v in variables]
    ds.close()
    return fields


def load_velocity(fname, vname=['x', 'y', 'vx', 'vy'],
        step=1, xlim=None, ylim=None, flipud=False):
    """ Load velocity grid from NetCDF4. """

    # Default variable names in the NetCDF file
    xvar, yvar, uvar, vvar = vname

    ds = Dataset(fname, "r")
    x = ds.variables[xvar][:]         # 1d 
    y = ds.variables[yvar][:]         # 1d

    # Limits to subset data
    if xlim is not None:
        x1, x2 = xlim
        y1, y2 = ylim
        j, = np.where( (x > x1) & (x < x2) )
        i, = np.where( (y > y1) & (y < y2) )
        x = x[j.min():j.max()]
        y = y[i.min():i.max()]
        U = ds.variables[uvar][i.min():i.max(),j.min():j.max()]  # 2d
        V = ds.variables[vvar][i.min():i.max(),j.min():j.max()]  # 2d
    else:
        U = ds.variables[uvar][:]  
        V = ds.variables[vvar][:] 

    ds.close()

    if flipud: U, V = np.flipud(U), np.flipud(V)

    return x[::step], y[::step], U[::step,::step], V[::step,::step]


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


_, _, _, E1 = load_velocity(f1, ['x','y','vx','v_err']) #, step=2, xlim=bbox[:2], ylim=bbox[2:])
x1, y1, U1, V1 = load_velocity(f1, ['x','y','vx','vy']) #, step=2, xlim=bbox[:2], ylim=bbox[2:])
x2, y2, U2, V2 = load_velocity(f2, ['x','y','VX','VY']) #, step=2, xlim=bbox[:2], ylim=bbox[2:])

S1 = np.sqrt(U1*U1 + V1*V1)
S2 = np.sqrt(U2*U2 + V2*V2)

X1, Y1 = np.meshgrid(x1, y1)
lon, lat = transform_coord(3031, 4326, X1, Y1)

# Fill in NaN values w/Gaussian interpolation (Eric) 
U2 = np.ma.filled(U2, fill_value=np.nan)  # masked-arr -> nd-arr
V2 = np.ma.filled(V2, fill_value=np.nan)
kernel = Gaussian2DKernel(5)
U2 = interpolate_replace_nans(U2, kernel)
V2 = interpolate_replace_nans(V2, kernel)

# Get missing values to be interpolated (Alex)
# Remove values above 10 m/s and the pole-hole
#S1 = np.ma.masked_where((E1 > 10) | (lat < -82.68), S1)
S1 = np.ma.masked_where((lat < -82.68), S1)   # only fillin the pole-hole
mask1 = np.ma.getmask(S1)

i,j = np.nonzero(mask1)
xp, yp = x1[j], y1[i]

# Interpolate missing values (Eric -> Alex)
X2, Y2 = np.meshgrid(x2, y2)
U1[i,j] = interp2d(X2, Y2, U2, xp, yp, order=1)
V1[i,j] = interp2d(X2, Y2, V2, xp, yp, order=1)

# Median filter
if 0:
    U1 = ndi.median_filter(U1, 3)
    V1 = ndi.median_filter(V1, 3)

if 1:
    with h5py.File(f3, 'w') as f:
        f['x'] = x1
        f['y'] = y1
        f['vx'] = U1
        f['vy'] = V1
    print 'data saved ->', f3

#--- Plot ------------------------------

S1 = np.sqrt(U1*U1 + V1*V1)
S2 = np.sqrt(U2*U2 + V2*V2)

#S1 = ndi.median_filter(S1, 3)

plt.plot(xp, yp, '.', rasterized=True)
plt.title('Interpolated points')

plt.figure()
plt.imshow(S1, extent=(x1.min(), x1.max(), y1.min(), y1.max()), vmin=0, vmax=1000)
plt.colorbar()
plt.figure()
plt.imshow(S2, extent=(x2.min(), x2.max(), y2.min(), y2.max()), vmin=0, vmax=1000)
plt.colorbar()

plt.show()

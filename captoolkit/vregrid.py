"""
Regrid velocity field onto height.

"""
import sys
import h5py
import pyproj
import warnings
import numpy as np
import pyresample as pr
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy import signal
from pyresample import utils
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

# Downsample grids for faster processing (for testing!)
step = 1

# Apply median filter to u and v before regridding
medfilt = True

# Variable names of height field
#hnames = ['x', 'y', 'height']  # CS2 DEM
hnames = ['x', 'y', 'elev']  # REMA

# Variable names of velocity field
vnames = ['x', 'y', 'vx', 'vy']  # Alex
#vnames = ['x', 'y', 'VX', 'VY']  # Eric

# Geographic limits for velocity field (x1,x2,y1,y2)
#region = (-610000, 500000, -1400000, -800000)  ## Ross RA limit
#region = (-600000, 410000, -1400000, -400000)   ## Ross Full
region = ()  # Use full boundaries

# Path to NetCDF velocity file
vfile = '/Users/paolofer/data/velocity/merged/ANT_G0240_0000_PLUS_450m_v2_c.h5'  # Merged
#vfile = '/Users/paolofer/data/velocity/gardner/ANT_G0240_0000.nc'  # Alex
#vfile = '/Users/paolofer/data/velocity/rignot/antarctica_ice_velocity_450m_v2.nc'  # Eric

# Path to HDF5 height file
#hfile = '/Users/paolofer/code/captoolkit/captoolkit/work/CS2_AD_GENDEM.h5_time_2010.500'
hfile = '/Users/paolofer/data/dem/REMA_1km_dem_filled.tif.h5'


def load_height(fname, vname=['x', 'y', 'z'],
        step=1, xlim=None, ylim=None, flipud=False):
    """ Load height grid from HDF5. """

    xvar, yvar, hvar = vname 

    f = h5py.File(fname, 'r')
    x = f[xvar][:]
    y = np.flipud(f[yvar][:])  # FIXME: WHY NEED THIS? BECAUSE OINTERP?
    h = f[hvar][:]

    # 2d -> 1d
    try:
        x, y = x[0,:], y[:,0] 
    except:
        pass

    # Limits to subset data
    if xlim is not None:
        x1, x2 = xlim
        y1, y2 = ylim
        j, = np.where( (x > x1) & (x < x2) )
        i, = np.where( (y > y1) & (y < y2) )
        x = x[j.min():j.max()]
        y = y[i.min():i.max()]
        h = h[i.min():i.max(),j.min():j.max()]

    f.close()

    return x[::step], y[::step], h[::step,::step,...]


def load_velocity(fname, vname=['x', 'y', 'vx', 'vy'],
        step=1, xlim=None, ylim=None, flipud=False):
    """ Load velocity grid from NetCDF4. """

    # Default variable names in the NetCDF file
    xvar, yvar, uvar, vvar = vname

    if '.h5' in fname:
        d = h5py.File(fname, "r")  # HDF5
    else:
        ds = Dataset(fname, "r")   # NetCDF4
        d = ds.variables

    x = d[xvar][:]         # 1d 
    y = d[yvar][:]         # 1d
    U = d[uvar][:]
    V = d[vvar][:]

    # Limits to subset data
    if xlim is not None:
        x1, x2 = xlim
        y1, y2 = ylim
        j, = np.where( (x > x1) & (x < x2) )
        i, = np.where( (y > y1) & (y < y2) )
        x = x[j.min():j.max()]
        y = y[i.min():i.max()]
        U = U[i.min():i.max(),j.min():j.max()]  # 2d
        V = V[i.min():i.max(),j.min():j.max()]  # 2d

    if '.h5' in fname:
        d.close()
    else:
        ds.close()

    if flipud: U, V = np.flipud(U), np.flipud(V)

    return x[::step], y[::step], U[::step,::step], V[::step,::step]


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


def regrid(h, lon2d_h, lat2d_h, lon2d_v, lat2d_v):
    """ Regrid height field (low res) onto velocity field (high res). """

    orig_grid = pr.geometry.SwathDefinition(lons=lon2d_h, lats=lat2d_h)
    targ_grid = pr.geometry.SwathDefinition(lons=lon2d_v, lats=lat2d_v)

    h[~np.isfinite(h)] = 0.

    ##NOTE: Interp using inverse-distance weighting
    wf = lambda r: 1/r**2
    h_interp = pr.kd_tree.resample_custom(orig_grid, h,
            targ_grid, radius_of_influence=10000, neighbours=10,
            weight_funcs=wf, fill_value=0.)

    '''
    wf = lambda r: 1/r**2
    u = pr.kd_tree.resample_custom(targ_grid, u,
            orig_grid, radius_of_influence=10000, neighbours=10,
            weight_funcs=wf, fill_value=0.)

    v = pr.kd_tree.resample_custom(targ_grid, v,
            orig_grid, radius_of_influence=10000, neighbours=10,
            weight_funcs=wf, fill_value=0.)

    h_interp = h
    '''
    return h_interp


def regrid_fields(h, x_h, y_h, u, v, x_v, y_v):
    """Regrid 2nd field onto 1st (i.e. regrid u/v onto h grid)."""

    # Generate 2d coordinate grids
    X_h, Y_h = np.meshgrid(x_h, y_h)
    X_v, Y_v = np.meshgrid(x_v, y_v)

    # x/y -> lon/lat 
    lon2d_h, lat2d_h = transform_coord(3031, 4326, X_h, Y_h)
    lon2d_v, lat2d_v = transform_coord(3031, 4326, X_v, Y_v)

    u = regrid(u, lon2d_v, lat2d_v, lon2d_h, lat2d_h)
    v = regrid(v, lon2d_v, lat2d_v, lon2d_h, lat2d_h)

    return h, u, v, x_h, y_h


def main(fh, fv):

    print 'loading data ...'

    if region:
        xlim, ylim = region[:2], region[2:]
    else:
        xlim, ylim = None, None

    x_h, y_h, h = load_height(
            fh, vname=hnames, xlim=xlim, ylim=ylim)

    x_v, y_v, u, v = load_velocity(
            fv, vname=vnames, xlim=xlim, ylim=ylim, step=step, flipud=False)

    # Replace NaNs with Zeros
    h_mask = ~np.isfinite(h)
    u_mask = ~np.isfinite(u)
    v_mask = ~np.isfinite(v)
    h[h_mask] = 0
    u[u_mask] = 0
    v[v_mask] = 0

    if medfilt:
        u = ndi.median_filter(u, 3)
        v = ndi.median_filter(v, 3)

    print 'regridding ...'
    h, u, v, x, y = regrid_fields(h, x_h, y_h, u, v, x_v, y_v)

    fo = fv + '_regrid_1km_REMA'
    with h5py.File(fo, 'w') as f:
        xvar, yvar, uvar, vvar = vnames
        f[xvar] = x
        f[yvar] = y
        f[uvar] = u
        f[vvar] = v

    print 'out ->', fo

if __name__ == '__main__':
    main(hfile, vfile)

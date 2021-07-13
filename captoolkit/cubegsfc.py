# -*- coding: utf-8 -*-
"""
Resample GSFC cube (t,x,y) onto height cube (y,x,t).

Notes:
    GSFC x/y coords are 2D.

"""
import warnings

import h5py
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numba import jit

warnings.filterwarnings('ignore')

#=== EDIT HERE ==================================================

fgsfc = '/Users/paolofer/data/firn/m2_hybrid_FAC.nc'
tgsfc = 'time'
xgsfc = 'x'
ygsfc = 'y'
zgsfc = 'FAC'

fcube = 'cube_full/FULL_CUBE_v2.h5'
tcube = 't'
xcube = 'x'
ycube = 'y'

saveas = 'fac_gsfc'

# Averaging window (tme steps = months)
window = 31  # GSFC dt = 5 days!!!

# Time interval for subsetting
t1, t2 = 1991.0, 2019.0

#=== END EDIT ===================================================

def h5read(fname, vnames):
    with h5py.File(fname, 'r') as f:
        variables = [f[v][()] for v in vnames]
        return variables if len(vnames) > 1 else variables[0]


def ncread(fname, vnames):
    with nc.Dataset(fname, "r") as ds:
        d = ds.variables
        variables = [d[v][:].filled(fill_value=np.nan) for v in vnames]
        return variables if len(vnames) > 1 else variables[0]


def h5save(fname, vardict, mode='a'):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(v)
            else:
                f[k] = np.squeeze(v)


@jit(nopython=True)
def running_mean_axis0(cube, window, out):
    """Fast moving average for 3D array along first dimension.

    Pass out = cube.copy() to keep original values at the ends (half window).
    """

    assert window % 2 > 0, 'Window must be odd!'
    half = int(window/2.)

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            series = cube[:,i,j]
            if np.isnan(series).all():
                continue

            for k in range(half, cube.shape[0]-half):
                start, stop = k-half, k+half+1
                series_window = series[start:stop]

                asum = 0.0
                count = 0
                for n in range(window):
                    asum += series_window[n]
                    count += 1

                out[k,i,j] = asum/count


def regrid_nearest_axis0(cube, t_cube, t_new):
    """Find nearest values along first dimension."""
    i_new = find_nearest(t_cube, t_new)
    return cube[i_new, :, :]


def find_nearest(arr, val):
    """Find index(es) of "nearest" value(s).

    Parameters
    ----------
    arr : array_like (ND)
        The array to search in (nd). No need to be sorted.
    val : scalar or array_like
        Value(s) to find.

    Returns
    -------
    out : tuple
        The index (or tuple if nd array) of nearest entry found. If `val` is a
        list of values then a tuple of ndarray with the indices of each value
        is return.

    See also
    --------
    find_nearest2

    """
    idx = []
    if np.ndim(val) == 0:
        val = np.array([val])
    for v in val:
        idx.append((np.abs(arr-v)).argmin())
    idx = np.unravel_index(idx, arr.shape)
    return idx if len(idx) > 1 else idx[0]


print('loading CUBE FLOAT x/y/t ...')
t_cube, x_cube, y_cube = h5read(fcube, [tcube, xcube, ycube])

print('loading CUBE GSFC x/y/t/z ...')
t_gsfc, x_gsfc, y_gsfc, gsfc = ncread(fgsfc, [tgsfc, xgsfc, ygsfc, zgsfc])

# Rotate 90 deg counter clockwise: (t, x, y) -> (t, y, x)
gsfc = np.rot90(gsfc, axes=[1, 2])
x_gsfc = np.rot90(x_gsfc)
y_gsfc = np.rot90(y_gsfc)

# 2D -> 1D coords
x_gsfc = x_gsfc[0, :]
y_gsfc = y_gsfc[:, 0]

# Set NaNs to zero as there are missing valid areas
gsfc[np.isnan(gsfc)] = 0.0

print('subsetting in time ...')
k, = np.where((t_gsfc > t1) & (t_gsfc < t2))
gsfc = gsfc[k[0]:k[-1], :, :]
t_gsfc = t_gsfc[k[0]:k[-1]]

print('averaging in time ...')
#da_gsfc = da_gsfc.rolling(time=window, center=True).mean()
out = gsfc.copy()
running_mean_axis0(gsfc, window, out)
gsfc = out

print('regridding in time ...')
#da_gsfc = da_gsfc.interp(time=t_cube)
gsfc = regrid_nearest_axis0(gsfc, t_gsfc, t_cube)
t_gsfc = t_cube

print('creating xarray ...')
da_gsfc = xr.DataArray(gsfc, dims=('time', 'y', 'x'),
        coords={'time': t_gsfc, 'y': y_gsfc, 'x': x_gsfc})

print('extending spatial boundaries ...')
for k in range(t_cube.size):
    da_gsfc.values[k, :, :] = interpolate_replace_nans(
            da_gsfc.values[k, :, :], Gaussian2DKernel(1), boundary='extend')

print('regridding in space ...')
da_gsfc = da_gsfc.interp(x=x_cube, y=y_cube)

#print('masking grounded ...')
#mask, = h5read(fcube, ['grounded_mask'])
#da_gsfc.coords['mask'] = (('y', 'x'), mask)
#da_gsfc = da_gsfc.where(da_gsfc.mask == 1)

# Save data to original cube file
da_gsfc = da_gsfc.transpose('y', 'x', 'time')  # (t,y,x) -> (y,x,t)
#h5save(fcube, {saveas: da_gsfc.values}, 'a')
h5save('FAC_GSFC.h5', {saveas: da_gsfc.values, 'x': da_gsfc.x.values,
       'y': da_gsfc.y.values, 't': da_gsfc.time.values}, 'w')
print('saved ->', fcube)

#--- Plot ----------------------------------------------

plot = False
if plot:

    A = h5read('FAC_GSFC.h5', ['fac_gsfc'])
    B = h5read(fcube, ['fac_imau'])
    C = h5read(fcube, ['fac_gemb'])
    t = h5read(fcube, ['t'])

    #i, j, name = 770, 365, 'PIG Grounded'
    i, j, name = 831, 364, 'PIG Floating'

    a = A[i,j,:]
    b = B[i,j,:]
    c = C[i,j,:]

    a -= np.nanmean(a)
    b -= np.nanmean(b)
    c -= np.nanmean(c)

    plt.plot(t, a, label='FAC GSFC')
    plt.plot(t, b, label='FAC IMAU')
    plt.plot(t, c, label='FAC GEMB')
    plt.ylabel('FAC (m)')
    plt.legend()
    plt.show()

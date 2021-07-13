# -*- coding: utf-8 -*-
"""
Regrid and extend IMAU FAC cube (m) to height cube.

Notes:
    FAC should be applied to height (h), not to thickness (H).
    FAC is not hydrostatically compensated so: altim dh = full dFAC + change.
    Before applying FAC to h all time series must be referenced to the same t.

"""
import warnings

import sys
import h5py
import pyproj
import numpy as np
import xarray as xr
import netCDF4 as nc
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import griddata
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

warnings.filterwarnings('ignore')

#=== EDIT HERE ==================================================

ffac = '/Users/paolofer/data/firn/FDM_FirnAir_ANT27_1979-2016.nc'
tvar = 'time'
xvar = 'lon'
yvar = 'lat'
zvar = 'FirnAir'

ferr = '/Users/paolofer/data/firn/IMAU-FDM_ANT27_FAC_uncertainty.nc'
zerr = 'FAC_sd'

fcube = 'cube_full/FULL_CUBE_v2.h5'
tcube = 't'
xcube = 'x'
ycube = 'y'

saveas = 'fac_imau'

# Averaging window (months)
window = 5

# Time interval for subsetting
t1, t2 = 1991.0, 2019.5

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
            f[k] = np.squeeze(v)


def day2dyr(time, since='1950-01-01 00:00:00'):
    """ Convert days since epoch to decimal years. """
    t_ref = Time(since, scale='utc').jd  # convert days to jd
    return Time(t_ref + time, format='jd', scale='utc').decimalyear


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Polar Stereo AnIS (x/y): 3031
        Polar Stereo GrIS (x/y): 3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    return pyproj.transform(proj1, proj2, x, y)


@jit(nopython=True)
def running_mean_axis0(cube, window, out):
    """Fast moving average for 3D array along first dimension.

    Make out = cube.copy() to keep original values at the ends (half window).
    """

    #assert window % 2 > 0, 'Window must be odd!'
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


def regrid3d(x1, y1, z1, x2, y2):
    """Regrid Z1(t,y,x) onto Z2(y,x,t), keeping the original time steps.

    Args:
        x1/y1/x2/y2 are 2D arrays.
        z1 is a 3D array.

    Notes:
        This only works for coords in 2D (for 1D see interp3d).
        It reverses the time dimension (0-axis -> 2-axis).
    """
    z2 = np.full((x2.shape[0], x2.shape[1], z1.shape[0]), np.nan)  # -> (y,x,t)
    for k in range(z1.shape[0]):
        print('regridding:', k, '/', z1.shape[0])
        z2[:,:,k] = griddata((x1.ravel(), y1.ravel()),
                              z1[k,:,:].ravel(),
                              (x2, y2),
                              fill_value=np.nan,
                              method='linear')
    return z2


def regrid2d(x1, y1, z1, x2, y2):
    """Regrid Z1(y,x) onto Z2(y,x).

    Args:
        x1/y1/z1/x2/y2 are 2D arrays.

    Notes:
        This only works for coords in 2D (for 1D see interp2d).
    """
    z2 = np.full((x2.shape[0], x2.shape[1]), np.nan)  # -> (y,x)
    z2[:,:] = griddata((x1.ravel(), y1.ravel()),
                        z1.ravel(),
                        (x2, y2),
                        fill_value=np.nan,
                        method='linear')
    return z2


print('loading CUBE file ...')
t_cube, x_cube, y_cube = h5read(fcube, [tcube, xcube, ycube])

print('loading FAC file ...')
t_fac, lon_fac, lat_fac, fac = ncread(ffac, [tvar, xvar, yvar, zvar])
err = ncread(ferr, [zerr])

# Subset FAC in time
kk, = np.where((t_fac > t1) & (t_fac < t2))
t_fac, fac = t_fac[kk], fac[kk,:,:]

if np.ndim(lon_fac) == 1:
    lon_fac, lat_fac = np.meshgrid(lon_fac, lat_fac)  # 1d -> 2d

# Transform geodetic -> polar stereo
#NOTE: Interpolation should be done on x/y stereo proj
X_fac, Y_fac = transform_coord(4326, 3031, lon_fac, lat_fac)  # 2d
X_cube, Y_cube = np.meshgrid(x_cube, y_cube)  # 2d

print('averaging in time ...')
#fac = running_mean_cube(fac, window, axis=0)
out = fac.copy()  #NOTE: Important to avoid garbage at the end
running_mean_axis0(fac, window, out)
fac = out

print('regridding in time ...')
da_fac = xr.DataArray(fac, [('t', t_fac),
                            ('y', range(fac.shape[1])),  # dummy coords
                            ('x', range(fac.shape[2]))])

fac = da_fac.interp(t=t_cube).values

rmse = np.zeros(fac.shape[1:])

fit_model = True
if fit_model:

    # Add extra time steps to test model extrapolation
    #dt = t_cube[1]-t_cube[0]
    #t_end = t_cube[-1]
    #t_cube = np.r_[t_cube, [t_end+dt, t_end+dt*2, t_end+dt*3, t_end+dt*4]]
    #fac = np.r_[fac, np.full((4, fac.shape[1], fac.shape[2]), np.nan)]

    print('fitting model ...')

    # Design matrix elements
    a0 = np.ones(len(t_cube))    # intercept (t0)
    a1 = t_cube - t_cube.mean()  # trend (dt)
    a3 = np.sin(2*np.pi*a1)      # seasonal sin
    a4 = np.cos(2*np.pi*a1)      # seasonal cos

    # Trend + seasonal model
    A = np.vstack((a0, a1, a3, a4)).T

    #NOTE: The original dimensions are (t,x,y)
    for i in range(fac.shape[1]):
        for j in range(fac.shape[2]):
            z = fac[:, i, j]

            i_valid = np.isfinite(z)
            if sum(i_valid) < 10:
                continue

            # Interpolate NaNs
            #z = np.interp(t_cube, t_cube[i_valid], z[i_valid])

            z_orig = z.copy()
            z_grad = np.gradient(z)  #NOTE: Leave dt=1

            # Fixed trend+amplitude model fit
            try:
                # Robust least squares
                fit = (sm.RLM(z_grad, A, missing='drop')
                        .fit(maxiter=3, tol=0.001))
                #fit = (sm.OLS(z_grad, A, missing='drop')
                #        .fit(maxiter=3, tol=0.001))
            except:
                plt.plot(t_cube, z)
                plt.show()
                print('SOMETHING WRONG WITH THE FIT... SKIPPING CELL!!!')
                continue

            coef = fit.params                         # coeffs
            std = fit.bse                             # std err
            resid = fit.resid                         # data - model
            rmse_ = np.nanstd(resid, ddof=1)          # RMSE of fit
            #r2 = np.full_like(t_cube, fit.rsquared)  # only for OLS
            #model = fit.fittedvalues                 # fitted model

            # Use fitted coeffs to predict values
            dt = t_cube - np.nanmean(t_cube)
            a0, a1, a3, a4 = coef
            model = a0 + a1*dt + a3*np.sin(2*np.pi*dt) + a4*np.cos(2*np.pi*dt)

            z_grad_ext = z_grad.copy()
            z_grad_ext[np.isnan(z_grad)] = model[np.isnan(z_grad)]
            z_cums_ext = np.cumsum(z_grad_ext)  # integrate
            bias = np.nanmean(z_orig - z_cums_ext)  # offset
            z_cums_ext += bias

            z_ext = z_orig.copy()
            z_ext[np.isnan(z_orig)] = z_cums_ext[np.isnan(z_orig)]

            fac[:,i,j] = z_ext
            rmse[i,j] = rmse_

            # Plot
            '''
            plt.figure()
            plt.plot(t_cube, model, label='model', color='r')
            plt.plot(t_cube, z_grad, label='data')
            plt.title('Derivative (m/yr)')
            plt.legend()

            plt.figure()
            plt.plot(t_cube, z_ext, label='model', color='r')
            plt.plot(t_cube, z_orig, label='data')
            plt.title('Cummulative (m)')
            plt.legend()
            plt.show()
            #sys.exit()
            continue
            '''

#NOTE: This is important for IMAU FAC (coarse resolutin)
# Extend outer boundary (to complete ice-shelf coverage)
print('extending boundaries ...')
for k in range(fac.shape[0]):
    fac[k, :, :] = interpolate_replace_nans(fac[k, :, :],
            Gaussian2DKernel(1), boundary='extend')
    err = interpolate_replace_nans(err,
            Gaussian2DKernel(1), boundary='extend')
    rmse = interpolate_replace_nans(rmse,
            Gaussian2DKernel(1), boundary='extend')

if 0:
    #FIXME: Always check if this is needed <<<<<<
    for k in range(fac.shape[0]):
        fac[k,:,:] = np.flipud(fac[k,:,:])
    Y_fac = np.flipud(Y_fac)

print('regridding in space ...')
fac = np.ma.masked_invalid(fac)
fac = regrid3d(X_fac, Y_fac, fac, X_cube, Y_cube)

err = np.ma.masked_invalid(err)
err = regrid2d(X_fac, Y_fac, err, X_cube, Y_cube)

rmse = np.ma.masked_invalid(rmse)
rmse = regrid2d(X_fac, Y_fac, rmse, X_cube, Y_cube)

plt.matshow(fac[:,:,10])
plt.show()

print('saving data ...')
h5save('FAC_IMAU.h5', {'fac': fac, 'x': x_cube, 'y': y_cube, 't': t_cube,
                       'std': err, 'rmse': rmse}, 'w')

h5save(fcube, {saveas: fac, saveas+'_std': err, saveas+'_rmse': rmse}, 'a')
print('data saved ->', fcube)

#--- Plot ----------------------------------------------

plot = True
if plot:

    # Load
    fac, t = h5read('FAC_IMAU.h5', ['fac', 't'])
    cube = h5read(fcube, ['dh_xcal_filt'])

    #i, j, name = 836, 368, 'PIG'
    i, j, name = 366, 147, 'Larsen C'
    #i, j, name = 510, 1600, 'Amery'

    a = fac[i,j,:]
    b = cube[i,j,:]

    a -= np.nanmean(a)
    b -= np.nanmean(b)

    a[np.isnan(a)] = 0.

    b_cor = b - a
    b_cor -= b_cor[k]

    plt.plot(t, a, linewidth=1, label='FAC', color='k')
    plt.plot(t, b, linewidth=2, label='h_unc')
    plt.plot(t, b_cor, linewidth=2, label='h_cor')
    plt.title('FAC - %s' % name)
    plt.ylabel('meters')
    plt.legend()

    plt.show()

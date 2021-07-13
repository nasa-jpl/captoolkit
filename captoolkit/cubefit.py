"""
Fit trend model to 3d time series.

TODO:
    - Specify time interval for mean trend map?

"""
import sys
import h5py
import pyproj
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

#--- EDIT ----------------------------------------

if 1:
    files = sys.argv[1:]  # for multiple cube files
else:
    files = ['/Users/paolofer/data/cryosat2/floating/ointerp/CS2_CUBE.h5']

xvar = 'x'
yvar = 'y'
tvar = 't'
hvar = 'h'

#-------------------------------------------------


def polyfit(t, ts, order=1):
    ii = np.isfinite(ts)
    if sum(ii) < 5: return np.repeat(0, order+1)
    t_, ts_ = t[ii], ts[ii]
    ts_mean = np.mean(ts_)
    t_ -= np.mean(t_)
    ts_ -= ts_mean
    coefs = poly.polyfit(t_, ts_, order)
    coefs[0] += ts_mean
    return coefs


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


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


#------------------------------------------------


# Get ice shelf mask
"""
if 1:
    with h5py.File('ishelf_mask_itslive_3km.h5', 'r') as f:
        masks = [f['mask'][:]]
else:
    masks = []
    for fname in files:
        with h5py.File(fname+'_mask', 'r') as f:
            masks.append(np.flipud(f['mask'][:]))

# Get latitudinal mask
if 1:
    with h5py.File(files[0], 'r') as f:
        x, y = f['x'][:], f['y'][:]
        X, Y = np.meshgrid(x, y)
        lon, lat = transform_coord(3031, 4326, X, Y)
        mask2 = (lat < -81.5)
"""

""" Time filtering (time series) """

for fname in files:
    with h5py.File(fname, 'r') as f:
    
        print('file:', fname)
        x = f[xvar][:]
        y = f[yvar][:]
        t = f[tvar][:]
        Z = f[hvar][:]
    
        bias =np.zeros_like(Z[:,:,0])
        trend =np.zeros_like(Z[:,:,0])
        accel = np.zeros_like(Z[:,:,0])
    
        for j in range(Z.shape[1]):
            for i in range(Z.shape[0]):
                print('grid cell:', i, j)
    
                ts = Z[i,j,:]
    
                if sum(np.isfinite(ts)) < 5: continue
    
                try:
                    a0, a1, a2 = polyfit(t, ts, order=2)
                except:
                    continue

                trd = poly.polyval(t, [a0,a1,a2])

                '''
                plt.plot(t, ts)
                plt.plot(t, trd)
                plt.show()
                '''
    
                bias[i,j] = a0
                trend[i,j] = a1
                accel[i,j] = a2
    
        plt.matshow(trend, vmin=-.2, vmax=.2, cmap='RdYlBu')
        plt.colorbar()
        plt.matshow(accel, vmin=-.02, vmax=.02, cmap='RdYlBu')
        plt.colorbar()
        plt.show()

    if 0:
        with h5py.File(fname, 'a') as f:
            f[hvar+'_bias'] = bias
            f[hvar+'_trend'] = trend
            f[hvar+'_accel'] = accel

    print('fields saved.')

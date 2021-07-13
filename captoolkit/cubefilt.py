
#TODO: Change the order of j/i -> i/j in the loops for speed !!!!!!

import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve


def polyfit(t, ts, order=1):
    """ """
    ii = np.isfinite(ts)
    t_, ts_ = t[ii], ts[ii]
    coef = np.polyfit(t_, ts_, order)
    trend = np.polyval(coef, t)
    return trend


def stdfilt(t, ts, nstd=3, order=1):
    """ """
    ts_ = ts.copy()
    trend = polyfit(t, ts_, order)
    res = ts_ - trend
    ii, = np.where(np.abs(res) > mad_std(res) * nstd)
    ts_[ii] = np.nan
    return ts_


def medfilt(t, ts, n=3):
    """ """
    ts_ext = np.r_[ts[n-1], ts, ts[-n]]
    ts_ = ts.copy()
    for i in range(ts_.shape[0]):
        i1, i2 = i, i+2
        ts_[i] = np.nanmean(ts_ext[i1:i2])   #FIXME: It's using Mean?!
    return ts_


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

def main():
    """ """

    files = sys.argv[1:]

    kernel_med = 5
    kernel_filt = Gaussian2DKernel(1)
    kernel_fill = Gaussian2DKernel(3)

    # Get ice shelf mask
    if 1:
        with h5py.File('data/ishelf_mask_itslive_3km.h5', 'r') as f:
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


    for k,fname in enumerate(files):
        #mask = masks[k]  # get one mask per file/cube
        mask = masks[0]

        """ Spatial filtering (fields) """

        if 1:
            with h5py.File(fname, 'a') as f:

                print('file:', fname)

                Z = f['h_res'][:]  # field to filter

                Z[Z==0] = np.nan

                for k in range(Z.shape[2]):
                    print('filtering slice:', k)
                    zz = Z[:,:,k]

                    if 1: zz = ndi.median_filter(zz, kernel_med)                                     # filter

                    Z[:,:,k] = zz

                    if 1:
                        # Plot
                        plt.matshow(mask)
                        plt.matshow(zz, vmin=-.5, vmax=.5, cmap='RdBu')
                        plt.show()

                try:
                    f['h_res_filt'] = Z
                except:
                    f['h_res_filt'][:] = Z

        """ Time filtering (time series) """

        if 1:
            with h5py.File(fname, 'a') as f:

                print('file:', fname)
                try:
                    Z = f['h_res_filt'][:]
                except: 
                    Z = f['h_res'][:]

                t = f['t_year'][:]

                for j in range(Z.shape[1]):
                    for i in range(Z.shape[0]):
                        print('grid cell:', i, j)

                        ts = Z[i,j,:]
                        if sum(np.isfinite(ts)) < 5: continue
                        ts_orig = ts.copy()

                        ts = stdfilt(t, ts, nstd=5, order=2)
                        if sum(np.isfinite(ts)) == 0: continue

                        #FIXME 1: No need to interp all t's, only t[i_nan] -> y[i_nan]
                        #FIXME 2: Extend the end points propagating values from  a
                        # fitted trend (w/sgolay), not original end points (for non overlaps)
                        ts = np.interp(t, t[np.isfinite(ts)], ts[np.isfinite(ts)])
                        #ts = medfilt(t, ts)

                        Z[i,j,:] = ts

                        '''
                        plt.plot(t, ts_orig)
                        plt.plot(t, ts)
                        plt.show()
                        '''
                try:
                    f['h_res_filt'] = Z
                except:
                    f['h_res_filt'][:] = Z

        if 1:
            with h5py.File(fname, 'a') as f:

                print('file:', fname)

                try:
                    Z = f['h_res_filt'][:]
                    #Z = f['h_res'][:]
                except: 
                    Z = f['h_res'][:]

                for k in range(Z.shape[2]):
                    print('filtering slice:', k)
                    zz = Z[:,:,k]

                    if 0: zz = convolve(zz, kernel_filt, boundary='extend')

                    if 1: zz = interpolate_replace_nans(zz, kernel_fill, boundary='extend')         # fillin

                    if 'ER1' in fname or 'ER2' in fname or 'ENV' in fname: zz[mask2] = np.nan        # mask
                    if 1: zz[mask] = np.nan                                                          # mask
                    # Fillin gaps
                    if 0:
                        zz_ = zz.copy()
                        kernel_fill = Gaussian2DKernel(9)
                        for _ in range(4):
                            zz_ = interpolate_replace_nans(zz_, kernel_fill, boundary='extend')         # fillin
                        zz_ = ndi.median_filter(zz_, 5)                                     # filter
                        ij = np.isnan(zz)
                        zz[ij] = zz_[ij]

                        zz[mask] = np.nan                                                          # mask
                    Z[:,:,k] = zz

                    '''
                    plt.matshow(mask)
                    plt.matshow(zz, vmin=-.5, vmax=.5, cmap='RdBu')
                    plt.show()
                    '''

                try:
                    f['h_res_filt'] = Z
                except:
                    f['h_res_filt'][:] = Z

    print('done.')


main()

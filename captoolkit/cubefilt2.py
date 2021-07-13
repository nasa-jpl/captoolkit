import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import savgol_filter

import warnings
warnings.simplefilter("ignore")


FILE = ('/Users/paolofer/code/captoolkit/captoolkit/work/cube_full/'
        'FULL_CUBE_v2.h5')

VAR_IN = 'h'
VAR_OUT = 'h_filt'


def filter_series(y, window=5, order=1, nstd=5, plot=False):
    """Filter anomalous residuals w.r.t. the trend.

    1. Exclude constant values (dy/dx = 0) from the analysis.
    2. Detect gross outliers.
    3. Fit piece-wise polynomial trend ignoring gross outliers.
    4. Interpolate trend on gross outliers.
    5. Filter residuals w.r.t. to the trend. 
    6. Interpolate residuals on outliers. 
    7. Add back trend + residuals + constant values.
    """
    if np.sum(np.isfinite(y)) < 5: return y
    y_ = y.copy()
    x = np.arange(len(y))

    # Detect and exclude constant values
    dydx = np.gradient(y_)
    idx_const = (dydx == 0)
    y_[idx_const] = np.nan

    # Detect and exclude goss outliers
    idx_gross = (np.abs(y_ - np.nanmean(y_)) > mad_std(y_)*5)
    idx_valid = ~idx_const & ~idx_gross

    if np.sum(idx_valid) < window: return y

    # Compute trend on valid entries
    y_trend = np.full_like(y, np.nan)
    y_trend[idx_valid] = savgol_filter(y[idx_valid], window, 
                                       order, 0, mode='nearest')

    # Interpolate trend on gross outliers
    y_trend[idx_gross] = np.interp(x[idx_gross],
                                   x[idx_valid],
                                   y_trend[idx_valid])

    # y_trend contains the NaNs that should be ignored
    y_resid = y - y_trend   

    a = np.abs(y_resid) 
    b = mad_std(y_resid) * nstd
    c = ~np.isnan(y_resid)
    idx_outliers = (a > b) & c

    if np.sum(idx_outliers) != 0:

        # Remove and inerpolate filtered residuals
        idx_valid = ~idx_outliers & ~idx_const
        y_resid[idx_outliers] = np.interp(x[idx_outliers],
                                          x[idx_valid],
                                          y_resid[idx_valid])

        # Reconstruct series: trend + filtered resid
        y_resid[idx_const] = 0
        y_trend[idx_const] = y[idx_const]
        y_filt = y_trend + y_resid

    else:
        y_filt = y

    if plot:
        plt.subplot(311)
        plt.plot(y)
        plt.plot(y_trend)
        plt.subplot(312)
        plt.plot(y_resid)
        plt.subplot(313)
        plt.plot(y)
        plt.plot(y_filt)
        plt.show()

    return y_filt 


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def main():
    if FILE:
        fname = FILE
    else:
        fname = sys.argv[1]

    print('filtering series ...')

    with h5py.File(fname, 'a') as f:
        Z = f[VAR_IN][:]

        Z_filt = np.apply_along_axis(filter_series, 2, Z, plot=False)

        '''
        # For testing only
        #i, j  = 962,877
        i, j  = 1131,669
        z = Z[i,j,:]
        z[-5] *= 10
        z_filt = filter_series(z, plot=True)
        '''

        try:
            f[VAR_OUT] = Z_filt
        except:
            f[VAR_OUT][:] = Z_filt

        print('saved ->',  fname)

main()

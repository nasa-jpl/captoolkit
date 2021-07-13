#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-calibration and fusion of multi-mission altimetry time series.

Align post-processed time series (from different data cubes),
into a single cross-calibrate and fused cube.

"""

import argparse
import h5py
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings('ignore')


OFILE = 'cube_full/FULL_CUBE_v2.h5'

def get_args():
    """ Get command-line arguments. """
    description = ('Cross-calibration of individual data cubes')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
            'files', metavar='files', type=str, nargs='+',
            help='file(s) to process (HDF5)')
    parser.add_argument(
            '-t', metavar=('ref_time'), dest='tref', type=float, nargs=1,
            help=('time to reference the solution to (yr), optional'),
            default=[2014],)
    parser.add_argument(
            '-v', metavar=('x','y','t','h','e','i'), dest='vnames', type=str, nargs=6,
            help=('name of variables in the HDF5-file'),
            default=['x','y','t_year','h_res_filt','None','None'],)
    return parser.parse_args()


def sgolay1d(h, window=3, order=1, deriv=0, dt=1.0, mode='nearest', time=None):
    """Savitztky-Golay filter with support for NaNs

    If time is given, interpolate NaNs otherwise pad w/zeros.

    dt is spacing between samples.
    """
    h2 = h.copy()
    ii, = np.where(np.isnan(h2))
    jj, = np.where(np.isfinite(h2))
    if len(ii) > 0 and time is not None:
        h2[ii] = np.interp(time[ii], time[jj], h2[jj])
    elif len(ii) > 0:
        h2[ii] = 0
    else:
        pass
    h2 = savgol_filter(h2, window, order, deriv, delta=dt, mode=mode)
    return h2 


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def find_nearest(arr, val):
    """Find index for "nearest" value.
    
    Parameters
    ----------
    arr : array_like, shape nd
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
    if np.ndim(val) == 0: val = np.array([val]) 
    for v in val: idx.append((np.abs(arr-v)).argmin())
    idx = np.unravel_index(idx, arr.shape)
    return idx


def get_bias(t1, h1, t2, h2):
    t1_, h1_ = t1[np.isfinite(h1)], h1[np.isfinite(h1)]
    t2_, h2_ = t2[np.isfinite(h2)], h2[np.isfinite(h2)]
    inter = np.intersect1d(t1_, t2_)
    if len(inter) != 0:
        i1 = np.in1d(t1_, inter)
        i2 = np.in1d(t2_, inter)
    else:
        i1 = -1
        i2 = 0
    return np.median(h2[i2]-h1[i1])  # bias


def crosscal(tcap, hcap, mcap, window=5):      ##FIXME: Test window=3
    """Separate missions, smooth and cross-calibrate.
    
    Args:
        tcap, hcap, mcap: 1D arrays containing all missions
            (a single grid cell).
    Return:
        dict1, dict2, dict3: 3 dicts {mission_id: value} with
            time, calibrated h and h_smoothed.
    """
    times, horig, hsmooth = {}, {}, {}

    # Separate and smooth 
    for m in np.unique(mcap):
        i_ts = (mcap==m)
        t, h = tcap[i_ts], hcap[i_ts]
        idx = np.isfinite(h)
        if sum(idx) < 5: continue
        #NOTE: For deriv=0 no need dt
        h_smooth = sgolay1d(h, window=window, order=1, deriv=0, time=t)
        times[int(m)] = t
        horig[int(m)] = h 
        hsmooth[int(m)] = h_smooth 

    # Cross-calibrate 
    if len(list(times.keys())) > 1:  # find offset
        keys = list(times.keys())
        for k in range(len(keys)-1):
            key1, key2 = keys[k], keys[k+1]
            t1, t2 = times[key1], times[key2]
            h1, h2 = hsmooth[key1], hsmooth[key2]
            bias = get_bias(t1, h1, t2, h2)
            horig[key2] -= bias
            hsmooth[key2] -= bias

    return times, horig, hsmooth


def align(t_uniq, tdict, hdict):
    """Align time series in dicts into a matrix (t_uniq x n_sats).""" 
    m_uniq = [m for m in list(tdict.keys())]
    hh = np.full((len(t_uniq), len(m_uniq)), np.nan)  # 2D (1 sat/col)

    for j, m in enumerate(m_uniq):
        t, h = tdict[m], hdict[m]
        ii = np.in1d(t_uniq, t) 
        if len(ii) == 0: continue
        hh[ii,j] = h
    return hh


def fuse(t_uniq, hh, ref_epoch=2014):
    """Merge time series by median of overlap, and reference to epoch.

    Args:
        t_uniq: 1D array with continous (cube) full time.
        hh: 2D array with 1 sat/column (t_uniq.shape[0] == hh.shape[0]).
    """
    h_uniq = np.nanmedian(hh, axis=1)

    i_valid = np.isfinite(h_uniq) 
    i_invalid = np.invert(i_valid)
    h_uniq[i_invalid] = np.interp(t_uniq[i_invalid], t_uniq[i_valid], h_uniq[i_valid])

    i_ref = find_nearest(t_uniq, ref_epoch)
    return h_uniq - h_uniq[i_ref]


# Main program
def main(args):

    # Pass arguments to internal variables
    files = args.files
    tref = args.tref[0]
    vnames = args.vnames[:]

    print('parameters:')
    for p in vars(args).items(): print(p)

    # Input variables names
    xvar, yvar, tvar, zvar, evar, ivar = vnames

    # If cubes for each mission are in separate files,
    # concatenate them and generate a single cube.
    # Each mission (on individual file) will be given a unique identifier.
    for nf, ifile in enumerate(files):
        print('processing file:', ifile, '...')

        if nf == 0:
            with h5py.File(ifile, 'r') as fi:
                x = fi[xvar][:]     # 1d
                y = fi[yvar][:]     # 1d  
                time = fi[tvar][:]  # 1d
                elev = fi[zvar][:]  # 3d
                mode = fi[ivar][:] if ivar in fi \
                        else np.full_like(time, nf)  # 1d
                sigma = fi[evar][:] if evar in fi \
                        else np.full_like(elev, np.nan)  # 3d
        else:
            with h5py.File(ifile, 'r') as fi:
                time = np.hstack((time, fi[tvar][:]))  # 1d
                elev = np.dstack((elev, fi[zvar][:]))  # 3d
                mode = np.hstack((mode, fi[ivar][:] if ivar in fi \
                        else np.full_like(fi[tvar][:], nf)))  # 1d
                sigma = np.dstack((sigma, fi[evar][:] if evar in fi \
                        else np.full_like(fi[zvar][:], np.nan)))  # 3d

    '''
    plt.matshow(elev[:,:,5], vmin=-.25, vmax=.25, cmap='RdBu')
    plt.show()
    sys.exit()
    '''

    if len(np.unique(mode)) < 2:
        print('it seems the files contain only one mission!')
        return

    # Output containers
    Z1 = np.full_like(elev, np.nan)
    Z2 = np.full_like(elev, np.nan)
    ei = np.full_like(elev, np.nan)
    ni = np.full_like(elev, np.nan)
    ti = time.copy()
    mi = mode.copy()

    t_uniq = np.unique(time)
    cube_xcal = np.full((y.shape[0], x.shape[0], t_uniq.shape[0]), np.nan)

    # Enter prediction loop
    for i in range(elev.shape[0]):
        for j in range(elev.shape[1]):
            
            if (i+j) % 10 == 0: print(i,j)
                
            #NOTE: Just for plotting
            #print i, j
            #if i % 100 != 0: continue
            #i, j = 845, 365
            #i, j = 153, 1361
            #i, j = 1022, 840 
            #i, j = 970, 880  # CS-2 only
            #i, j = 100, 1170  # fig1
            #i, j = 100, 766  # fig2

            # Parameters for model-solution
            tcap = time[:]
            mcap = mode[:]
            hcap = elev[i,j,:]
            scap = sigma[i,j,:]

            if sum(np.isfinite(hcap)) < 10: continue


            #NOTE: Double check this works for all necessary case.
            try:
                times, horig, hsmooth = crosscal(tcap, hcap, mcap)  # 1D arr -> dict
                hh = align(t_uniq, times, horig)                    # dict -> 2D arr
                h_uniq = fuse(t_uniq, hh, ref_epoch=tref)           # 2D arr -> 1D arr
            except:
                print("Something didn't work, skipping!!!")
                continue

            if 0:
                plt.plot(t_uniq, np.gradient(h_uniq))
                plt.show()
                continue

            if sum(np.isfinite(h_uniq)) == 0: continue

            # For plotting only
            dhdt_uniq = sgolay1d(h_uniq, window=5, order=1, deriv=1, dt=t[1]-t[0])
            dhdt_uniq2 = sgolay1d(h_uniq, window=11, order=1, deriv=1, dt=t[1]-t[0])

            # Dict{mission: tseries} -> List[tseries1 + tseries2 + ..]
            h_orig, h_smooth = [], []
            [h_orig.extend(hi) for hi in list(horig.values())]
            [h_smooth.extend(hi) for hi in list(hsmooth.values())]

            cube_xcal[i,j,:] = h_uniq

            for k_, m_ in enumerate(np.unique(mcap)):
                kk, = np.where(m_ == mcap)
                Z1[i,j,kk] = h_orig[k_] 
                Z2[i,j,kk] = h_smooth[k_]


            # Plot
            if 0:
                plt.figure(figsize=(8,3.5))
                plt.ylabel('H(t), m')
                for k in list(times.keys()):
                    t, ho = times[k], horig[k]
                    t, hf = times[k], hsmooth[k]
                    ho *= 9.2
                    hf *= 9.2
                    plt.plot(t, ho, linewidth=2.5)
                    plt.plot(t, hf, '0.4')
                #plt.savefig('height_change_tseries.png')

                h_uniq = cube_xcal[i,j,:]

                plt.figure(figsize=(8,7))
                '''
                plt.subplot(211)
                plt.plot(t_uniq, h_uniq, linewidth=3)
                plt.ylabel('H(t), m')
                '''
                dhdt_uniq *= 9.2
                dhdt_uniq *= 9.2
                plt.subplot(211)
                plt.plot(t_uniq, dhdt_uniq, linewidth=3)
                plt.ylabel('dH/dt(t), m/yr')
                plt.subplot(212)
                plt.plot(t_uniq, dhdt_uniq2, linewidth=3)
                plt.ylabel('dH/dt(t), m/yr')
                #plt.savefig('height_rate_tseries.png')
                plt.show()

    if 0:
        print('Saving data to file ...')
        
        with h5py.File(OFILE, 'a') as f:
            f['h'] = cube_xcal
            f['h_orig'] = Z1
            f['h_smooth'] = Z2
            f['t'] = t_uniq
            f['t_orig'] = ti
            f['sat_orig'] = mi  # sat id
            f['x'] = x
            f['y'] = y

        print('out ->', OFILE)
        return


args = get_args() 

main(args)

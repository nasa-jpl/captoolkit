#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filter satellite tracks (segments) with along-track running window.

Two-step filtering: (1) 3-point median, (2) hanning-window weighted sum.

This version also filters along-track radar-waveform parameters.

Notes:
    If 'apply' is selected (-a), the original (unfiltered) variables are
    renamed to 'var_unfilt', otherwise a filtered variable is created: 'var_filt'.

Example:
    python trackfilt_wf.py -f ~/data/ers2/floating/latest/AntIS_ERS2_OCN_READ_A_ROSS_RM_IBE_TIDE_MERGED.h5 
    -v t_sec h_cor bs lew tes -a -n 16

"""

##TODO: Revisit the end points after the hanning-window:
# Sometimes, they tend to be pushed up or down.
# (if processing tiles w/buffer this is not an issue) 

import os
import sys
import h5py
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import medfilt as median_filt


##NOTE: The two end weights in the window are zero!
# So the number of pts being averaged is N-2
WINDOW_SIZE = 11


def get_args():
    """ Get command-line arguments. """
    msg = 'Apply along-track running filter (Hanning window).'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
            '-f', metavar='file', dest='files', type=str, nargs='+',
            help='HDF5 file(s) to process',
            required=True)
    parser.add_argument(
            '-v', metavar=('t', 'h','bs', 'lew', 'tes'), dest='vnames',
            type=str, nargs=5,
            help=('name of vars to filter along-segment (t in secs, default)'),
            default=[None], required=True)
    parser.add_argument(
            '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
            help="number of jobs for parallel processing",
            default=[1],)
    parser.add_argument(
            '-a', dest='apply', action='store_true',
            help=('apply correction to height in addition to saving'),
            default=False)
    parser.add_argument(
            '-p', dest='plot', action='store_true',
            help=('plot random segments, do not save any data'),
            default=False)
    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def get_segments(time, tmax=10):
    """
    Partition time array into segments with time breaks > tmax.

    Returns an array w/unique identifiers (per point) for each segment.
    """
    n = 0
    trk = np.zeros(time.shape)
    for k in xrange(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax:
            n += 1
        trk[k] = n
    return trk


def hanning_filt(h, window_size=9):

    #NOTE: The weights are normalized to sum to 1
    window = np.hanning(window_size) / float(np.hanning(window_size).sum())

    h_filt = np.convolve(h, window, mode='same')

    #TODO: This is a temporary fix. Idealy, this filter should be applied
    # to the full tracks before masking, so end-point effects wont matter.
    # For now, fill in the end points (the missing half window)
    # (if processing tiles w/buffer this is not an issue)
    n_half = int(window_size/2.) - 2
    if 1:
        h_filt[:n_half], h_filt[-n_half:] = h[:n_half], h[-n_half:]
    elif 0:
        h_filt[:n_half], h_filt[-n_half:] = np.nan, np.nan
    else:
        pass

    return h_filt


def plot(t_seg, x_seg, y_seg, h_seg, h_filt,
        b_seg, b_filt, w_seg, w_filt, s_seg, s_filt): 

    plt.figure()
    plt.plot(x_seg, y_seg, '.', rasterized=True)
    plt.xlabel('Longitude (deg)')
    plt.xlabel('Latitude (deg)')
    
    plt.figure()
    plt.subplot(411)
    plt.plot(t_seg, h_seg, '.', rasterized=True)
    plt.plot(t_seg, h_filt, '.', rasterized=True)
    plt.ylabel('Height (m)')
    
    plt.subplot(412)
    plt.plot(t_seg, b_seg, '.', rasterized=True)
    plt.plot(t_seg, b_filt, '.', rasterized=True)
    plt.ylabel('Bs (dB)')
    
    plt.subplot(413)
    plt.plot(t_seg, w_seg, '.', rasterized=True)
    plt.plot(t_seg, w_filt, '.', rasterized=True)
    plt.ylabel('LeW (m)')
    
    plt.subplot(414)
    plt.plot(t_seg, s_seg, '.', rasterized=True)
    plt.plot(t_seg, s_filt, '.', rasterized=True)
    plt.ylabel('TeS (slope)')
    plt.xlabel('Time along-track (sec)')
    
    plt.show()


def main(ifile, vnames, apply_=False, plot_=False):

    print 'processing file:', ifile, '...'
    
    tvar, hvar, bvar, wvar, svar = vnames

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:
        t = fi[tvar][:]
        h = fi[hvar][:]
        b = fi[bvar][:]
        w = fi[wvar][:]
        s = fi[svar][:]

        if plot_:
            # To plot the tracks
            x = fi['lon'][:]
            y = fi['lat'][:]

    # Generate output containers
    hfilt = np.full_like(h, np.nan)
    bfilt = np.full_like(b, np.nan)
    wfilt = np.full_like(w, np.nan)
    sfilt = np.full_like(s, np.nan)

    # Get segment numbers (break when dt > 0.1 sec)
    segments = get_segments(t, tmax=0.1)
    segments_unique = np.unique(segments)
    n_segments = len(segments_unique)

    if plot_:
        # Shuffle to plot random segments
        random.shuffle(segments_unique)

    for k, segment in enumerate(segments_unique):

        if k%1000 == 0:
            print 'segment #:', int(segment), 'of', n_segments

        # Select points from each segment
        ii, = np.where( (segments == segment) & ~np.isnan(h) )

        if len(ii) < WINDOW_SIZE:
            continue
        
        t_seg = t[ii]
        h_seg = h[ii]
        b_seg = b[ii]
        w_seg = w[ii]
        s_seg = s[ii]

        if plot_:
            x_seg = x[ii]
            y_seg = y[ii]

        """ Two-step along-segment filtering. """

        h_seg = median_filt(h_seg, kernel_size=3)
        b_seg = median_filt(b_seg, kernel_size=3)
        w_seg = median_filt(w_seg, kernel_size=3)
        s_seg = median_filt(s_seg, kernel_size=3)

        h_filt = hanning_filt(h_seg, WINDOW_SIZE)
        b_filt = hanning_filt(b_seg, WINDOW_SIZE)
        w_filt = hanning_filt(w_seg, WINDOW_SIZE)
        s_filt = hanning_filt(s_seg, WINDOW_SIZE)

        hfilt[ii] = h_filt
        bfilt[ii] = b_filt
        wfilt[ii] = w_filt
        sfilt[ii] = s_filt

        if plot_ and len(ii) > 100:
            plot(t_seg, x_seg, y_seg, h_seg, h_filt,
                    b_seg, b_filt, w_seg, w_filt, s_seg, s_filt)
            continue

    print 'saving data ...'

    if not plot_:
        with h5py.File(ifile, 'a') as fi:
            if apply_:
                # Copy original
                fi[hvar+'_unfilt'] = fi[hvar][:]
                fi[bvar+'_unfilt'] = fi[bvar][:]
                fi[wvar+'_unfilt'] = fi[wvar][:]
                fi[svar+'_unfilt'] = fi[svar][:]
                fi.flush()
                # Update original
                fi[hvar][:] = hfilt
                fi[bvar][:] = bfilt
                fi[wvar][:] = wfilt
                fi[svar][:] = sfilt
                fi.flush()
            else:
                fi[hvar+'_filt'] = hfilt
                fi[bvar+'_filt'] = bfilt
                fi[wvar+'_filt'] = wfilt
                fi[svar+'_filt'] = sfilt
                fi.flush()
            
        # Rename file
        os.rename(ifile, ifile.replace('.h5', '_FILT.h5'))


if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.files[:]   # input files
    vnames = args.vnames[:]  # lon/lat/h/time variable names
    njobs = args.njobs[0]    # parallel writing
    apply_ = args.apply      # Apply cor in addition to saving
    plot_ = args.plot        # Apply cor in addition to saving

    print_args(args)
    
    # In case a string is passed to avoid "Argument list too long"
    if len(ifiles) == 1:ifiles = glob(ifiles[0])

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, vnames, apply_, plot_) for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, vnames, apply_, plot_) for ifile in ifiles)

    print 'done!'

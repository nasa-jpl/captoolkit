#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filter segments with along-segment running window (hanning-window weighted sum).

Example:


Notes:
    If 'apply' is selected (-a), the original (unfiltered) variables are
    renamed to 'var_unfilt', otherwise a filtered variable is created 'var_filt'.

"""

import os
import sys
import h5py
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


#NOTE: The two end weights in the window are zero!
# So number of pts being averaged is N-2
WINDOW_SIZE = 9


def get_args():
    """ Get command-line arguments. """

    msg = 'Apply along-track running filter (Hanning window).'
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument(
            '-f', metavar='file', dest='files', type=str, nargs='+',
            help='HDF5 file(s) to process',
            required=True)

    parser.add_argument(
            '-v', metavar=('t', 'h'), dest='vnames',
            type=str, nargs=2,
            help=('name of variables to filter along-segment in the HDF5'),
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


def plot(t_seg, x_seg, y_seg, h_seg, h_filt):
        

    plt.figure()
    plt.plot(x_seg, y_seg, '.', rasterized=True)
    plt.xlabel('Longitude (deg)')
    plt.xlabel('Latitude (deg)')
    
    plt.figure()
    plt.plot(t_seg, h_seg, '.', rasterized=True)
    plt.plot(t_seg, h_filt, '.', rasterized=True)
    plt.ylabel('Height (m)')
    plt.xlabel('Time along-track (sec)')
    
    plt.show()


def main(ifile, vnames, apply_=False, plot_=False):

    print 'processing file:', ifile, '...'
    
    tvar, hvar = vnames

    # Load full data into memory (only once)
    with h5py.File(ifile, 'r') as fi:

        t = fi[tvar][:]
        h = fi[hvar][:]

        if plot_:
            # To plot the tracks
            x = fi['lon'][:]
            y = fi['lat'][:]

    # Generate output containers
    hfilt = np.full_like(h, np.nan)

    # Get segment numbers (break when dt > 0.1 sec)
    segments = get_segments(t, tmax=0.1)
    segments_unique = np.unique(segments)
    n_segments = len(segments_unique)

    if plot_:
        # Shuffle to plot random segments
        random.shuffle(segments_unique)

    for k, segment in enumerate(segments_unique):

        if k%100 == 0:
            print 'segment #:', int(segment), '/', n_segments

        # Select points from each segment
        ii, = np.where( (segments == segment) & ~np.isnan(h) )

        if len(ii) < WINDOW_SIZE:
            continue
        
        t_seg = t[ii]
        h_seg = h[ii]

        if plot_:
            x_seg = x[ii]
            y_seg = y[ii]

        #NOTE: The weights are normalized to sum to 1
        window = np.hanning(WINDOW_SIZE) / float(np.hanning(WINDOW_SIZE).sum())

        h_filt = np.convolve(h_seg, window, mode='same')

        #TODO: This is a temporary fix. Idealy, this filter should be applied
        # to the full tracks before masking, so end-point effects wont matter.
        # Fill in the end points (the missing half window)
        n_half = int(WINDOW_SIZE/2.) - 2
        if 0:
            h_filt[:n_half], h_filt[-n_half:] = h_seg[:n_half], h_seg[-n_half:]
        elif 1:
            h_filt[:n_half], h_filt[-n_half:] = np.nan, np.nan
        else:
            pass

        hfilt[ii] = h_filt

        if plot_ and len(ii) > 100:

            plot(t_seg, x_seg, y_seg, h_seg, h_filt)

            continue

    print 'saving data ...'

    with h5py.File(ifile, 'a') as fi:

        if apply_:
            # Copy original
            fi[hvar+'_unfilt'] = fi[hvar][:]
            fi.flush()

            # Update original
            fi[hvar][:] = hfilt
            fi.flush()

        else:
            fi[hvar+'_filt'] = hfilt
            fi.flush()
        

if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.files[:]   # input files
    vnames = args.vnames[:]  # lon/lat/h/time variable names
    njobs = args.njobs[0]    # parallel writing
    apply_ = args.apply      # Apply cor in addition to saving
    plot_ = args.plot        # Apply cor in addition to saving

    print_args(args)

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, vnames, apply_, plot_) for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, vnames, apply_, plot_) for ifile in ifiles)

    print 'done!'

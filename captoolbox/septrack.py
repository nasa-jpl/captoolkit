#!/usr/bin/env python
import sys
import h5py
import numpy as np

YVAR = 'lat'
TVAR = 't_sec'
TMAX = 10  # secs

files = sys.argv[1:]


def segment_number(time, tmax=1):
    """
    Partition time array into segments with breaks > tmax.

    Returns an array w/unique identifiers for each segment.
    """
    n = 0
    trk = np.zeros(time.shape)
    for k in xrange(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax:
            n += 1
        trk[k] = n
    return trk


def track_type(time, lat, tmax=1):
    """
    Determines ascending and descending tracks.

    Defines unique tracks as segments with time breaks > tmax,
    and tests whether lat increases or decreases w/time.
    """

    # Generate track numbers
    tracks = segment_number(time, tmax=tmax)

    # Output index array
    i_asc = np.zeros(tracks.shape, dtype=bool)

    # Loop trough individual tracks
    for track in np.unique(tracks):

        # Get all points from an individual track
        i_track, = np.where(track == tracks)

        # Test tracks length
        if len(i_track) < 2:
            continue
        
        # Test if lat increases (asc) or decreases (des) w/time
        i_min = time[i_track].argmin()
        i_max = time[i_track].argmax()
        lat_diff = lat[i_track][i_max] - lat[i_track][i_min]

        # Determine track type
        if lat_diff > 0:
            i_asc[i_track] = True

    # Output index vector's
    return i_asc, np.invert(i_asc)


for fname in files:

    with h5py.File(fname, 'r') as f:

        # Sort time 
        time = f[TVAR][:]
        lat = f[YVAR][:]

        # Get indices of asc and des tracks
        i_asc, i_des = track_type(time, lat, tmax=TMAX)

        print 'file:', fname
        print '# asc tracks:', len(time[i_asc])
        print '# des tracks:', len(time[i_des])

        # Save asc and des data in separate files
        with h5py.File(fname.replace('.h5', '_A.h5'), 'w') as fa, \
                h5py.File(fname.replace('.h5', '_D.h5'), 'w') as fd:

            for k in f.keys():
                var = f[k][:]
                fa[k] = var[i_asc]
                fd[k] = var[i_des]


        # Plot (for testing)
        if 0:
            lon = f['lon'][:]
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(lon[i_asc], lat[i_asc], '.')
            plt.figure()
            plt.plot(lon[i_des], lat[i_des], '.')
            plt.show()

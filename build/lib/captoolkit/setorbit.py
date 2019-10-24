#!/usr/bin/env python
"""
Calculate unique identifiers for each track (segments of data),
and update the 'orbit' variable in the HDF5.

Set 'key' (below) for sorting the input files according tile number. 

Set 'start' (below) for different track counting between missions.

"""
import sys
import h5py
import numpy as np


# Set keyword present in the file name to sort files by
key = 'tile'

# Start counting tracks from (to avoid repeated numbers between missions)
start = 70000


def segment_number(time, tmax=1):
    """
    Partition time array into segments with breaks > tmax.

    Returns an array w/unique identifiers for each segment.
    """
    n = 0
    trk = np.zeros(time.shape)
    for k in range(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax:
            n += 1
        trk[k] = n
    return trk


ifiles = sys.argv[1:]

# Sort input files on keyword number if provided
if key:
    import re
    print('sorting input files ...')
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    ifiles.sort(key=natkey)


# Add offset to ensure continuity of numbers between files
offset = 0

for ifile in ifiles:

    print(('file:', ifile))

    with h5py.File(ifile, 'a') as f:

        time = f['t_year'][:]
        orbit = segment_number(time, tmax=1e-5) + offset + start
        f['orbit'][:] = orbit

        # Update offset
        offset = orbit.max() + 1

        print((orbit[[0,-1]]))

print('done.')

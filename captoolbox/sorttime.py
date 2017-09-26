"""
Sort (in place) all 1d variables from an HDF5.

"""
import sys
import h5py
import numpy as np

files = sys.argv[1:]

for fname in files:

    print 'sorting file:', fname, '...'

    with h5py.File(fname, 'a') as f:

        # Sort time 
        i_sort = f['t_sec'][:].argsort()

        for var in f.values():
            var[:] = var[:][i_sort]

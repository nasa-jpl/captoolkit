#!/usr/bin/env python
"""
Sort (in place) all 1d variables in HDF5 file(s).

Args:
    varname: name of variable to use for sorting.
    file: file(s) to sort. 

Example:
    sorttime.py varname file1 file2 ...

"""
import sys
import h5py
import numpy as np


def sort_file(fname, varname):
    """ 
    Sort all 1d variables in an HDF5.

    Args:
        fname: HDF5 file with equal lenght 1d variables. 
        varname: name of variable to use for sorting.
    """
    with h5py.File(fname, 'a') as f:

        # Sort time 
        i_sort = f[varname][:].argsort()

        for var in f.values():
            var[:] = var[:][i_sort]


if __name__ == "__main__":

    varname = sys.argv[1]
    files = sys.argv[2:]

    for fname in files:
        print 'sorting file:', fname, '...'
        sort_file(fname, varname)


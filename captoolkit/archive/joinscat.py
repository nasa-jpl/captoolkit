#!/usr/bin/env python
"""
Join a set of geographical tiles (individual files).

It reads and writes in chunks, so it doesn't load all the data into memory.

It uses bbox and proj information from the file name for removing the
overlap between tiles if this information is present, otherwise just
merges all the data points.

Notes:
    * The HDF5 files must contain equal-leght 1d arrays only.
    * For a more general joining (heterogeneous file) see 'join2.py'.

"""
import os
import re
import sys
import h5py
import pyproj
import argparse
import tables as tb
import numpy as np


def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser(
            description='Join tiles (individual files).')

    parser.add_argument(
            'files', metavar='files', type=str, nargs='+',
            help='files to join into a single HDF5 file')

    parser.add_argument(
            '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
            help=('output file for joined tiles'),
            default=['joined_tiles.h5'],)

    parser.add_argument(
            '-v', metavar=('id','r2'), dest='vnames', type=str, nargs=2,
            help=('name of unique-id and r-squared vars in the HDF5 files'),
            default=['t_sec', 'r2'],)

    parser.add_argument(
            '-k', metavar=('keyword'), dest='key', type=str, nargs=1,
            help=('keyword in file name for sorting by tile number (<tile>_N.h5)'),
            default=[None],)

    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def overlap(a, b):
    """
    Find *both* indices of the intersection of two numpy arrays.

    a and b can be of different length.
    
    Taken from:
        https://www.followthesheep.com/?p=1366 
    """
    a1 = np.argsort(a)
    b1 = np.argsort(b)

    sort_left_a = a[a1].searchsorted(b[b1], side='left')
    sort_right_a = a[a1].searchsorted(b[b1], side='right')
    
    sort_left_b = b[b1].searchsorted(a[a1], side='left')
    sort_right_b = b[b1].searchsorted(a[a1], side='right')

    # # To find the difference:
    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in b but not in a?
    # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # To find the intersection:
    # which values of b are also in a?
    inds_b = (sort_right_a - sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a = (sort_right_b - sort_left_b > 0).nonzero()[0]

    return a1[inds_a], b1[inds_b]


# Pass arguments 
args = get_args()
ifiles = args.files     # input files
ofile = args.ofile[0]   # output file
idvar = args.vnames[0]  # unique-id variable name
r2var = args.vnames[1]  # r-squared variable name
key = args.key[0]       # keyword for sorting 

print_args(args)

assert len(ifiles) > 1

# Sort input files on keyword number if provided
if key:
    print 'sorting input files ...'
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    ifiles.sort(key=natkey)

print 'joining %d tiles ...' % len(ifiles)

with h5py.File(ofile, 'w') as fo:

    # Create resizable output file using info from first input file
    firstfile = ifiles.pop(0)

    with h5py.File(firstfile) as fi:

        # Create resizable arrays for all variables in the input file
        # The arrays can be of any shape, the first dim will be resizeable
        for key, val in fi.items():
            maxshape = (None,) + fi[key][:].shape[1:]
            fo.create_dataset(key, data=val[:], maxshape=maxshape)

    print firstfile

    # Iterate over the remaining input files
    for ifile in ifiles:

        print ifile
    
        fi = h5py.File(ifile)
    
        id_out, id_in = fo[idvar][:], fi[idvar][:]
        r2_out, r2_in = fo[r2var][:], fi[r2var][:]

        # Find the common points

        i_overlap_out, i_overlap_in = overlap(id_out, id_in)  # indices w.r.t. full arrs

        print len(i_overlap_out), len(i_overlap_in)
        sys.exit()
        r2_overlap_out, r2_overlap_in = r2_out[i_overlap_out], r2_in[i_overlap_in]

        # Find which ones need update, and update overlap set
        i_update, = np.where(r2_overlap_out < r2_overlap_in)
        r2_overlap_ou[i_update] = r2_overlap_in[i_update]

        # Update output container
        r2_out[i_overlap_out] = r2_overlap_out

        # Remove overlap values from input arr
        r2_in[i_overlap] = np.nan

        # Get indices of remaining valid points
        i_valid, = np.where(~np.isnan(r2_in))

        # Loop through each variable and append input chunk to output container
        for key, val in fi.items():

            # Get lengths of input chunk and output container
            length_next = fi[key][:][i_valid].shape[0]
            length = fo[key].shape[0]
    
            # Resize the dataset to accommodate next chunk
            fo[key].resize(length + length_next, axis=0)
    
            # Write next chunk
            fo[key][length:] = val[:][i_valid]
            fo.flush()

            # Update length of new output container
            length += length_next
    
        fi.close()
    
print 'output ->', ofile

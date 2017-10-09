#!/usr/bin/env python
"""
Merges several HDF5 files into a single or multiple larger file(s).

Example
    python merge.py ifile*.h5 -o outfile.h5

    python merge.py ifile*.h5 -o outfile.h5 -m 3

Notes
    * It merges files in the order they are read.
    * The parallel option only works with the -m option!
    * See complementary program: split.py

"""
import os
import sys
import h5py
import argparse
import numpy as np
from glob import glob


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Merges several HDF5 files.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='HDF5 files to merge')

parser.add_argument(
        '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
        help=('output file name'),
        default=[None], required=True,)

parser.add_argument(
        '-m', metavar='nfiles', dest='nfiles', type=int, nargs=1,
        help=('number of merged files (blocks)'),
        default=[None], required=False,)

parser.add_argument(
        '-v', metavar='var', dest='vnames', type=str, nargs='+',
        help=('only merge specific vars if given, otherwise merge all'),
        default=[],)

parser.add_argument(
        '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
        help=('number of jobs for parallel processing when using -m'),
        default=[1],)

# Global variables
args = parser.parse_args()
ifile = args.files[:]
ofile = args.ofile[0]
nfiles = args.nfiles[0]
vnames = args.vnames[:]
njobs = args.njobs[0]

# In case a string is passed to avoid "Argument list too long"
if len(ifile) == 1:
    ifile = glob(ifile[0])

print 'number of files to merge:', len(ifile)

if nfiles > 1:

    # List of groups of input files
    ifile = list(np.array_split(ifile, nfiles))

    # List of output file names
    fname = os.path.splitext(ofile)[0] + '_%02d.h5'
    ofile = [(fname % k) for k in xrange(len(ifile))]

else:

    ifile = [ifile]  # list of lists
    ofile = [ofile]  # list of strs


def get_total_len(ifiles):
    """ Get total output length from all input files. """
    N = 0
    for fn in ifiles:
        with h5py.File(fn) as f:
            N += f.values()[0].shape[0]
    return N


def get_var_names(ifile):
    with h5py.File(ifile, 'r') as f:
        vnames = f.keys()
    return vnames


def main(ifiles, ofile):

    # Get length of output containers (from all input files)
    N = get_total_len(ifiles)

    # Get var names from first file, if not provided
    variables = vnames if vnames else get_var_names(ifiles[0])

    with h5py.File(ofile, 'w') as f:

        # Create empty output containers (w/compression optimized for speed)
        [f.create_dataset(key, (N,), dtype='float64', compression='lzf') \
                for key in variables]

        # Iterate over the input files
        k1 = 0
        for ifile in ifiles:
            print ifile
    
            # Write next chunk (the input file)
            with h5py.File(ifile) as f2:
                k2 = k1 + f2.values()[0].shape[0]  # first var/first dim

                # Iterate over all variables
                for key in variables:
                    f[key][k1:k2] = f2[key][:]

            k1 = k2
    
    print 'merged', len(ifiles), 'files'
    print 'output ->', ofile


if njobs == 1:
    print 'Running sequential code ...'
    [main(fi, fo) for fi,fo in zip(ifile, ofile)]

else:
    print 'Running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(fi, fo) for fi,fo in zip(ifile, ofile))


print 'merged files:', len(ifile)

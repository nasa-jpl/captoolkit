#!/usr/bin/env python
"""
Merges several HDF5 files into a single or multiple larger file(s).

Example
    python merge.py infiles*.h5 -o outfile.h5

    python merge.py infiles*.h5 -o outfile.h5 -m 3

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
        '-o', metavar='outfile', dest='outfile', type=str, nargs=1,
        help=('output file name'),
        default=[None], required=True,)

parser.add_argument(
        '-m', metavar='nfiles', dest='nfiles', type=int, nargs=1,
        help=('number of merged files (blocks)'),
        default=[None], required=False,)

parser.add_argument(
        '-v', metavar='var', dest='vnames', type=str, nargs='+',
        help=('merge specific vars f given, otherwise merge all'),
        default=[],)

parser.add_argument(
        '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
        help=('number of jobs for parallel processing: only for -m!'),
        default=[1],)

# Global variables
args = parser.parse_args()
infiles = args.files[:]
outfile = args.outfile[0]
nfiles = args.nfiles[0]
vnames = args.vnames[:]
njobs = args.njobs[0]

# In case a string is passed to avoid "Argument list too long"
if len(infiles) == 1:
    infiles = glob(infiles[0])

print 'number of files to merge:', len(infiles)

if nfiles > 1:

    # List of groups of input files
    infiles = np.array_split(infiles, nfiles)

    # List of output file names
    fname = os.path.splitext(outfile)[0] + '_%02d.h5'
    outfiles = [(fname % k) for k in xrange(len(infiles))]

else:

    infiles = [infiles]  # list of lists
    outfiles = [outfile]  # list of strs


def main(ifiles, ofile):

    print 'merging files:', len(ifiles), '...'
    
    ifiles = list(ifiles)
    
    with h5py.File(ofile, 'w') as f:
    
        # Create resizable output file using info from first input file
        firstfile = ifiles.pop(0)
        print firstfile
        
        # Merge specific vars if given, otherwise merge all
        with h5py.File(firstfile) as f2:
            maxshape = (None,) + f2.values()[0].shape[1:]
            variables = vnames if vnames else f2.keys()
            [f.create_dataset(key, data=f2[key], maxshape=maxshape) \
                    for key in variables]
    
        # Iterate over the remaining input files
        for infile in ifiles:
    
            print infile
    
            f2 = h5py.File(infile)
            length_next = f2.values()[0].shape[0]  # first var
            length = f.values()[0].shape[0]
    
            for key in variables:
                # Resize the datasets to accommodate next chunk
                f[key].resize(length + length_next, axis=0)
    
                # Write next chunk
                f[key][length:] = f2[key]

            f.flush()
            f2.close()
            length += length_next
    
    print 'output ->', ofile


if njobs == 1:
    print 'Running sequential code ...'
    [main(fi, fo) for fi,fo in zip(infiles, outfiles)]

else:
    print 'Running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(fi, fo) for fi,fo in zip(infiles, outfiles))


print 'merged files:', len(infiles)

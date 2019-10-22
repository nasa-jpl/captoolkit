"""
Splits large 1d HDF5 file(s) into smaller ones.

Example:

    python split.py file.h5 -k 16

To see available options:

    python split.py -h

Also see complementary program: 'merge.py'

"""
import os
import sys
import h5py
import argparse
import numpy as np


# Default number of smaller files to split to
KFILES = 2

# Default njobs for sequential/parallel run
NJOBS = 1


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Splits large HDF5 file(s) into smaller ones.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='HDF5 file(s) to split')

parser.add_argument(
        '-k', metavar='nfiles', dest='nfiles', type=int, nargs=1,
        default=[KFILES],
        help=('number of smaller files to split to (-n 2)'))

parser.add_argument(
        '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
        default=[NJOBS],
        help=('number of jobs for parallel processing (-n 1)'))

# Global variables
args = parser.parse_args()
files = args.files
nfiles = args.nfiles[0]
njobs = args.njobs[0]


def partition(length, parts):
    """Partitions 'length' into (approximately) equal 'parts'."""
    sublengths = [length/parts] * parts
    for i in range(length % parts):  # treatment of remainder
        sublengths[i] += 1
    return sublengths


def main(infile):

    print(('input <- ', infile))

    with h5py.File(infile) as f:

        # Determine the total legth of input file
        total_legth = list(f.values())[0].shape[0]

        # Determine the length of output files
        lengths = partition(total_legth, nfiles)

        # Determine the names of output files
        fname = os.path.splitext(infile)[0] + '_%03d.h5'
        outfiles = [(fname % k) for k in range(len(lengths))]

        i1, i2 = 0, 0
        for outfile, length in zip(outfiles, lengths):

            i2 += length

            # Save chunks of each variable from f -> f2 
            with h5py.File(outfile, 'w') as f2:
                for key in list(f.keys()):
                    f2[key] = f[key][i1:i2]

            i1 = i2
            print(('output ->', outfile))


if njobs == 1:
    # Sequential code
    print('Running sequential code ...')
    [main(f) for f in files]

else:
    # Parallel code
    print(('Running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

#!/opt/anaconda3/bin/python
"""
Converts HDF5 (1d-arrays) to ASCII (columns).

"""
import sys
import os
import h5py
import argparse
import numpy as np


# Default njobs for sequential/parallel run
NJOBS = 1


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Convert HDF5 files to ASCII.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='ASCII file(s) to convert')

parser.add_argument(
        '-v', '--vars', metavar='', type=str, nargs='+',
        default=None,
        help=('name of variables in ASCII file (-v lon lat time...)'))

parser.add_argument(
        '-n', '--njobs', metavar='', type=int, nargs=1,
        default=[NJOBS],
        help=('number of jobs for parallel processing (-n 1)'))

# Global variables
args = parser.parse_args()
files = args.files
vars = args.vars
njobs = args.njobs[0]

print(('input files:', files))
print(('variables:', vars))
print(('njobs:', njobs))


def main(infile):
    print('converting file...')
    outfile = os.path.splitext(infile)[0] + '.txt'

    # Read full HDF5 data in memory  
    with h5py.File(infile) as f:
        variables = vars if vars else list(f.keys())
        cols = [f[v][:] for v in variables]

    # Write full ASCII data on disk
    np.savetxt(outfile, np.column_stack(cols), fmt='%.6f')

    print(('input <-', infile))
    print(('output ->', outfile))
    print(('ascii columns:', variables))


if njobs == 1:
    # Sequential code
    print('Running sequential code...')
    [main(f) for f in files]
else:
    # Parallel code
    print(('Running parallel code (%d jobs)...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

"""
Check for NaNs in a given variable and remove the respective "rows".

"""
import os
import h5py
import argparse
import numpy as np


def rename_file(fname, suffix='_NONAN'):
    path, ext = os.path.splitext(fname)
    os.rename(fname, path + suffix + ext)


# Define command-line arguments
parser = argparse.ArgumentParser(description='Remove NaN values')

parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')

parser.add_argument(
        '-v', metavar=('h_cor'), dest='vname', type=str, nargs=1,
        help=('Variable to search for NaNs'),
        default=['h_cor'],)

parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[1],)

args = parser.parse_args()

print 'parameters:'
for p in vars(args).iteritems():
    print p

files  = args.files
vname  = args.vname[0]
njobs  = args.njobs[0]


def main(ifile):

    with h5py.File(ifile, 'a') as f:

        x = f[vname][:]
        i_valid = ~np.isnan(x)

        n_valid = np.sum(i_valid)

        if n_valid == len(x):
            print 'no NaNs to remove!'
            return

        for k,v in f.items():
            y = v[:]
            del f[k]
            f[k] = y[i_valid]

    percent = 100 * (len(x)-n_valid) / float(len(x))
    print 'removed %g rows out of %g (%.2f %%)' % \
            (len(x)-n_valid, len(x), percent)

    rename_file(ifile, suffix='_NONAN')


if njobs == 1:
    print 'running sequential code ...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)



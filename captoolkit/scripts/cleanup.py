"""
Check for empty and currupted files and remove them from current dir.

"""
import os
import h5py
import argparse
from glob import glob


def is_empty(ifile):
    """If file is corruted or empty, return True."""
    try:
        with h5py.File(ifile, 'r') as f:
            if bool(f.keys()) and f['lon'].size > 0:
                return False
            else:
                return True
    except:
        return True


# Define command-line arguments
parser = argparse.ArgumentParser(description='Remove empty/corrupted files')
parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')
parser.add_argument(
        '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[1],)
args = parser.parse_args()

print('parameters:')
for p in vars(args).iteritems(): print(p)

files  = args.files
njobs  = args.njobs[0]

if len(files) == 1: files = glob(files[0])


def main(ifile):
    if is_empty(ifile):
        path, fname = os.path.dirname(ifile), os.path.basename(ifile)
        empty_files = path + '/EMPTY_FILES'
        if not os.path.exists(empty_files): os.makedirs(empty_files)
        os.rename(ifile, empty_files+'/'+fname)


if njobs == 1:
    print('running sequential code ...')
    [main(f) for f in files]

else:
    print('running parallel code ({0:d} jobs) ...'.format(njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

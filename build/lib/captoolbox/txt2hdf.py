"""
Converts ASCII tables to HDF5 (1d arrays).

Reads and writes in chunks for very large files.

Example:

    python txt2hdf.py /path/to/file.txt -v lon lat time...

To convert several files in parallel (say, 8 jobs):

    python txt2hdf.py /path/to/files/*.txt -n 8 -v lon lat time...

To see available options:

    python txt2hdf.py -h

Note: For parallel processing you might need to install the 'joblib' module.
(It will be included by default in Anaconda soon)

"""
import os
import h5py
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import OrderedDict


# Default chunksize (number of lines) for I/O
CHUNKSIZE = 100000

# Default njobs for sequential/parallel run
NJOBS = 1


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Convert ASCII tables to HDF5.\n'
        'If no variable names are provided, save table as 2d array.')

parser.add_argument(
        'files', metavar=('files'), type=str, nargs='+',
        help='ASCII file(s) to convert (it can be a directory) PASS FILES FIRST!')

parser.add_argument(
        '-v', metavar=('vars'), dest='vnames',  type=str, nargs='+',
        help=('name of variables in ASCII file (-v lon lat time...)'),
        default=[None],)

parser.add_argument(
        '-e', metavar=('.txt'), dest='ext', type=str, nargs=1,
        help=('extension of ASCII files'),
        default=['.txt'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help=('number of jobs for parallel processing (-n 1)'),
        default=[NJOBS],)

parser.add_argument(
        '-c', metavar=('chunk'), dest='chunk', type=int, nargs=1,
        help=('chunksize (# of lines) for I/O (-c 100000)'),
        default=[CHUNKSIZE],)

# Global variables
args = parser.parse_args()
files = args.files
vnames = args.vnames
njobs = args.njobs[0]
chunksize = args.chunk[0]
ext = args.ext[0]
sep = None  # If None, tries to determine  #TODO: Try a more clever way!


def list_files(path, endswith='.txt'):
    """List files in dir recursively."""
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if f.endswith(endswith)]


def init_vec(f, names, data):
    """Initialize resizable 1d arrays to hold the outputs."""
    
    # Iterate over 'names' and columns of 'data'
    dset = [(name, f.create_dataset(name, data=d, maxshape=(None,))) \
            for name, d in zip(names, data.T)]  # -> list
    return OrderedDict(dset)


def init_mat(f, name, data):
    """Initialize resizable 2d array to hold the output."""
    
    # Iterate over 'names' and columns of 'data'
    mat = f.create_dataset(name, data=data, maxshape=(None,data.shape[1]))
    return {name: mat}


def save_as_vec(dset, nrow, data):
    """Save 'data' columns as chunks in 1d arrays."""

    # Iterate over columns
    for name, d in zip(dset.keys(), data.T):
    
        # Resize the datasets to accommodate next chunk of rows
        dset[name].resize(nrow + data.shape[0], axis=0)
    
        # Write next chunk
        dset[name][nrow:] = d
    
    return dset


def save_as_mat(dset, nrow, data):
    """Save 'data' as a chunk in a 2d array."""

    # Resize the datasets to accommodate next chunk of rows
    dset['data'].resize(nrow + data.shape[0], axis=0)
    
    # Write next chunk
    dset['data'][nrow:] = data
    
    return dset


# Create variable name if not given
names = 'data' if vnames[0] is None else vnames

# Generate file list if directory given
if len(files) == 1 and os.path.isdir(files[0]):
    files = list_files(files[0], endswith=ext)

# In case a string is passed to avoid "Argument list too long"
elif len(files) == 1:
    files = glob(files[0])

else:
    pass


def main(infile):

    print 'converting ASCII table to HDF5 ...'

    outfile = os.path.splitext(infile)[0] + '.h5'

    with h5py.File(outfile, 'w') as f:

        # Read the first chunk to get the column structure
        reader = pd.read_table(infile, sep=sep, header=None,
                               chunksize=chunksize, engine='python')

        chunk = reader.get_chunk(chunksize)
        nrows, ncols = chunk.shape

        if names == 'data':
            dset = init_mat(f, names, chunk.values)
        else:
            dset = init_vec(f, names, chunk.values)

        print 'lines saved:', nrows, '...'

        # Read chunks of ASCII file
        for chunk in reader:

            if names == 'data':
                dset = save_as_mat(dset, nrows, chunk.values)
            else:
                dset = save_as_vec(dset, nrows, chunk.values)

            nrows += chunk.shape[0]
            print 'lines saved:', nrows, '...'

    print 'input <- ', infile
    print 'output ->', outfile
    print 'variable names in HDF5:', names


if njobs == 1:
    print 'Running sequential code...'
    [main(f) for f in files]

else:
    print 'Running parallel code (%d jobs)...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

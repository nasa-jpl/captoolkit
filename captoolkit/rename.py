#!/usr/bin/env python
import os
import sys
import h5py
from glob import glob

files = sys.argv[1:]

if len(files) == 1: files = glob(files)  # pass str for 'list too long'


def list_files(path, endswith='.txt'):
    """ List files in dir 'path' recursively. """
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if f.endswith(endswith)]


def rename_file(fname, prefix='', suffix=''):
    """ Add prefix and/or suffix to file name. """
    path, fname_ = os.path.split(fname)
    bname, ext = os.path.splitext(fname_)
    os.rename(fname, os.path.join(path, prefix+bname+suffix+ext))


def replace_ext(fname, new_ext='.h5'):
    """ Replace the extension of file name. """
    os.rename(fname, os.path.splitext(fname)[0] + new_ext)


def replace_text(fname, old_text='mnt', new_text='u'):
    """ Replace all occurrences of a 'text' in an ascii file. """
    # Read in the file
    with open(fname, 'r') as f:
      fdata = f.read()

    # Replace the target string
    fdata = fdata.replace(old_text, new_text)

    # Write the file out again
    with open(fname, 'w') as f:
      f.write(fdata)


def rename_var(fname):
    """ Rename defined variables in HDF5 file. """
    with h5py.File(fname) as f:
        f['bs'] = f['bs_ice1']
        f['lew'] = f['lew_ice2']
        f['tes'] = f['tes_ice2']
        del f['bs_ice1']
        del f['lew_ice2']
        del f['tes_ice2']


def add_time(fname):
    """ Add time variable to HDF5 file. """
    with h5py.File(fname) as f:
        time = f['t_sec'][:] / (365.25 * 24 * 3600.)
        f['time'] = time


print 'renaming %g files ...' % len(files)

if 1:
    for fname in files:
        rename_file(fname, suffix='_NONAN')
        #replace_ext(fname, new_ext='.h5')
        #rename_var(fname)
        #add_time(fname)
else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=16, verbose=5)(
        #delayed(rename_file)(fname) for fname in files)
        delayed(rename_var)(fname) for fname in files)
        #delayed(add_time)(fname) for fname in files)

print 'done.'

#!/usr/bin/env python
import os
import sys
import h5py
from glob import glob

files = sys.argv[1:]
#files = glob(sys.argv[1])


def rename_file(fname):
    suffix = '_floating'
    path, ext = os.path.splitext(fname)
    newfname = path + suffix + ext
    os.rename(fname, newfname)
    print fname, '->', newfname


def rename_var(fname):
    with h5py.File(fname) as f:
        f['bs'] = f['bs_ice1']
        f['lew'] = f['lew_ice2']
        f['tes'] = f['tes_ice2']
        del f['bs_ice1']
        del f['lew_ice2']
        del f['tes_ice2']


def add_time(fname):
    with h5py.File(fname) as f:
        time = f['t_sec'][:] / (365.25 * 24 * 3600.)
        f['time'] = time


if 1:
    for fname in files:
        #rename_file(fname)
        rename_var(fname)
        #add_time(fname)
else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=16, verbose=5)(
        #delayed(rename_file)(fname) for fname in files)
        delayed(rename_var)(fname) for fname in files)
        #delayed(add_time)(fname) for fname in files)

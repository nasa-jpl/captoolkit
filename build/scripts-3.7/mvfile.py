#!/opt/anaconda3/bin/python
"""
Rename files: add prefix and/or suffix and/or replace ext.

It reads files with a pattern from a dir (recursively), optional.

"""
import os
import sys
import h5py
import argparse
from glob import glob


def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description=('Rename files'))
    parser.add_argument(
            'file', type=str, nargs='+',
            help='files to rename',
            default=[None],)
    parser.add_argument(
            '-p', metavar='prefix', dest='prefix', type=str, nargs=1,
            help=('text to add at the beguining of file name'),
            default=[''],)
    parser.add_argument(
            '-s', metavar='suffix', dest='suffix', type=str, nargs=1,
            help=('text to add at the end of file name'),
            default=[''],)
    parser.add_argument(
            '-e', metavar='ext', dest='ext', type=str, nargs=1,
            help=('new extension to replace in file name'),
            default=[''],)
    parser.add_argument(
            '-r', metavar='pattern', dest='pattern', type=str, nargs=1,
            help=('rename files (with pattern) in dir recursively'),
            default=[''],)
    parser.add_argument(
            '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
            help=('number of jobs for parallel processing'),
            default=[1],)
    return parser.parse_args()


def list_files(path, pattern='.txt'):
    """ List files (with pattern) in dir recursively. """
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if pattern in f]


def rename_file(fname, prefix='', suffix='', ext=''):
    """ Add prefix and/or suffix to file name. """
    path, fname_ = os.path.split(fname)
    bname, ext_ = os.path.splitext(fname_)
    if ext: ext_ = ext
    os.rename(fname, os.path.join(path, prefix+bname+suffix+ext_))


if __name__ == '__main__':

    args = get_args()
    files = args.file[:]
    prefix = args.prefix[0]
    suffix = args.suffix[0]
    ext = args.ext[0]
    pattern = args.pattern[0]
    njobs = args.njobs[0]

    for arg in list(vars(args).items()): print(arg)  # print params

    if os.path.isdir(files[0]): files = list_files(files[0], pattern=pattern)

    if len(files) == 1: files = glob(files[0])  # pass str for 'list too long'

    if njobs == 1:
        print('running sequential code ...')
        [rename_file(f, prefix, suffix, ext) for f in files]
    else:
        print(('running parallel code (%d jobs) ...' % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(rename_file)(f, prefix, suffix, ext) for f in files)

    print(('renamed files:', len(files)))

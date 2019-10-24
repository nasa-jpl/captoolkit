#!/opt/anaconda3/bin/python
"""
Replace text in an ASCII file.

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
            description=('Replace text in ASCII files'))
    parser.add_argument(
            'file', type=str, nargs='+',
            help='files to read',
            default=[None],)
    parser.add_argument(
            '-a', metavar='oldtext', dest='oldtext', type=str, nargs=1,
            help=('text to be replaced'),
            default=[''],)
    parser.add_argument(
            '-b', metavar='newtext', dest='newtext', type=str, nargs=1,
            help=('text to be added'),
            default=[''],)
    parser.add_argument(
            '-r', metavar='pattern', dest='pattern', type=str, nargs=1,
            help=('read files (with pattern) in dir recursively'),
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


if __name__ == '__main__':

    args = get_args()
    files = args.file[:]
    oldtext = args.oldtext[0]
    newtext = args.newtext[0]
    pattern = args.pattern[0]
    njobs = args.njobs[0]

    for arg in list(vars(args).items()): print(arg)  # print params

    if os.path.isdir(files[0]): files = list_files(files[0], pattern=pattern)

    if len(files) == 1: files = glob(files[0])  # pass str for 'list too long'

    if njobs == 1:
        print('running sequential code ...')
        [replace_text(f, oldtext, newtext) for f in files]
    else:
        print(('running parallel code (%d jobs) ...' % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(replace_text)(f, oldtext, newtext) for f in files)

    print(('processed files:', len(files)))

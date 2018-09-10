#!/usr/bin/env python
"""
Merge tiles in the time dimension keeping the original tiling (grid).

Set of tiles can be located in different folders.

Merge set of tiles in parallel (optional).

Example:
    python mergetile.py /input/files1/*.h5 /input/files2/*.h5 \
            -o /output/file.h5 -v orbit lon lat t_year h_cor satid -n 4

"""
import re
import os
import sys
import h5py
import argparse
import numpy as np
from glob import glob


# Set keyword referent to 'tile_num' that is present in file name.
key = 'tile'


def get_tile_num(fname, key='tile'):
    """ Given 'key' extract 'num' from 'key_num' in string. """ 
    l = os.path.splitext(fname)[0].split('_')  # fname -> list
    i = l.index(key)                          
    return int(l[i+1])


def get_key_num(fname, key='tile', sep='_'):
    """ Given 'key' extract 'key_num' from string. """
    l = os.path.splitext(fname)[0].split(sep)  # fname -> list
    i = l.index(key)                  
    return key + sep + l[i+1]


def sort_by_key(files, key='tile'):
    """ Sort list by 'key_num' for given 'key'. """
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    return files.sort(key=natkey)  # sort inplace


def group_by_key(key_file_pairs):
    """ [(k1,f1), (k2,f2),..] -> {k1:[f1,..,fn], k2:[f2,..,fm],..}. """
    d = {}
    [d.setdefault(k, []).append(v) for k,v in key_file_pairs]
    return d


def add_suffix(fname, suffix=''):
    path, ext = os.path.splitext(fname)
    return path + suffix + ext


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Merge tiles in time keeping the original tiling.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='tiles to merge. If different dirs, ONE STRING PER FOLDER')

parser.add_argument(
        '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
        help=('output file name'),
        default=['merged.h5'],)

parser.add_argument(
        '-v', metavar='var', dest='vnames', type=str, nargs='+',
        help=('merge specific vars if given, otherwise merge all'),
        default=[],)

parser.add_argument(
        '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
        help=('number of jobs for parallel processing'),
        default=[1],)

# Global variables
args = parser.parse_args()
ifiles = args.files[:]
ofile = args.ofile[0]
vnames = args.vnames[:]
njobs = args.njobs[0]

# If single string with multiple paths
if len(ifiles) == 1:

    # List of substrings (paths)
    ifiles = ifiles[0].split()

    # Expand substrings into lists of file names
    file_sets = [glob(s) for s in ifiles]

    # Sort (in place) each file set by tile number (keep set order as given)
    [sort_by_key(file_set, key) for file_set in file_sets]

    # Generate flat list with file names     
    ifiles = [item for sublist in file_sets for item in sublist]

# Get key-file pais: list -> [(k1,f1), (k2,f2),..]
key_file_pairs = [(get_key_num(f, key), f) for f in ifiles]

# Group files for each tile: dict -> {k1:[f1..fn], k2:[f2..fm]}
tiles = group_by_key(key_file_pairs)

print 'number of tiles:', len(tiles)
print 'number of files:', len(ifiles)


def main(tile_num, files, ofile):
    """ Merge 'files' into 'ofile' with suffix 'tile_num'. """

    ofile = add_suffix(ofile, suffix='_'+tile_num) 

    print 'merging tile:', tile_num, '(%d files)' % len(files), '...'
    
    with h5py.File(ofile, 'w') as f:
    
        # Create resizable output file using info from first input file
        file0 = files.pop(0)
        print file0
        
        # Merge specific vars if given, otherwise merge all
        with h5py.File(file0, 'r') as f2:
            maxshape = (None,) + f2.values()[0].shape[1:]
            variables = vnames if vnames else f2.keys()
            [f.create_dataset(k, data=f2[k], maxshape=maxshape) \
                    for k in variables]
    
        # Iterate over the remaining input files
        for ifile in files:
    
            print ifile
    
            f2 = h5py.File(ifile)
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
    
    print 'out ->', ofile


if njobs == 1:
    print 'Running sequential code ...'
    [main(k,fs,ofile) for k,fs in tiles.items()]

else:
    print 'Running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(k,fs,ofile) for k,fs in tiles.items())

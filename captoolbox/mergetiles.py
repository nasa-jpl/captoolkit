#!/usr/bin/env python
"""
Merges tiles from different missions keeping the original tiling (grid).

Example:
    mergetiles.py "/input/files1/*.h5" "/input/files2/*.h5" \
            -o /output/file.h5 -v orbit lon lat t_year h_cor satid -n 4

Todo:
    Check that all passed sets of tiles contain the same tiles in them
    (i.e. remove tiles that are not repeated across all the sets)

"""
import os
import sys
import h5py
import argparse
import numpy as np
from glob import glob

# Set keyword present in the file name to sort files by
key = 'tile'

# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Merges tiles from different missions.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='tiles to merge => ONE STRING PER TILE SET')

parser.add_argument(
        '-o', metavar='outfile', dest='outfile', type=str, nargs=1,
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
infiles = args.files[:]
outfile = args.outfile[:]
vnames = args.vnames[:]
njobs = args.njobs[0]

assert len(infiles) > 1


def tile_num(fname):
    """ Extract tile number from file name. """
    l = os.path.splitext(fname)[0].split('_')  # fname -> list
    i = l.index('tile')
    return int(l[i+1])


def all_same(items):
    """ Determine if all items of a list are equal. """
    return all(x == items[0] for x in items)


# Expand strings into lists of file names
tile_sets = [glob(s) for s in infiles]

print 'number of tile sets to merge:', len(infiles)

# Sort input files by tile number 
if key:
    import re
    print 'sorting input files ...'
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    [ifiles.sort(key=natkey) for ifiles in tile_sets]  # sort inplace

# Group common tiles into a list 
tile_groups = [l for l in np.vstack(tile_sets).T]

# Check that all grouped tiles have the same number
for tiles in tile_groups:
    tile_nums = [tile_num(tile) for tile in tiles]
    assert all_same(tile_nums)

# Repeat output file name for each tile-merge in parallel
outfiles = np.repeat(outfile, len(tile_groups))


def main(ifiles, ofile):

    # Get the tile number
    tnum = tile_num(ifiles[0])

    # Add tile number to output file name
    path, ext = os.path.splitext(ofile)
    ofile = path + '_tile_' + str(tnum) + ext

    ifiles = list(ifiles)
    
    print 'merging tile:', tnum, '(%d files)' % len(ifiles), '...'
    
    with h5py.File(ofile, 'w') as f:
    
        # Create resizable output file using info from first input file
        firstfile = ifiles.pop(0)
        print firstfile
        
        # Merge specific vars if given, otherwise merge all
        with h5py.File(firstfile, 'r') as f2:
            maxshape = (None,) + f2.values()[0].shape[1:]
            variables = vnames if vnames else f2.keys()
            [f.create_dataset(key, data=f2[key], maxshape=maxshape) \
                    for key in variables]
    
        # Iterate over the remaining input files
        for infile in ifiles:
    
            print infile
    
            f2 = h5py.File(infile)
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
    
    print 'output ->', ofile


if njobs == 1:
    print 'Running sequential code ...'
    [main(fi, fo) for fi,fo in zip(tile_groups, outfiles)]

else:
    print 'Running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(fi, fo) for fi,fo in zip(tile_groups, outfiles))


print 'merged files:', len(infiles)

#!/usr/bin/env python
"""Join a set of geographical tiles (subgrids in individual files).

Reads and writes in chunks, so it doesn't load all the data into memory.

Uses bbox and proj information from the file name for removing the
overlap between tiles if this information is present, otherwise just
"concatenates" all the subgrids. 

Notes:
    - The HDF5 files must all have the same 2d shape with 1d x/y coords.
    - For joining data points into a single large file see 'join.py'.
    - If several mosaics (i.e. 1 grid per time step), joins each in parallel.
    - Groups input files with same 'time_key' (see below) for multiple mosaics.

    Bedmap boundaries: -b -3333000 3333000 -3333000 3333000
    Ross boundaries: -b -600000 410000 -1400000 -400000

Examples:
    python joingrd.py ~/data/cryosat2/floating/latest/*_D_* 
        -b -600000 410000 -1400000 -400000 -k tile -o joined_D.h5

"""
import os
import re
import sys
import h5py
import pyproj
import argparse
import itertools
import tables as tb
import numpy as np

time_key = 'time'


def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Join tiles (individual files).')
    parser.add_argument(
            'files', metavar='files', type=str, nargs='+',
            help='files to join into a single HDF5 file')
    parser.add_argument(
            '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
            help=('output file for joined tiles'),
            default=['joined_tiles.h5'],)
    parser.add_argument(
            '-v', metavar=('lon','lat'), dest='vnames', type=str, nargs=2,
            help=('name of lon/lat variables in the files'),
            default=['x', 'y'],)
    parser.add_argument(
            '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geographic region (deg or m)'),
            default=[], required=True)
    parser.add_argument(
            '-u', dest='flipy', action='store_true',
            help=('flip final grids upside-down (y-axis)'),
            default=False)
    parser.add_argument(
            '-z', metavar=None, dest='comp', type=str, nargs=1,
            help=('compress joined file(s)'),
            choices=('lzf', 'gzip'), default=[None],)
    parser.add_argument(
            '-k', metavar=('keyword'), dest='key', type=str, nargs=1,
            help=('keyword in file name for sorting by tile number '
                  '(e.g. "tile"_NNN.h5)'),
            default=[None],)
    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help='for parallel processing of multiple grids, optional',
            default=[1],)
    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def get_tile_bbox(fname, key='bbox'):
    """ Extract bbox info from file name. """
    fname = fname.split('_')  # fname -> list
    i = fname.index(key)
    return map(float, fname[i+1:i+5])  # m


def get_tile_proj(fname, key='epsg'):
    """ Extract EPSG number from file name. """
    fname = fname.split('_')  # fname -> list
    i = fname.index(key)
    return fname[i+1]


def get_key_num(fname, key='bin', sep='_'):
    """ Given 'key' extract 'key_num' from string. """
    l = os.path.splitext(fname)[0].split(sep)  # fname -> list
    i = l.index(key)                  
    return key + sep + l[i+1]


def get_tile_lenght(tile_bbox):
    xmin, xmax, ymin, ymax = tile_bbox
    return np.abs(xmax-xmin), np.abs(ymax-ymin)


def get_num_tiles(grid_bbox, tile_dx, tile_dy):
    """ How many tiles per row and col -> (ny,nx). """ 
    xmin, xmax, ymin, ymax = grid_bbox
    return (int(np.abs(ymax-ymin)/tile_dy), int(np.abs(xmax-xmin)/tile_dx))


def get_tile_shape(fname, vname):
    # vname is name of a 2d grid.
    with h5py.File(fname, 'r') as f: return f[vname].shape


def get_grid_shape(tile_shape, num_tiles):
    return tuple(np.array(tile_shape) * np.array(num_tiles))


def get_grid_names(fname):
    """ Return all 2d '/variable' names in the HDF5. """
    with h5py.File(fname, 'r') as f:
        vnames = [k for k in f.keys() if f[k].ndim == 2]
    return vnames


def create_output_grids(ofile, grid_shape, vnames):
    with h5py.File(ofile, 'w') as f:
        [f.create_dataset(v, grid_shape, 'f', fillvalue=np.nan) for v in vnames]


def save_output_coord(ofile, xy, vnames):
    with h5py.File(ofile, 'a') as f:
        f[vnames[0]] = xy[0]
        f[vnames[1]] = xy[1]


def get_grid_coord(bbox, grid_shape):
    xmin, xmax, ymin, ymax = bbox
    ny, nx = grid_shape
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)


def map_tile_to_grid(x_grid, y_grid, tile_bbox):
    xmin, xmax, ymin, ymax = tile_bbox
    i, = np.where( (y_grid >= ymin) & (y_grid <= ymax) )
    j, = np.where( (x_grid >= xmin) & (x_grid <= xmax) )
    return (i, j)


def get_tile_position(x_grid, y_grid, tile_bbox):
    i, j =  map_tile_to_grid(x_grid, y_grid, tile_bbox)
    return (i[0], i[-1]+1, j[0], j[-1]+1)


def group_by_key(files, key='bin'):
    """Group files by key_number."""
    files_sorted = sorted(files, key=lambda f: get_key_num(f, key=key))
    return [list(v) for k,v in itertools \
            .groupby(files_sorted, key=lambda f: get_key_num(f, key=key))]


def flipud(fname, vnames):
    with h5py.File(fname, 'a') as f:
        for v in vnames: f[v][:] = np.flipud(f[v][:])
        print 'final grids flipped upside-down'


def join(ifiles, suffix=''):
    """Join a set of subgrids into a mosaic."""
    assert len(ifiles) > 1

    # Sort input files on keyword number if provided
    if key:
        print 'sorting input files ...'
        natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
        ifiles.sort(key=natkey)

    if suffix: suffix = '_' + get_key_num(ifiles[0], key=suffix)
    ofile_ = ofile + suffix

    # Generate empty output grids w/coords on disk
    tile_bbox = get_tile_bbox(ifiles[0])
    dx, dy = get_tile_lenght(tile_bbox)
    num_tiles = get_num_tiles(grid_bbox, dx, dy)
    vnames = get_grid_names(ifiles[0])
    tile_shape = get_tile_shape(ifiles[0], vnames[0])
    grid_shape = get_grid_shape(tile_shape, num_tiles)
    x_grid, y_grid = get_grid_coord(grid_bbox, grid_shape)
    create_output_grids(ofile_, grid_shape, vnames)
    save_output_coord(ofile_, (x_grid, y_grid), (xvar, yvar))

    # Iterate over tiles
    for ifile in ifiles:
        print 'tile:', ifile
        
        tile_bbox = get_tile_bbox(ifile)
        i1,i2,j1,j2 = get_tile_position(x_grid, y_grid, tile_bbox)

        with h5py.File(ifile, 'r') as fi, h5py.File(ofile_, 'a') as fo:
            for v in vnames: fo[v][i1:i2,j1:j2] = fi[v][:]

    if flipy: flipud(ofile_, vnames) 

    print 'joined tiles:', len(ifiles)
    print 'out ->', ofile_



# Pass arguments 
args = get_args()
ifiles = args.files       # input files
ofile = args.ofile[0]     # output file
grid_bbox = args.bbox[:]  # bounding box EPSG (m) or geographical (deg)
xvar = args.vnames[0]     # lon variable names
yvar = args.vnames[1]     # lat variable names
flipy = args.flipy        # flip final grid upside-down
key = args.key[0]         # keyword for sorting 
comp = args.comp[0]
njobs = args.njobs[0]

print_args(args)

try:
    allfiles = group_by_key(ifiles, key=time_key)
except:
    allfiles = [ifiles]

multiple_grids = len(allfiles) > 1

if njobs == 1 or not multiple_grids:
    print 'Running sequential code ...'
    [join(ifiles) for ifiles in allfiles]

else:
    print 'Running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(join)(ifiles, time_key) for ifiles in allfiles)

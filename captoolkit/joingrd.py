#!/usr/bin/env python
"""
Join a set of geographical tiles (subgrids in individual files).

It reads and writes in chunks, so it doesn't load all the data into memory.

It uses bbox and proj information from the file name for removing the
overlap between tiles if this information is present, otherwise just
"concatenates" all the subgrids. 

Notes:
    The HDF5 files must contain equal-leght 2d arrays with 1d x/y coords.
    For joining data points into a single large file see 'join.py'.

    Bedmap boundaries: -b -3333000 3333000 -3333000 3333000
    Ross boundaries: -b -600000 400000 -1400000 -1000    => old
    Ross boundaries: -b -610000 410000 -1400000 -400000  => new

Examples:
    python joingrd.py ~/data/cryosat2/floating/latest/*_D_* -b -600000 400000 -1400000 -1000 -d 100 -k tile -o joined_D.h5

"""
##NOTE: For now it assumes the tile has same lenght in x and y (i.e. a square)

import os
import re
import sys
import h5py
import pyproj
import argparse
import tables as tb
import numpy as np


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
            '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
            help=('block size of tile (km)'),
            default=[], required=True)

    parser.add_argument(
            '-z', metavar=None, dest='comp', type=str, nargs=1,
            help=('compress joined file(s)'),
            choices=('lzf', 'gzip'), default=[None],)

    parser.add_argument(
            '-k', metavar=('keyword'), dest='key', type=str, nargs=1,
            help=('keyword in file name for sorting by tile number (<tile>_N.h5)'),
            default=[None],)

    return parser.parse_args()


def print_args(args):
    print('Input arguments:')
    for arg in list(vars(args).items()):
        print(arg)


def get_bbox(fname):
    """ Extract bbox info from file name. """
    fname = fname.split('_')  # fname -> list
    i = fname.index('bbox')
    return list(map(float, fname[i+1:i+5]))  # m


def get_proj(fname):
    """ Extract EPSG number from file name. """
    fname = fname.split('_')  # fname -> list
    i = fname.index('epsg')
    return fname[i+1]


def get_num_tiles(grid_bbox, dxy):
    """ How many tiles per row and col -> (ny,nx). """ 
    xmin, xmax, ymin, ymax = grid_bbox
    return (int(np.abs(ymax-ymin)/dxy), int(np.abs(xmax-xmin)/dxy))


def get_tile_shape(fname, vname):
    # vname is name of a 2d grid.
    with h5py.File(fname, 'r') as f:
        return f[vname].shape


def get_grid_shape(tile_shape, num_tiles):
    return tuple(np.array(tile_shape) * np.array(num_tiles))


def get_grid_names(fname):
    """ Return all 2d '/variable' names in the HDF5. """
    with h5py.File(fname, 'r') as f:
        vnames = [k for k in list(f.keys()) if f[k].ndim == 2]
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


def where_tile(x_grid, y_grid, tile_bbox):
    xmin, xmax, ymin, ymax = tile_bbox
    i, = np.where( (y_grid >= ymin) & (y_grid <= ymax) )
    j, = np.where( (x_grid >= xmin) & (x_grid <= xmax) )
    return (i, j)


def map_tile_to_grid(x_grid, y_grid, tile_bbox):
    i,j =  where_tile(x_grid, y_grid, tile_bbox)
    return (i[0], i[-1]+1, j[0], j[-1]+1)


# Pass arguments 
args = get_args()
ifiles = args.files       # input files
ofile = args.ofile[0]     # output file
grid_bbox = args.bbox[:]  # bounding box EPSG (m) or geographical (deg)
dxy = args.dxy[0] * 1e3   # tile length (km -> m)
xvar = args.vnames[0]     # lon variable names
yvar = args.vnames[1]     # lat variable names
comp = args.comp[0]
key = args.key[0]         # keyword for sorting 

print_args(args)

assert len(ifiles) > 1

# Sort input files on keyword number if provided
if key:
    print('sorting input files ...')
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    ifiles.sort(key=natkey)

# Generate empty output grids w/coords on disk
vnames = get_grid_names(ifiles[0])
num_tiles = get_num_tiles(grid_bbox, dxy)
tile_shape = get_tile_shape(ifiles[0], vnames[0])
grid_shape = get_grid_shape(tile_shape, num_tiles)
x_grid, y_grid = get_grid_coord(grid_bbox, grid_shape)
create_output_grids(ofile, grid_shape, vnames)
save_output_coord(ofile, (x_grid, y_grid), ('x', 'y'))

# Iterate over tiles
for ifile in ifiles:

    print(('tile:', ifile))
    
    tile_bbox = get_bbox(ifile)
    i1,i2,j1,j2 = map_tile_to_grid(x_grid, y_grid, tile_bbox)

    with h5py.File(ifile, 'r') as fi, h5py.File(ofile, 'a') as fo:
        for v in vnames:
            fo[v][i1:i2,j1:j2] = fi[v][:]

print(('joined tiles:', len(ifiles)))
print(('out ->', ofile))

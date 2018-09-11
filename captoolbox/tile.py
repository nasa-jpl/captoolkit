#!/usr/bin/env python
"""
Program for tiling geographical data to reduce data volumes and allow parallization. Takes
input file which is given the columns for longitude and latitude, overlap (km), block size
of tiles (km) and the projection used to make the tiles (EPSG-format).

The code can deal with very-large files by reading/searching/saving in chunks.

Large files may slowdown the processing, but it's guarantee to complete the tilling.

Notes:
    bedmap boundaries: -b -3333000 3333000 -3333000 3333000

Change Log:

    - added imports
    - added HDF5 I/O
    - added parallelization
    - added in- and out-of-memory processing
    - added command-line args
    - added much more...

"""
import os
import sys
import h5py 
import pyproj
import argparse
import tables as tb
import pandas as pd
import numpy as np
from glob import glob


# Optimal chunk size
chunks = 100000


def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser(
            description='Split geographical data into (overlapping) tiles')

    parser.add_argument(
            'file', metavar='file', type=str, nargs='+',
            help='single or multiple file(s) to split in tiles (HDF5)')
            
    parser.add_argument(
            '-b', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geograph. region (deg or m), optional'),
            default=False,)
            
    parser.add_argument(
            '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
            help=('block size of tile (km)'),
            default=[], required=True)

    parser.add_argument(
            '-r', metavar=('buffer'), dest='dr', type=float, nargs=1,
            help=("overlap between tiles (km)"),
            default=[0],)

    parser.add_argument(
            '-v', metavar=('lon','lat'), dest='vnames', type=str, nargs=2,
            help=('name of x/y variables'),
            default=[None, None],)

    parser.add_argument(
            '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help="for parallel writing of multiple tiles, optional",
            default=[1],)

    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    return pyproj.transform(proj1, proj2, x, y)


def get_xy(ifile, vnames=['lon', 'lat'], proj='3031'):
    """ Get lon/lat from input file and convert to x/y. """
    xvar, yvar = vnames
    with tb.open_file(ifile) as fi:
        x = fi.get_node('/', xvar)[:]
        y = fi.get_node('/', yvar)[:]
    return transform_coord('4326', proj, x, y)


def add_suffix(fname, suffix=''):
    path, ext = os.path.splitext(fname)
    return path + suffix + ext


def get_tile_bboxs(grid_bbox, dxy):
    """ Define bbox of tiles given bbox of grid and tile size. """
    
    xmin, xmax, ymin, ymax = grid_bbox

    # Number of tile edges on each dimension
    Nns = int(np.abs(ymax - ymin) / dxy) + 1
    New = int(np.abs(xmax - xmin) / dxy) + 1
    
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin, xmax, New)
    yg = np.linspace(ymin, ymax, Nns)
    
    # Vector of bbox for each tile   ##NOTE: Nested loop!
    bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg

    return bboxs


def get_tile_data(ifile, x, y, bbox, buff=1, proj='3031', tile_num=0):
    """ Extract data within bbox and save to individual file. """

    xmin, xmax, ymin, ymax = bbox

    # Open input file (out-of-core)
    fi = tb.open_file(ifile)

    # Get all 1d variables into a list (out-of-core)
    points = [fi.get_node('/', v.name) for v in fi.list_nodes('/')]

    npts = 0
    nrow = x.shape[0]
    first_iter = True

    # Read and write in chunks
    for i in range(0, nrow, chunks):

        k = min(i+chunks, nrow)

        x_chunk = x[i:k]
        y_chunk = y[i:k]

        # Get the tile indices
        idx, = np.where( (x_chunk >= xmin-buff) & (x_chunk <= xmax+buff) & 
                         (y_chunk >= ymin-buff) & (y_chunk <= ymax+buff) )
        
        # Leave chunk if outside tile
        if len(idx) == 0: continue

        # Get chunk of data in-memory, and
        # Query chunk in-memory
        points_chunk = [d[i:k] for d in points]  # -> list of 1d chunks
        points_chunk = [d[idx] for d in points_chunk]

        # Create output file on first iteration
        if first_iter:

            # Define file name
            suffix = ('_bbox_%d_%d_%d_%d_buff_%g_epsg_%s_tile_%03d' % \
                     (xmin, xmax, ymin, ymax, buff/1e3, proj, tile_num))

            ofile = add_suffix(ifile, suffix)

            fo = tb.open_file(ofile, 'w')

            # Initialize containers (lenght=0)
            out = [fo.create_earray('/', v.name, tb.Float64Atom(), shape=(0,))
                   for v in points]

            first_iter = False
        
        # Save chunk
        [v.append(d) for v, d in zip(out, points_chunk)]
        npts += points_chunk[0].shape[0]

        fo.flush()

    if npts != 0: print 'tile %03d: #points' % tile_num, npts, '...'

    try:
        fo.close()
    except:
        pass

    fi.close()


def count_files(ifiles, key='*tile*'):
    saved = []
    for f in ifiles: 
        path, ext = os.path.splitext(f)
        saved.extend(glob(path + key))
    return len(saved)


# Pass arguments 
args    = get_args()
ifiles  = args.file[:]      # input file(s)
vnames  = args.vnames[:]    # lon/lat variable names
bbox_   = args.bbox         # bounding box EPSG (m) or geographical (deg)
dr      = args.dr[0] * 1e3  # buffer (km -> m)
dxy     = args.dxy[0] * 1e3 # tile length (km -> m)
proj    = args.proj[0]      # EPSG proj number
njobs   = args.njobs[0]     # parallel writing

print_args(args)

# Generate list of items (only once!) for parallel proc
print 'generating list of tasks (files x tiles) ...'

bboxs = get_tile_bboxs(bbox_, dxy)                                       # [b1, b2..]
xys = [get_xy(f, vnames, proj) for f in ifiles]                          # [(x1,y1), (x2,y2)..]
fxys = [(f,x,y) for f,(x,y) in zip(ifiles, xys)]                         # [(f1,x1,y1), (f2,x2,y2)..]
fxybs = [(f,x,y,b,n+1) for (f,x,y) in fxys for n,b in enumerate(bboxs)]  # [(f1,x1,y1,b1,1), (f2,x2,y2,b2,2)..]

print 'number of files:', len(ifiles)
print 'number of tiles:', len(bboxs)
print 'number of tasks:', len(fxybs)

# For each bbox scan full file

if njobs == 1:
    print 'running sequential code ...'
    [get_tile_data(f, x, y, b, dr, proj, n) for f,x,y,b,n in fxybs]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(get_tile_data)(f, x, y, b, dr, proj, n) for f,x,y,b,n in fxybs)  

print 'number of tiles with data:', count_files(ifiles)

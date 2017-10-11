#!/usr/bin/env python
"""
Program for tiling geographical data to reduce data volumes and allow parallization. Takes
input file which is given the columns for longitude and latitude, overlap (km), block size
of tiles (km) and the projection used to make the tiles (EPSG-format).

Change Log:

    - added imports
    - added HDF5 I/O
    - added parallelization
    - added in- and out-of-memory processing
    - added command-line args

"""
import os
import sys
import pyproj
import argparse
import tables as tb
import pandas as pd
import numpy as np


# Defie command-line arguments
parser = argparse.ArgumentParser(
        description='Split geographical data into (overlapping) tiles')

parser.add_argument(
        'file', metavar='file', type=str, nargs=1,
        help='single file to split in tiles (ASCII, HDF5 or Numpy)')

parser.add_argument(
        '-d', metavar=('length'), dest='dxy', type=float, nargs=1,
        help=('block size of tile (km)'),
        default=[], required=True)

parser.add_argument(
        '-b', metavar=('buffer'), dest='dr', type=float, nargs=1,
        help=("overlap between tiles (km)"),
        default=[0],)

parser.add_argument(
        '-v', metavar=('lon','lat'), dest='vnames', type=str, nargs=2,
        help=('name of x/y variables'),
        default=[None, None],)

parser.add_argument(
        '-c', metavar=(1,2), dest='cols', type=int, nargs=2,
        help=("column of lon/lat in the file"),
        default=[None, None],)

parser.add_argument(
        '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
        help="for parallel writing of multiple tiles, optional",
        default=[1],)


args = parser.parse_args()

# Pass arguments 
ifile = args.file[0]       # input file
vnames = args.vnames       # lon/lat variable names
icols = args.cols          # lon/lat columns
dr    = args.dr[0] * 1e3   # buffer (km -> m)
dxy   = args.dxy[0] * 1e3  # tile length (km -> m)
proj  = args.proj[0]       # EPSG proj number
njobs = args.njobs[0]      # parallel writing

# Optimal chunk size
chunks = 1000

if vnames[0] is not None:
    # Get var names
    xvar, yvar = vnames
    variables = True
else:
    # Get columns
    cx, cy = icols
    variables = False

print 'input file:', ifile
print 'x/y var names:', vnames
print 'x/y col numbers:', icols
print 'tile length (km):', dxy * 1e-3
print 'tile overlap (km):', dr * 1e-3
print 'tile projection:', proj
print 'n jobs:', njobs


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


print 'loading and transforming coords ...'

# Load lon/lat in-memory
def get_bboxs(ifile):
    """ Define tiles (bbox). """

    print 'building bboxes ...'
    with tb.open_file(ifile) as fi:

        if variables:
            # Read 1d arrays
            x = fi.get_node('/', xvar)[:]
            y = fi.get_node('/', yvar)[:]
            nrow = x.shape[0]

        else:
            # Read 2d matrix
            data = fi.root.data
            nrow, ncol = data.shape

            # in-memory
            x = data[:, cx]
            y = data[:, cy]

        # Convert into sterographic coordinates
        x, y = transform_coord('4326', proj, x, y)

        # Number of tile edges on each dimension 
        Nns = int(np.abs(y.max() - y.min()) / dxy) + 1
        New = int(np.abs(x.max() - x.min()) / dxy) + 1

        # Coord of tile edges for each dimension
        xg = np.linspace(x.min(), x.max(), New)
        yg = np.linspace(y.min(), y.max(), Nns)

        # Vector of bbox for each tile
        bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                           for s,n in zip(yg[:-1], yg[1:])]
        del xg, yg

        print 'total partitions:', len(bboxs)
        return bboxs


def get_tile(bbox, n):

    # Bounding box
    xmin, xmax, ymin, ymax = bbox
    
    # Open input file (out-of-core)
    fi = tb.open_file(ifile)

    if variables:
        # Get all 1d variables into a list
        Points = [fi.get_node('/', v.name) for v in fi.list_nodes('/')]

    else:
        # Get 2d matrix
        Points = fi.root.data

    npts = 0
    first_iter = True

    # Read and write in chunks
    for i in range(0, nrow, chunks):

        k = min(i+chunks, nrow)

        x_chunk = x[i:k]
        y_chunk = y[i:k]

        # Get the tile indices
        idx, = np.where( (x_chunk >= xmin-dr) & (x_chunk <= xmax+dr) & 
                         (y_chunk >= ymin-dr) & (y_chunk <= ymax+dr) )
        
        # Leave chunk if outside tile
        if len(idx) == 0: continue

        # Get chunk of data in-memory, and
        # Query chunk in-memory
        if variables:
            Points_chunk = [d[i:k] for d in Points]  # -> list of 1d chunks
            Points_chunk = [d[idx] for d in Points_chunk]

        else:
            # 2d array
            Points_chunk = Points[i:k,:]  # -> 2d chunk
            Points_chunk = Points_chunk[idx,:]

        # Create output file in first iteration
        if first_iter:

            # Define file name
            sufix = ('_bbox_%d_%d_%d_%d_buff_%g_epsg_%s_tile_%d' % \
                     (xmin, xmax, ymin, ymax, dr/1e3, proj, n))

            path, ext = os.path.splitext(ifile)
            ofile = path + sufix + ext

            fo = tb.open_file(ofile, 'w')

            # Initialize containers (lenght=0)
            if variables:
                out = [fo.create_earray('/', v.name, tb.Float64Atom(), shape=(0,))
                       for v in Points]

            else:
                out = fo.create_earray('/', 'data', tb.Float64Atom(), shape=(0,ncol))

            first_iter = False
        
        # Save chunk
        if variables:
            [v.append(d) for v, d in zip(out, Points_chunk)]
            npts += Points_chunk[0].shape[0]

        else:
            out.append(Points_chunk)
            npts += Points_chunk.shape[0]

        fo.flush()


    print 'tile %d, points' % n, npts, '...'

    try:
        fo.close()
    except:
        pass

    fi.close()

    """
    # Output strings
    prog  = "prog.py "
    opath = ofile + " "
    spath = "path" + ofile[0:-4] + ".xyz" + " "
    bbox  = str(xmin) + " " + str(xmax) + " " + str(ymin) + " " + str(ymax)
    args  = " args"
    
    # Print out log file for processing
    print prog + opath + spath + bbox + args
    """


# Get bbox of all tiles
bboxs = get_bboxs(ifile)

if njobs == 1:
    print 'running sequential code ...'
    [get_tile(bbox, n+1) for n, bbox in enumerate(bboxs)]

else:
    print 'running parallel code (%d jobs) ...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(
            delayed(get_tile)(bbox, n+1) for n, bbox in enumerate(bboxs))

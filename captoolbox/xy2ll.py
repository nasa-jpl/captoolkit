#!/usr/bin/env python
"""
Convert polar stereographic coords (x/y) to geodetic coords (lon/lat).

"""
import os
import sys
import h5py
import pyproj
import argparse
import numpy as np


def get_args():
    """ Get command-line arguments. """

    parser = argparse.ArgumentParser(
            description='Convert x/y -> lon/lat.')

    parser.add_argument(
            'files', metavar='files', type=str, nargs='+',
            help='HDF5 files to convert')

    parser.add_argument(
            '-v', metavar=('x','y'), dest='vnames', type=str, nargs=2,
            help=('name of x/y variables in the files'),
            default=['x', 'y'],)

    parser.add_argument(
            '-n', metavar=('n_jobs'), dest='njobs', type=int, nargs=1,
            help="number of jobs for parallel processing",
            default=[1],)

    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def main(ifile, xvar, yvar):

    print 'converting:', ifile, '...'

    with h5py.File(ifile, 'a') as fi:

        x = fi[xvar][:]
        y = fi[yvar][:]

        if x.shape[0] < 1:
            print 'file is empty, skiiping.'
            pass
        elif np.nanmax(np.abs(x)) < 360 and (y[~np.isnan(y)] < 0).all():
            print 'coords are lon/lat, skiiping.'
            pass
        else:
            # From x/y -> lon/lat
            lon, lat = transform_coord(3031, 4326, x, y)

            fi[xvar][:] = lon
            fi[yvar][:] = lat

            fi.flush()


if __name__ == '__main__':

    # Pass arguments 
    args = get_args()
    ifiles = args.files    # input files
    xvar = args.vnames[0]  # lon variable names
    yvar = args.vnames[1]  # lat variable names
    njobs = args.njobs[0]  # parallel writing

    print_args(args)

    assert len(ifiles) > 1

    if njobs == 1:
        print 'running sequential code ...'
        [main(ifile, xvar, yvar) for ifile in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(
                delayed(main)(ifile, xvar, yvar) for ifile in ifiles)

    print 'done!'

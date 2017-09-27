#!/usr/bin/env python
"""
Join a set of geographical tiles (individual files).

It uses bbox and proj information from the file name for removing the
overlap between tiles if this information is present, otherwise just
merges all the data points.

"""
import os
import re
import sys
import h5py
import pyproj
import argparse
import tables as tb
import numpy as np


# Define command-line arguments
parser = argparse.ArgumentParser(
        description='Join tiles (individual files).')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='files to join into a single HDF5 file')

parser.add_argument(
        '-o', metavar=('outfile'), dest='ofile', type=str, nargs=1,
        help=('output file for joined tiles'),
        default=['joined_tiles.ht'],)

parser.add_argument(
        '-v', metavar=('lon','lat'), dest='vnames', type=str, nargs=2,
        help=('name of lon/lat variables in each file'),
        default=['lon', 'lat'],)

parser.add_argument(
        '-k', metavar=('keyword'), dest='key', type=str, nargs=1,
        help=('keyword preceding numbers for sorting files, optional'),
        default=[None],)


args = parser.parse_args()

# Pass arguments 
ifiles = args.files      # input files
ofile  = args.ofile[0]   # output file
xvar   = args.vnames[0]  # lon variable names
yvar   = args.vnames[1]  # lat variable names
key    = args.key[0]     # keyword for sorting 

print 'input files:', len(ifiles)
print 'output file:', ofile
print 'x/y var names:', xvar, yvar
print 'key for sorting files:', key


assert len(ifiles) > 1

# Sort input files on keyword number if provided
if key:
    print 'sorting input files ...'
    natkey = lambda s: int(re.findall(key+'_\d+', s)[0].split('_')[-1])
    ifiles.sort(key=natkey)


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """
    
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def get_bbox(fname):
    """ Extract bbox info from file name. """

    fname = fname.split('_')  # fname -> list
    i = fname.index('bbox')
    return map(float, fname[i+1:i+5])  # m


def get_proj(fname):
    """ Extract EPSG number from file name. """

    fname = fname.split('_')  # fname -> list
    i = fname.index('epsg')
    return fname[i+1]


print 'joining %d tiles ...' % len(ifiles)


with h5py.File(ofile, 'w') as fo:

    # Create resizable output file using info from first input file
    firstfile = ifiles.pop(0)

    with h5py.File(firstfile) as fi:

        if '_bbox' in firstfile:
            print 'merging only points in bbox ...'

            bbox = get_bbox(firstfile)
            proj = get_proj(firstfile)

            xmin, xmax, ymin, ymax = bbox
            lon, lat = fi[xvar][:], fi[yvar][:]

            x, y = transform_coord('4326', proj, lon, lat)

            idx, = np.where( (x >= xmin) & (x <= xmax) & 
                             (y >= ymin) & (y <= ymax) )
            
            if len(idx) == 0:
                print 'first file is empty!'
                sys.exit()

        else:
            print 'merging all points ...'
            idx = None

        maxshape = (None,) + fi.values()[0][:][idx].shape[1:]
        [fo.create_dataset(key, data=d[:][idx], maxshape=maxshape) \
                for key,d in fi.items()]

    print firstfile

    # Iterate over the remaining input files
    for ifile in ifiles:
    
        print ifile
    
        fi = h5py.File(ifile)
    
        if '_bbox' in ifile:
    
            bbox = get_bbox(ifile)
            proj = get_proj(ifile)
    
            xmin, xmax, ymin, ymax = bbox
            lon, lat = fi[xvar][:], fi[yvar][:]
    
            x, y = transform_coord('4326', proj, lon, lat)
    
            idx, = np.where( (x >= xmin) & (x <= xmax) & 
                             (y >= ymin) & (y <= ymax) )
            
            if len(idx) == 0:
                continue

        else:
            idx = None
    
        length_next = fi.values()[0][:][idx].shape[0]  # get info from first var
        length = fo.values()[0].shape[0]
    
        for key,val in fi.items():
            # Resize the datasets to accommodate next chunk
            fo[key].resize(length + length_next, axis=0)
    
            # Write next chunk
            fo[key][length:] = val[:][idx]
            fo.flush()
    
        fi.close()
        length += length_next
    
print 'output ->', ofile

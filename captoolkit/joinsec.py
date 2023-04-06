#!/usr/bin/python
"""
Join a set of geographical tiles (individual files) and is a simpler version
of "join.py" adapted for the output from "fitsec.py".

It uses bbox and proj information from the file name for removing the
overlap between tiles if this information is present, otherwise just
merges all the data points.

Note:
    - This program should only be used for outputs from fitsec.py

Example:
    python ~/joinsec.py ./*_SEC.h5 -o merged.h5

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
    Johan Nilsson (johan.nilsson@jpl.nasa.gov) [Modified from FS version]
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
    Jet Propulsion Laboratory, California Institute of Technology

"""
import warnings
warnings.filterwarnings("ignore")
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
        default=['joined_tiles.h5'],)

args = parser.parse_args()

# Pass arguments
ifiles = args.files      # input files
ofile  = args.ofile[0]   # output file

print(('input files:', len(ifiles)))
print(('output file:', ofile))

assert len(ifiles) > 1

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
    return list(map(float, fname[i+1:i+5]))  # m


def get_proj(fname):
    """ Extract EPSG number from file name. """

    fname = fname.split('_')  # fname -> list
    i = fname.index('epsg')
    return fname[i+1]


print(('joining %d tiles ...' % len(ifiles)))

# Create output file to save data
fo = h5py.File(ofile, 'w')

# Create resizable output file using info from first input file
firstfile = ifiles.pop(0)

# Get the variable names in file and time
with h5py.File(firstfile) as fi:
    keys = []
    for key in list(fi.keys()):
        keys.append(key)
        if key == "time":
            time = fi["time"][:]

# Convert to a list
keys = list(keys)

# Loop trough keys for each file
for key in keys:
    if key == "time": continue
    tmp = []
    # Add data to each file
    for f in ifiles:

        #print('merging only points in bbox ...')
        bbox = get_bbox(f)
        proj = get_proj(f)

        # Extract bbox
        xmin, xmax, ymin, ymax = bbox

        # Read variable and lat/lon coords
        with h5py.File(f) as fi:
            val = fi[key][:]

            if val.ndim == 1:
                if key == 'lon(t)' or key == 'lat(t)':
                    lon, lat = fi['lon(t)'][:], fi['lat(t)'][:]
                else:
                    lon, lat = fi['lon'][:], fi['lat'][:]
            else:
                lon, lat = fi['lon(t)'][:], fi['lat(t)'][:]

        # Check for empty file
        if len(val) == 0: continue

        # Transform to wanted coordinates
        x, y = transform_coord('4326', proj, lon, lat)

        # Get indec of data inside the tile (reject overlap)
        idx, = np.where((x >= xmin) & (x <= xmax) &
                    (y >= ymin) & (y <= ymax))

        ndim = val.ndim

        # Keep data inside bbox
        if ndim == 1:
            val = val[idx]
        else:
            val = val[idx,:]

        # Append the variable to list
        tmp.append(val)

    print("Finished joining:",key)

    # Create dataset and add the stacked variable with compression
    if ndim == 1:
        fo.create_dataset(key,data=np.hstack(tmp), compression='lzf')
    else:
        fo.create_dataset(key,data=np.vstack(tmp), compression='lzf')

# Add time to the dataset separatly
fo.create_dataset("time",data=time, compression='lzf')
fo.close()

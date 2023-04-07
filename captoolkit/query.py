"""
Query entire data base (tens of thousands of HDF5 files) and 
extract selected variables within search radius given locations.

Example:
    python get_tseries.py ~/data/ers1/floating/filt_scat_det/joined_pts_ad.h5

Notes:
    To plot extracted time series => plot_tseries2.py

"""
import sys
import h5py
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.spatial import cKDTree

from scipy import optimize

#--- EDIT --------------------------------------------------

# Time series locations (center of search radius)
locations = [
        (-750000, 900000),    # Filchner
        (-1300000, 700000),   # Ronne 1
        (-1000000, 700000),   # Ronne 2
        (-1200000, 400000),   # Ronne 3
        (-1596860, -307885),  # PIG
        (-2189260, 1132550),  # Larsen C 1
        (-2250000, 1100000),  # Larsen C 2
        (-47411, -994331),    # Ross
        (1961740, 704720),    # Amery
        (2275530, -1054890),  # Totten
        ]

# If locations in lon/lat
lonlat = False

#-----------------------------------------------------------


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def is_empty(ifile):
    """If file is corruted or empty, return True."""
    try:
        with h5py.File(ifile, 'r') as f:
            if bool(f.keys()) and f['lon'].size > 0:
                return False
            else:
                return True
    except:
        return True


# Define command-line arguments
parser = argparse.ArgumentParser(description='Queries full data base')
parser.add_argument(
        'files', metavar='file', type=str, nargs='+',
        help='file(s) to process (HDF5)')
parser.add_argument(
        '-o', metavar=('ofile'), dest='ofile', type=str, nargs=1,
        help=('output file name'),
        default=['tseries_from_query.h5'],)
parser.add_argument(
        '-v', metavar=('var'), dest='vnames', type=str, nargs='+',
        help=('Variables to extract'),
        default=['lon', 'lat', 't_year', 'h_cor'],)
parser.add_argument(
        '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
        help=('search radius (km)'),
        default=[5],)
args = parser.parse_args()

for p in vars(args).iteritems(): print p

files  = args.files[:]
ofile  = args.ofile[0]
vnames  = args.vnames[:]
r = args.radius[0] * 1e3

xvar, yvar = vnames[:2]

# Pass str if "Argument list too long"
if len(files) == 1: files = glob(files[0])  

if lonlat:
    locations = [transform_coord(4326, 3031, lo, la) for lo, la in locations]

out = {}

for ifile in files:

    if is_empty(ifile): continue

    with h5py.File(ifile, 'r') as f:
        lon = f[xvar][:]
        lat = f[yvar][:]

        x, y = transform_coord(4326, 3031, lon, lat)

        print 'building kd-tree for:', ifile
        Tree = cKDTree(zip(x, y))

        # Query the Tree for locations
        for loc, (xi,yi) in enumerate(locations):
            idx = Tree.query_ball_point((xi, yi), r)
            
            if len(idx) == 0: continue

            # Extract vars
            for k in vnames: out.setdefault(k, []).extend(f[k][:][idx])
            out.setdefault('loc', []).extend(np.repeat(loc, len(idx)))
                
if out: 
    with h5py.File(ofile, 'w') as f:
        for k,v in out.items(): f[k] = v
    print 'out ->', ofile

if 0:
    # Plot for testing
    plt.figure()
    plt.plot(out[xvar], out[yvar], '.')
    plt.figure()
    plt.plot(out['t_year'], out['h_cor'], '.')
    plt.show()

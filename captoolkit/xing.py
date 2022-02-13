#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
import sys
import glob
import pyproj
import pandas as pd
import numpy as np
import h5py
import argparse
from scipy.spatial import cKDTree
from altimutils import tiffread
from altimutils import interp2d
from altimutils import transform_coord
from altimutils import mad_std

"""

Program for point-to-point comparison of two altimetry datasets using closest
point inside a provided seacrh radius and time window. It requires two h5
files: One as reference and the other as comparison. Statistics are always
computed as file1 minus file2. Further, the user can provide a slope
raster to provide the surface slope value for each comparison point.

Notes:
    To improve computational speed the data can be sub-sampled for each dataset
    so only n:th point in ref. and m:th point in obs. are used.

    Suggest file with largest amount of observations be file2 as this will
    decrease computational time in the form of looping.

    If dummy variable name provided for time the time-span between observation
    is not used to rejectr data (all data is compared).

Example:
    xing.py file.h5 file2.h5 -d 50 -p 3413 -v lon lat t h -u lon lat t h \
    -t 0.083 -i 1 1 -m CS2 ATM -o stats.csv

    xing.py file.h5 file2.h5 -d 50 -p 3413 -v lon lat t h -u lon lat t h \
    -t 0.083 -i 1 1 -m CS2 ATM -o stats.csv -n slope.tif

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
    Jet Propulsion Laboratory, California Institute of Technology
"""

# Output description of solution
des = ('Program for P2P comparison between two altimetry datasets.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=des)

parser.add_argument(
        'file1', metavar='file1', type=str, nargs=1,
        help='input file-1 (.h5)',)

parser.add_argument(
        'file2', metavar='file2', type=str, nargs=1,
        help='input file-2 (.h5)',)

parser.add_argument(
        '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
        help='output filename for statistics (.csv)',)

parser.add_argument(
        '-d', metavar='dxy', dest='dxy', type=float, nargs=1,
        help=('spatial resolution of comparison grid (m)'),
        default=[50],)

parser.add_argument(
        '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
        help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
        default=['3031'],)

parser.add_argument(
        '-v', metavar=('x','y','t','h'), dest='vnames_ref', type=str, nargs=4,
        help=('name of varibales in reference file'),
        default=['lon','lat','t_year','h_elv'],)

parser.add_argument(
        '-u', metavar=('x','y','t','h'), dest='vnames_com', type=str, nargs=4,
        help=('name of varibales in comparison file'),
        default=['lon','lat','t_year','h_elv'],)

parser.add_argument(
        '-s', metavar=('s_min','s_max'), dest='slope', type=float, nargs=2,
        help=('min and max slope interval (deg)'),
        default=[0.0,1.0],)

parser.add_argument(
        '-t', metavar='dt', dest='tspan', type=float, nargs=1,
        help=('only compare data with time-span < dt'),
        default=[3./12],)

parser.add_argument(
        '-n', metavar=('slope_file'), dest='fslope',  type=str, nargs=1,
        help='name of slope raster file',
        default=[None],)

parser.add_argument(
        '-i', metavar=('n_ref','n_com'), dest='ncomp', type=int, nargs=2,
        help=('sub-sample data using every n:th point'),
        default=[1,1],)

parser.add_argument(
        '-m', metavar=('fname1','fname2'), dest='fname', type=str, nargs=2,
        help=('name of data column for header (string)'),
        default=['Ref.','Obs.'],)

# Create parser argument container
args = parser.parse_args()

# Pass arguments
fref = args.file1[0]
fcom = args.file2[0]
ofile = args.ofile[0]
dxy = args.dxy[0]
proj = args.proj[0]
vref = args.vnames_ref[:]
cref = args.vnames_com[:]
s_min = args.slope[0]
s_max = args.slope[1]
tspan = args.tspan[0]
fslp = args.fslope[0]
nref = args.ncomp[0]
ncom = args.ncomp[1]
fname1 = args.fname[0]
fname2 = args.fname[1]

# Check for slope file
if fslp is not None:

    # Read slope file if needed
    (X, Y, Z) = tiffread(fslp)[0:3]
    
# Load file
with h5py.File(fref, 'r') as fr:

    # Load ref. variables
    xr = fr[vref[0]][:]
    yr = fr[vref[1]][:]
    tr = fr[vref[2]][:] if vref[2] in fr else np.ones(xr.shape)
    zr = fr[vref[3]][:]

# Sub-selection of array
xr = xr[::nref]
yr = yr[::nref]
tr = tr[::nref]
zr = zr[::nref]

# Copy locations
lon_r, lat_r = xr[:], yr[:]

# Transform to wanted coordinate system
(xr, yr) = transform_coord('4326', proj, xr, yr)

# Load file
with h5py.File(fcom, 'r') as fr:

    # Load com. variables
    xc = fr[cref[0]][:]
    yc = fr[cref[1]][:]
    tc = fr[cref[2]][:] if cref[2] in fr else np.ones(xc.shape)
    zc = fr[cref[3]][:]

# Checks if time rejection used
if np.all(tc == 1) and np.all(tr == 1):
    print('-> time-span rejection not used comparing all data...')

# Sub-selection of array
xc = xc[::ncom]
yc = yc[::ncom]
tc = tc[::ncom]
zc = zc[::ncom]

# Transform to wanted coordinate system
(xc, yc) = transform_coord('4326', proj, xc, yc)

# Boundary limits: the smallest spatial domain (m)
xmin = max(np.nanmin(xr), np.nanmin(xc))
xmax = min(np.nanmax(xr), np.nanmax(xc))
ymin = max(np.nanmin(yr), np.nanmin(yc))
ymax = min(np.nanmax(yr), np.nanmax(yc))

# Index of data
idx_c = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)
idx_r = (xr > xmin) & (xr < xmax) & (yr > ymin) & (yr < ymax)

# Cut to same area as reference
xc, yc, zc, tc = xc[idx_c], yc[idx_c], zc[idx_c], tc[idx_c]
xr, yr, zr, tr = xr[idx_r], yr[idx_r], zr[idx_r], tr[idx_r]

# Construct KD-Tree
tree = cKDTree(np.c_[xc, yc])

# Output vector
dz = np.ones(len(zr)) * np.nan
xo = np.ones(len(zr)) * np.nan
yo = np.ones(len(zr)) * np.nan
z1 = np.ones(len(zr)) * np.nan
z2 = np.ones(len(zr)) * np.nan
t1 = np.ones(len(zr)) * np.nan
t2 = np.ones(len(zr)) * np.nan
do = np.ones(len(zr)) * np.nan

# Loop trough reference points
for kx in range(len(xr)):

    # Query KD-Tree
    dr, ky = tree.query(np.c_[xr[kx], yr[kx]], k=1)
    
    # Check if we should compute
    if (dr > dxy) or np.abs(tr[kx] - tc[ky]) > tspan:
        continue

    # Compute difference
    dz[kx] = zr[kx] - zc[ky]

    # Save location where we have difference
    z1[kx] = zr[kx]
    z2[kx] = zc[ky]
    xo[kx] = lon_r[kx]
    yo[kx] = lat_r[kx]
    t1[kx] = tr[kx]
    t2[kx] = tc[ky]
    do[kx] = dr

# Check if we are binning by slope
if fslp:

    # Interpolate slope to data
    slp = interp2d(X, Y, Z, xr, yr, order=1)

else:

    # No slope provided
    slp = np.ones(len(zr)) * 9999

# Find NaN-values
i_nan = ~np.isnan(dz)

# Create data container
foo = np.vstack((xo[i_nan],yo[i_nan],\
                  t1[i_nan],t2[i_nan],\
                  z1[i_nan],z2[i_nan],
                  dz[i_nan],t1[i_nan]-t2[i_nan],\
                  do[i_nan],slp[i_nan])).T

# Round data to four decimals
foo = np.around(foo, 4)

# Name of ouput columsn
cols = ['lon', 'lat', 't1', 't2', fname1, fname2, fname1+'-'+fname2, \
        't1-t2', 'Distance', 'Slope']

# Create data frame
df = pd.DataFrame(foo, columns=cols)

# Save to csv
df.to_csv(ofile, sep=',', index=False)

# Compute statistics
avg = np.around(np.nanmedian(dz),3)
std = np.around(mad_std(dz),3)
nob = len(dz[~np.isnan(dz)])

# Print statistics to screen
print(fname1+'-'+fname2, 'Mean:', avg, 'Std.dev:', std, 'Nobs:', nob)

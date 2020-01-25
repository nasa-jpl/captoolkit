#!/usr/bin/env python2
import warnings
warnings.filterwarnings("ignore")
import sys
import glob
import pyproj
import pandas as pd
import numpy as np
import h5py
import argparse
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates
from scipy.stats import binned_statistic_2d
from scipy.spatial import cKDTree

"""

    Program for computing statistics between two altimetry data sets

"""

def interp2d(xd, yd, data, xq, yq, **kwargs):
    """ Interpolator from raster to point """

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0, :]
    yd = yd[:, 0]

    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1] - xd[0]), (yd[1] - yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq * ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq * ones(xq.size)

    xp = (xq - xd[0]) * (nx - 1) / (xd[-1] - xd[0])
    yp = (yq - yd[0]) * (ny - 1) / (yd[-1] - yd[0])

    coord = np.vstack([yp, xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def sigma_filter(x, xmin=-9999, xmax=9999, tol=5, alpha=5):
    """ Iterative outlier filter """

    # Set default value
    tau = 100.0

    # Remove data outside selected range
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan

    # Initiate counter
    k = 0

    # Outlier rejection loop
    while tau > tol:

        # Compute initial rms
        rmse_b = mad_std(x)

        # Compute residuals
        dh_abs = np.abs(x - np.nanmedian(x))

        # Index of outliers
        io = dh_abs > alpha * rmse_b

        # Compute edited rms
        rmse_a = mad_std(x[~io])

        # Determine rms reduction
        tau = 100.0 * (rmse_b - rmse_a) / rmse_a

        # Remove data if true
        if tau > tol or k == 0:
            # Set outliers to NaN
            x[io] = np.nan

            # Update counter
            k += 1

    return x


def geotiffread(ifile):
    """ Read Geotiff file """

    file = gdal.Open(ifile, GA_ReadOnly)

    metaData = file.GetMetadata()
    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    proj = src.ExportToWkt()

    Nx = file.RasterXSize
    Ny = file.RasterYSize

    trans = file.GetGeoTransform()

    dx = trans[1]
    dy = trans[5]

    Xp = np.arange(Nx)
    Yp = np.arange(Ny)

    (Xp, Yp) = np.meshgrid(Xp, Yp)

    X = trans[0] + (Xp + 0.5) * trans[1] + (Yp + 0.5) * trans[2]
    Y = trans[3] + (Xp + 0.5) * trans[4] + (Yp + 0.5) * trans[5]

    band = file.GetRasterBand(1)

    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def wrapTo360(lon):
    """ Wrap longitude to 360 deg """
    positiveInput = (lon > 0.0)
    lon = np.mod(lon, 360.0)
    lon[(lon == 0) & positiveInput] = 360.0
    return lon


# Wrap longitude to 180 deg
def wrapTo180(lon):
    """Wrap longitude to 180 deg """
    q = (lon < -180.0) | (180.0 < lon)
    lon[q] = wrapTo360(lon[q] + 180.0) - 180.0
    return lon


# Output description of solution
description = ('Program for computing statistics between two altimetry datasets.')

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
        '-r', metavar='fref', dest='fref', type=str, nargs='+',
        help='reference file(s)',
        required=True)

parser.add_argument(
        '-f', metavar='fcomp', dest='fcomp', type=str, nargs='+',
        help='comparison files(s)',
        required=True)

parser.add_argument(
        '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
        help='name of output statistics file',)

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
        default=['lon','lat','t_year','h_cor'],)

parser.add_argument(
        '-u', metavar=('x','y','t','h'), dest='vnames_com', type=str, nargs=4,
        help=('name of varibales in comparison file'),
        default=['lon','lat','t_year','h_cor'],)

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

# Create parser argument container
args = parser.parse_args()

# Pass arguments
fref  = args.fref
fcom  = args.fcomp
ofile = args.ofile[0]
dxy   = args.dxy[0]
proj  = args.proj[0]
vref  = args.vnames_ref[:]
cref  = args.vnames_com[:]
s_min = args.slope[0]
s_max = args.slope[1]
tspan = args.tspan[0]
fslp  = args.fslope[0]
nref  = args.ncomp[0]
ncom  = args.ncomp[1]

# Initiate statistics
Fref = []
Fcom = []
mean = []
stdv = []
rmse = []
vmin = []
vmax = []
nobs = []

# Check for slope file
if fslp is not None:

    # Read slope file
    (X, Y, Z) = geotiffread(fslp)[0:3]

# Loop trough reference list
for f_ref in fref:

    # Load file
    with h5py.File(f_ref, 'r') as fr:

        # Load ref. variables
        xr = fr[vref[0]][::nref]
        yr = fr[vref[1]][::nref]
        tr = fr[vref[2]][::nref]
        zr = fr[vref[3]][::nref]

    # Copy locations
    lon_r, lat_r = xr[:], yr[:]

    # Transform to wanted coordinate system
    (xr, yr) = transform_coord('4326', proj, xr, yr)

    # Compute bounding box
    xmin, xmax, ymin, ymax = np.min(xr), np.max(xr), np.min(yr), np.max(yr)

    # Loop through comparison list
    for f_com in fcom:

        # Load file
        with h5py.File(f_com, 'r') as fr:

            # Load com. variables
            xc = fr[cref[0]][::ncom]
            yc = fr[cref[1]][::ncom]
            tc = fr[cref[2]][::ncom]
            zc = fr[cref[3]][::ncom]

        # Check mean time difference
        # if np.abs(tr.mean() - tc.mean()) > 3 * tspan: continue

        # Transform to wanted coordinate system
        (xc, yc) = transform_coord('4326', proj, xc, yc)

        # Index of data
        idx = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)

        # Cut to same area as reference
        xc, yc, zc, tc = xc[idx], yc[idx], zc[idx], tc[idx]

        # Construct KD-Tree
        tree = cKDTree(list(zip(xc, yc)))

        # Output vector
        dz = np.ones(len(zr)) * np.nan
        xo = np.ones(len(zr)) * np.nan
        yo = np.ones(len(zr)) * np.nan
        z1 = np.ones(len(zr)) * np.nan
        z2 = np.ones(len(zr)) * np.nan
        t1 = np.ones(len(zr)) * np.nan
        t2 = np.ones(len(zr)) * np.nan

        # Loop trough reference points
        for kx in range(len(xr)):

            # Query KD-Tree
            dr, ky = tree.query((xr[kx], yr[kx]), k=1)

            # Check if we should compute
            if dr > dxy: continue

            if np.abs(tr[kx]-tc[ky]) > tspan: continue
            
            # Compute difference
            dz[kx] = zr[kx] - zc[ky]

            # Save location where we have difference
            z1[kx] = zr[kx]
            z2[kx] = zc[ky]
            xo[kx] = lon_r[kx]
            yo[kx] = lat_r[kx]
            t1[kx] = tr[kx]
            t2[kx] = tc[ky]

        # If no data skip
        if np.all(np.isnan(dz)):
            continue

        # Light filtering of outliers
        dz = sigma_filter(dz)

        # Check if we are binning by slope
        if fslp:

            # Interpolate slope to data
            slp = interp2d(X, Y, Z, xr, yr, order=1)

            # Cull using surface slope
            dz[(slp < s_min) & (slp > s_max)] = np.nan

        else:

            # No slope provided
            slp = np.ones(len(zr)) * 9999

        # Find NaN-values
        inan = ~np.isnan(dz)

        # Save to csv file
        data = {'lat'   : np.around(yo[inan],4),
                'lon'   : np.around(xo[inan],4),
                't_ref' : np.around(t1[inan],3),
                't_com' : np.around(t2[inan],3),
                'v_ref' : np.around(z1[inan],3),
                'v_com' : np.around(z2[inan],3),
                'v_diff': np.around(dz[inan],3),
                'dt'    : np.around(t1[inan]-t2[inan],3),
                'slope' : np.around(slp[inan],3),}

        # Get name only and not path to files
        f_ref_i = f_ref[f_ref.rfind('/') + 1:]
        f_com_i = f_com[f_com.rfind('/') + 1:]

        # Create data frame
        df = pd.DataFrame(data, columns=['lat', 'lon', 't_ref', 't_com', 'v_ref', 'v_com', 'v_diff', 'dt', 'slope'])

        # Save to csv
        df.to_csv(f_ref_i+'_'+f_com_i+'.csv', sep=',', index=False)

        # Compute statistics
        avg = np.around(np.nanmean(dz),3)
        std = np.around(np.nanstd(dz),3)
        rms = np.around(np.sqrt(avg**2 + std**2),3)
        min = np.around(np.nanmin(dz),3)
        max = np.around(np.nanmax(dz),3)
        nob = len(dz[~np.isnan(dz)])

        # Save all the stats
        Fref.append(f_ref_i)
        Fcom.append(f_com_i)
        mean.append(avg)
        stdv.append(std)
        rmse.append(rms)
        vmin.append(min)
        vmax.append(max)
        nobs.append(nob)

        # Print statistics to screen
        print(('Ref:' ,f_ref_i, 'Comp:', f_com_i, 'Mean:', avg, 'Std:', std, 'RMS:', rms, 'Nobs:', nob))

        # Plot data if wanted
        if 0:
            plt.figure()
            plt.hist(dz[~np.isnan(dz)], 50)
            plt.show()

# Compute weighted averages
m = np.asarray(mean)
n = np.asarray(nobs)
s = np.asarray(stdv)

# Compute weights
w = n / (s * s * np.sum(n))

# Weighted average and standard deviation
aw = np.sum(w * m)/np.sum(w)
sw = np.sqrt(1 / np.sum(w))

# Print weighted statistics
print('#############################################################')
print(('| Weighted Statistics |', 'Wmean:', np.around(aw, 2), 'Wstd:', np.around(sw, 2), \
        'WRMSE:', np.around(np.sqrt(aw**2 + sw**2), 2), '|'))
print('#############################################################')

# Create data container
raw_data = {'Reference'  : Fref,
            'Comparison' : Fcom,
            'Mean'       : mean,
            'Std.dev'    : stdv,
            'RMSE'       : rmse,
            'Min'        : vmin,
            'Max'        : vmax,
            'Nobs'       : nobs,}

# Create data frame
df = pd.DataFrame(raw_data, columns = ['Reference','Comparison','Mean','Std.dev','RMSE','Min','Max','Nobs'])

# Save to csv
df.to_csv(ofile, sep=',', index=False)

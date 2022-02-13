#!/usr/bin/env python
import os
import sys
import glob
import h5py
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from numba import jit
from altimutils import tiffread
from altimutils import interp2d
from altimutils import transform_coord
"""

Program computes a correction for the slope-induced error for radar altimeters
given an a-priori DEM (.tif). The user can select either between the
direct-method or the relocation method, please see Bamber et al. 1994
(Ice Sheet Altimeter Processing Scheme) for more information. The program can
process a single file or several files (tracks) in parallel. Data is saved in
new variables provided a user defined suffix. The program needs as input:
longitude, latitude, elevation and range to be able to compute the correction.
If the range not avalibale a constant altitude can be provided to compute the
range as R = A - h. The program also provides an estimate of the distance to the
reflection point up-slope that can be used for quality control (dist_cor).

Note:
    To obtain the best results it is recommended that the user provide a DEM at
    resultion that roughly corresponds to the pulse-limited footprint of the
    radar altimeters (~3km). QGIS is a good open-source software to perform the
    resmapling using the reproject option (averaging).

    If constant altitude is used and range is provided "-a" will overide the
    use of the range variable. The sensitivity to the correction of using a
    constant altitude is relativly low.

    The user can provide a maximum allowed slope value for the correction and
    if the estimted slope exceeds this value the max-value is used for the
    computation. A maxmimum value of 1.5 degrees is a good tradeoff for the old
    pulse-limited missions as this roughly the limit of their capability to
    measure topography.


Example:
    corrslope.py ./file(s).h5 -d dem.tif -m RM -j 3413 -l 1.5 \
    -v lon lat h range -n 32

    corrslope.py ./file(s).h5 -d dem.tif -m RM -j 3413 -l 1.5 \
    -v lon lat h_elv dummy -n 32 -a 800


Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)
    Jet Propulsion Laboratory, California Institute of Technology

"""

# Define command-line arguments
parser = argparse.ArgumentParser(description='Slope correction for altimetry')

parser.add_argument(
    'files', metavar='file', type=str, nargs='+',
    help='files to process (h5) ')

parser.add_argument(
    '-o', metavar=('outdir'), dest='outdir', type=str, nargs=1,
    help='output dir, default same as input',
    default=[None],)

parser.add_argument(
    '-d', metavar=('fdem'), dest='fdem', type=str, nargs=1,
    help='name of DEM file (.tif)',
    default=[None],)

parser.add_argument(
    '-m', metavar=None, dest='mode', type=str, nargs=1,
    help=('corr. type: direct (DM) or relocation (RM) method'),
    choices=('DM', 'RM'), default=['RM'],)

parser.add_argument(
    '-j', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
    help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
    default=['3031'],)

parser.add_argument(
    '-k', metavar=('kernel_size'), dest='kern', type=int, nargs=1,
    help=('smoothing of DEM using kernel-average'),
    default=[None],)

parser.add_argument(
    '-l', metavar=('max_slope'), dest='smax', type=float, nargs=1,
    help=('max value allowed for slope (deg)'),
    default=[None],)

parser.add_argument(
    '-v', metavar=('x', 'y', 'h', 'r'), dest='vnames', type=str, nargs=4,
    help='lon/lat/height/range variable names in HDF5',
    default=['lon', 'lat', 'height', 'range'],)

parser.add_argument(
    '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
    help="for parallel processing of multiple files",
    default=[1],)

parser.add_argument(
    '-a', metavar=('altitude'), dest='alt', type=float, nargs=1,
    help=('constant altitude if range not avaliable (km)'),
    default=[None],)

parser.add_argument(
    '-s', metavar=('suffix'), dest='suffix', type=str, nargs=1,
    help=('suffix for corrected vars, default is "_cor"'),
    default=['_cor'],)

parser.add_argument(
    '-e', dest='ending', action='store_true',
    help=('add RM or DM to filename'),
    default=[False])

args = parser.parse_args()

# Data input
files = args.files
opath = args.outdir[0]
fdem = args.fdem[0]
mode = args.mode[0]
proj = args.proj[0]
kern = args.kern[0]
smax = args.smax[0]
vnames = args.vnames
njobs = args.njobs[0]
alt = args.alt[0]
add_suffix = args.suffix[0]
ending = args.ending[0]

# Print parameters to screen
print('parameters:')
for arg in list(vars(args).items()):
    print(arg)

@jit(nopython=True)
def lpfilt(image, kernel):
    """
        Low-pass filter using kernel average
    """

    # Copy original array
    image_filt = image.copy()

    # Get index of center coordinate
    ki = int(np.floor(kernel / 2.))

    # Shape of new array
    (n, m) = image.shape

    # Loop trough raster
    for i in range(ki, n - ki, 1):
        for j in range(ki, m - ki, 1):

            # Get window
            img = image[i-ki:i+ki+1, j-ki:j+ki+1]

            # Predicted filtered value
            image_filt[i, j] = np.nanmean(img)

    # Return filtered image
    return image_filt


@jit(nopython=True)
def gradient(Z, L):
    """
        Computes slope in x and y direction from DEM using the
        Zevenbergen & Thorne algorithm
    """

    # Initiate output parameters
    Sx = np.ones(Z.shape) * np.nan
    Sy = np.ones(Z.shape) * np.nan
    PC = np.ones(Z.shape) * np.nan

    # Shape of new array
    (n, m) = Z.shape

    if dx == dy:
    	L = dx

    # Loop trough raster
    for i in range(1, n - 1, 1):
        for j in range(1, m - 1, 1):

            # Extract 3 x 3 kernel
            z1 = Z[i-1, j+1]
            z2 = Z[i-0, j+1]
            z3 = Z[i+1, j+1]
            z4 = Z[i-1, j-0]
            z5 = Z[i-0, j-0]
            z6 = Z[i+1, j+0]
            z7 = Z[i-1, j-1]
            z8 = Z[i+0, j-1]
            z9 = Z[i+1, j-1]

            #G = (-z4 + z6) / (2 * L)
            #H = (+z2 - z8) / (2 * L)

            G = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8. * L)
            H = ((z3 + 2.0 * z2 + z1) - (z9 + 2.0 * z8 + z7)) / (8. * L)

            D = (0.5 * (z4 + z6) - z5) / (L ** 2)
            E = (0.5 * (z2 + z8) - z5) / (L ** 2)
            F = (-z1 + z3 + z7 - z9) / (4.0 * L ** 2)

            # Compute surface slope in x and y in m/m
            if G == 0 or H == 0:
                # Set to zero
                Sx[i, j] = 0
                Sy[i, j] = 0
                PC[i, j] = 0
            else:
                # Add values
                Sx[i, j] = G
                Sy[i, j] = H
                PC[i, j] = 2.0*(D*G*G + E*H*H + F*G*H) / (G*G + H*H)

    # Return gradients and curvature
    return Sx, Sy, PC

# Get file list from directory
if len(files) == 1:
    files = glob.glob(files[0])

# Warning for constant altitude
if alt is not None:
    print('-> WARNING! Constant altitude is used!')

# Change to radians
if smax is not None:
    smax *= np.pi / 180.0

# Ellipsoid parameters - WGS84
a = 6378137.0
f = 1.0 / 298.2572235630
b = (1 - f) * a
e2 = (a * a - b * b) / (a * a)

print('-> Reading elevation model ...')

# Load DEM from memory
Xd, Yd, Zd, dx, dy = tiffread(fdem)[0:5]

# Smooth DEM
if kern is not None:

    print('-> Smoothing elevation model ...')

    # Filter the input grids
    Zd = lpfilt(Zd.copy(), kern)

print('-> Computing directional slope ...')

# Compute surface gradient in x and y direction
Sx, Sy = gradient(Zd.copy(), dx)[0:2]

# Main algorithm
def main(ifile):

    import warnings
    warnings.filterwarnings("ignore")

    # Check for empty file
    if os.stat(ifile).st_size == 0:
        return

    # Get variable names
    xvar, yvar, zvar, rvar = vnames

    # Set output variable names
    oxvar = xvar + add_suffix
    oyvar = yvar + add_suffix
    ozvar = zvar + add_suffix

    # Load data points - HDF5
    with h5py.File(ifile) as f:

        lon = f[xvar][:]
        lat = f[yvar][:]
        elv = f[zvar][:]
        rng = f[rvar][:] if rvar in f else np.zeros(lon.shape)

    # Check if empty file
    if len(lon) == 0:
        return

    # Satellite elevation
    h = elv.copy()

    # Check if range is available
    if alt is not None:

        # Set altitude from input
        A = alt * 1e3

        # Calulate range
        R = A - h

    else:

        # Get range estimates
        R = rng.copy()

    # Reproject coordinates to conform with grid
    (x, y) = transform_coord('4326', proj, lon.copy(), lat.copy())

    # Satellite coordinates in radians
    lon *= np.pi / 180
    lat *= np.pi / 180

    # Interpolate slopes to data
    s_x = interp2d(Xd, Yd, Sx, x, y, order=1)
    s_y = interp2d(Xd, Yd, Sy, x, y, order=1)

    # Check for North or South Hemisphere
    if np.all(lat < 0):
        # South Hemisphere (longitude)
        phi_corr = np.arctan2(x,y)
    else:
        # North Hemisphere (longitude)
        phi_corr = np.arctan2(x,-y)

    # Compute aspect and correct to geopraphical North
    asp = np.arctan2(s_y,-s_x) - phi_corr.copy()

    # Compute slope magnitude in radians
    slp = np.arctan(np.sqrt(s_x**2 + s_y**2))

    # Edit slope and set to maximum allowed
    if smax is not None:
        slp[slp > smax] = smax

    # Curvature parameters in lon/lat directions
    if mode == 'RM':

        # Takes into account azimuth
        r_lat = (a * (1 - e2)) / ((1 - e2 * (np.sin(lat) ** 2)) ** (1.5))
        r_lon = (a * np.cos(lat)) / (np.sqrt(1 - e2 * np.sin(lat) ** 2))
        r_tot = (r_lat * r_lon) / (r_lat * np.cos(lat) * np.sin(asp)**2
                - r_lon * np.cos(asp)**2)

    else:

        # No azimuth dependancy
        r_lon = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        r_lat = (a * (1 - e2)) / ((1 - e2 * (np.sin(lat) ** 2)) ** (1.5))
        r_tot = np.sqrt(r_lat * r_lon) + R * np.cos(slp) + h

    # Correction for Earth Curvature
    dR = ((R * np.sin(slp)) ** 2) / (2 * r_tot)

    # Distance to relection point based on slope
    d_slp = R * np.sin(slp)

    # Direct method (DM)
    if mode == "DM":

        # Slope correction - Direct method
        h_cor = R - (R * np.cos(slp)**(-1) + dR)

        # Corrected elevation
        h_echo = h + h_cor

        # Dictionary to save data into HDF5
        OFILEd = {ozvar:h_echo,'dist_cor':d_slp}

    # Relocation method (RM)
    if mode == "RM":

        # Slope correction - Relocation method
        h_cor = R - (R * np.cos(slp) + dR)
        
        # Correct elevation
        h_echo = h + h_cor

        # Directional correction based on slope and aspect
        dlat = R * np.sin(slp) * np.cos(asp) / r_lat
        dlon = R * np.sin(slp) * np.sin(asp) / r_lon

        # Migrate to approximate echo location
        lat_echo = lat.copy() + dlat
        lon_echo = lon.copy() + dlon

        # Converte to degrees
        lat_echo *= 180 / np.pi
        lon_echo *= 180 / np.pi

        # Dictionary to save data into HDF5
        OFILEd = {oxvar:lon_echo,oyvar:lat_echo,\
        ozvar:h_echo,'dist_cor':d_slp}

    # Get output file name (to replace input name)
    path, fname = os.path.split(ifile)
    name, ext = os.path.splitext(fname)
    suffix = '_DM' if mode == 'DM' else '_RM'
    path = opath if opath else path

    # Add file ending if needed
    if ending:
        ofile = os.path.join(path, name + suffix + ext)
    else:
        ofile = os.path.join(path, name + ext)

    # Save corrections
    with h5py.File(ifile, 'a') as f:
        for k, v in list(OFILEd.items()):
            try:
                f[k] = v
            except:
                f[k][:] = v

    os.rename(ifile, ofile)
    print('output file:', ofile, 'Average correction:',\
        np.around(np.nanmean(h - h_echo),2),'m')

if njobs == 1:
    print('running sequential code ...')
    [main(f) for f in files]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

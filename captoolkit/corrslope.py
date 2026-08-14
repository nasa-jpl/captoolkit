#!/usr/bin/env python
import os
import sys
sys.path.append("/home/nilssonj/Altimetry/pyAltim")
import glob
import h5py
import pyproj
import argparse
import numpy as np
from numba import jit
from altimutils import tiffread
from altimutils import interp2d

"""
Program computes a correction for slope-induced errors in radar altimetry using
an a-priori DEM (.tif). The user can select between the direct method (DM) and
the relocation method (RM); see Bamber et al. (1994), Ice Sheet Altimeter
Processing Scheme, for more information.

The program can process a single file or multiple files (tracks) in parallel.
Corrected data are saved as new variables using a user-defined suffix. The
required input variables are longitude, latitude, elevation, and range. If
range is not available, a constant satellite altitude can be provided and the
range is estimated as R = A - h.

DEM slopes are computed in the projected coordinate system and corrected for
the local projection scale. The grid-based slope direction is converted to a
true geographic azimuth, and Earth-curvature corrections are computed using
the WGS84 ellipsoid. For the relocation method, the estimated reflection point
is relocated along the true geographic up-slope direction using an ellipsoidal
geodesic. The program also provides the horizontal distance to the relocated
reflection point (dist_cor), which can be used for quality control.

Note:
    The DEM must use a projected coordinate reference system with linear units,
    such as EPSG:3031 for Antarctica or EPSG:3413 for Greenland. Geographic
    coordinate systems such as EPSG:4326 are not supported for the DEM slope
    calculation.

    To obtain the best results, it is recommended that the DEM resolution
    roughly corresponds to the pulse-limited footprint of the radar altimeter
    (~3 km). A higher-resolution DEM can be resampled using spatial averaging
    before applying the correction.

    If a constant altitude is provided using "-a", it overrides the range
    variable supplied with "-v". The sensitivity of the slope correction to
    using a constant satellite altitude is relatively small.

    The user can provide a maximum allowed slope for the correction. If the
    estimated DEM slope exceeds this value, the maximum slope is used while
    retaining the estimated up-slope direction. A maximum slope of 1.5 degrees
    is generally a reasonable choice for older pulse-limited radar altimeters.

Example:
    corrslope.py ./file(s).h5 -d dem.tif -m RM -j 3413 -l 1.5 \
    -v lon lat h range -n 32

    corrslope.py ./file(s).h5 -d dem.tif -m RM -j 3413 -l 1.5 \
    -v lon lat h_elv dummy -n 32 -a 800


Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Johan Nilsson   (johan.n.nilsson@geo.uu.se)
    Fernando Paolo  (paolofer@jpl.nasa.gov)
    Alex Gardner     (alex.s.gardner@jpl.nasa.gov)

    Department of Earth Sciences, Uppsala University
    Jet Propulsion Laboratory, California Institute of Technology
"""


parser = argparse.ArgumentParser(description='Slope correction for radar altimetry')

parser.add_argument(
    'files', metavar='file', type=str, nargs='+',
    help='files to process (h5)')

parser.add_argument(
    '-o', metavar='outdir', dest='outdir', type=str, nargs=1,
    help='output dir, default same as input',
    default=[None])

parser.add_argument(
    '-d', metavar='fdem', dest='fdem', type=str, nargs=1,
    help='name of DEM file (.tif)',
    required=True)

parser.add_argument(
    '-m', metavar=None, dest='mode', type=str, nargs=1,
    help='corr. type: direct (DM) or relocation (RM)',
    choices=('DM', 'RM'), default=['RM'])

parser.add_argument(
    '-j', metavar='epsg_num', dest='proj', type=str, nargs=1,
    help='projection: EPSG number (AnIS=3031, GrIS=3413)',
    default=['3031'])

parser.add_argument(
    '-k', metavar='kernel_size', dest='kern', type=int, nargs=1,
    help='smoothing of DEM using kernel-average',
    default=[None])

parser.add_argument(
    '-l', metavar='max_slope', dest='smax', type=float, nargs=1,
    help='max value allowed for slope (deg)',
    default=[None])

parser.add_argument(
    '-v', metavar=('x', 'y', 'h', 'r'), dest='vnames', type=str, nargs=4,
    help='lon/lat/height/range variable names in HDF5',
    default=['lon', 'lat', 'height', 'range'])

parser.add_argument(
    '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
    help='parallel processing of multiple files',
    default=[1])

parser.add_argument(
    '-a', metavar='altitude', dest='alt', type=float, nargs=1,
    help='constant altitude if range is not available (km)',
    default=[None])

parser.add_argument(
    '-s', metavar='suffix', dest='suffix', type=str, nargs=1,
    help='suffix for corrected vars, default "_cor"',
    default=['_cor'])

parser.add_argument(
    '-e', dest='ending', action='store_true',
    help='add RM or DM to filename',
    default=False)

args = parser.parse_args()


# Inputs
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
ending = args.ending

print('parameters:')
for arg in vars(args).items():
    print(arg)


# DEM smoothing
@jit(nopython=True)
def lpfilt(image, kernel):

    image_filt = image.copy()
    ki = kernel // 2
    n, m = image.shape

    for i in range(ki, n - ki):
        for j in range(ki, m - ki):
            img = image[i-ki:i+ki+1, j-ki:j+ki+1]
            image_filt[i, j] = np.nanmean(img)

    return image_filt


# DEM gradient
@jit(nopython=True)
def gradient(Z, dx, dy):

    Sx = np.full(Z.shape, np.nan)
    Sy = np.full(Z.shape, np.nan)

    n, m = Z.shape

    for i in range(1, n - 1):
        for j in range(1, m - 1):

            z1 = Z[i-1, j+1]
            z2 = Z[i,   j+1]
            z3 = Z[i+1, j+1]

            z4 = Z[i-1, j]
            z5 = Z[i,   j]
            z6 = Z[i+1, j]

            z7 = Z[i-1, j-1]
            z8 = Z[i,   j-1]
            z9 = Z[i+1, j-1]

            # dz/dx
            Sx[i, j] = ((z1 + 2.0*z2 + z3) - (z7 + 2.0*z8 + z9)) / (8.0 * dx)

            # dz/dy
            Sy[i, j] = ((z9 + 2.0*z6 + z3) - (z7 + 2.0*z4 + z1)) / (8.0 * dy)

    return Sx, Sy


# Get files
if len(files) == 1:
    tmp = glob.glob(files[0])
    if len(tmp) > 0:
        files = tmp


# Options
if alt is not None:
    print('-> WARNING! Constant altitude is used!')

if smax is not None:
    smax = np.deg2rad(smax)

if kern is not None:
    if kern < 1:
        raise ValueError('Kernel size must be >= 1')
    if kern % 2 == 0:
        raise ValueError('Kernel size must be odd: 3, 5, 7, ...')


# Projection and ellipsoid
crs_geo = pyproj.CRS.from_epsg(4326)
crs_grid = pyproj.CRS.from_epsg(int(proj))

if crs_grid.is_geographic:
    raise ValueError(
        'DEM CRS must be projected. Geographic CRS such as EPSG:4326 '
        'cannot be used directly for DEM slope calculations.'
    )

to_grid = pyproj.Transformer.from_crs(crs_geo, crs_grid, always_xy=True)
to_geo = pyproj.Transformer.from_crs(crs_grid, crs_geo, always_xy=True)

proj_grid = pyproj.Proj(crs_grid)
geod = pyproj.Geod(ellps='WGS84')


# WGS84 ellipsoid
a = 6378137.0
f = 1.0 / 298.257223563
b = (1.0 - f) * a
e2 = (a*a - b*b) / (a*a)


# Read DEM
print('-> Reading elevation model ...')
Xd, Yd, Zd, dx0, dy0 = tiffread(fdem)[0:5]

# Determine signed spacing directly from coordinate arrays
dx = np.nanmedian(np.diff(Xd[0, :]))
dy = np.nanmedian(np.diff(Yd[:, 0]))

print('-> DEM spacing:', dx, dy)

if not np.isfinite(dx) or not np.isfinite(dy) or dx == 0 or dy == 0:
    raise ValueError('Invalid DEM grid spacing')


# Smooth DEM
if kern is not None:
    print('-> Smoothing elevation model ...')
    Zd = lpfilt(Zd.copy(), kern)


# Compute DEM gradient
print('-> Computing directional slope ...')
Sx, Sy = gradient(Zd.copy(), dx, dy)

print('-> Median gradients:', np.nanmedian(Sx), np.nanmedian(Sy))


def main(ifile):

    import warnings
    warnings.filterwarnings('ignore')

    if not os.path.isfile(ifile):
        print('File not found:', ifile)
        return

    if os.stat(ifile).st_size == 0:
        print('Empty file:', ifile)
        return

    xvar, yvar, zvar, rvar = vnames

    oxvar = xvar + add_suffix
    oyvar = yvar + add_suffix
    ozvar = zvar + add_suffix

    # Load input data
    with h5py.File(ifile, 'r') as f:

        lon = np.asarray(f[xvar][:], dtype=np.float64)
        lat = np.asarray(f[yvar][:], dtype=np.float64)
        h = np.asarray(f[zvar][:], dtype=np.float64)

        if alt is None:
            if rvar not in f:
                raise KeyError(
                    f'Range variable "{rvar}" not found. '
                    f'Provide range variable or use -a.'
                )

            R = np.asarray(f[rvar][:], dtype=np.float64)

    if lon.size == 0:
        return

    # Constant satellite altitude
    if alt is not None:
        A = alt * 1e3
        R = A - h

    # Project coordinates
    x, y = to_grid.transform(lon, lat)

    # Interpolate DEM gradients
    s_x = interp2d(Xd, Yd, Sx, x, y, order=1)
    s_y = interp2d(Xd, Yd, Sy, x, y, order=1)

    # Projection scale
    factors = proj_grid.get_factors(lon, lat)

    k_mer = np.asarray(factors.meridional_scale)
    k_par = np.asarray(factors.parallel_scale)

    # For polar stereographic these should be nearly identical
    k_proj = np.sqrt(k_mer * k_par)

    # Optional diagnostic
    scale_diff = np.abs(k_mer - k_par)

    print(
        '-> Projection scale:',
        'mean =', np.nanmean(k_proj),
        'max difference =', np.nanmax(scale_diff)
    )

    # Convert map-coordinate gradient to physical ground gradient
    s_x_ground = s_x * k_proj
    s_y_ground = s_y * k_proj

    # Slope magnitude
    slp_dem = np.arctan(np.hypot(s_x_ground, s_y_ground))
    slp = slp_dem.copy()

    if smax is not None:
        slp = np.minimum(slp, smax)

    # Grid aspect, clockwise from grid north
    grid_asp = np.mod(np.arctan2(s_x, s_y), 2.0*np.pi)

    # Explicit handling of flat terrain
    flat = np.hypot(s_x, s_y) < 1e-12
    grid_asp[flat] = 0.0

    # Convert grid aspect to true geographic azimuth
    step = max(abs(dx), abs(dy))

    x_test = x + step * np.sin(grid_asp)
    y_test = y + step * np.cos(grid_asp)

    lon_test, lat_test = to_geo.transform(x_test, y_test)

    asp_deg, _, _ = geod.inv(lon, lat, lon_test, lat_test)
    asp_deg = np.mod(asp_deg, 360.0)
    asp_deg[flat] = 0.0

    asp = np.deg2rad(asp_deg)

    # Ellipsoid curvature
    lat_rad = np.deg2rad(lat)
    sin_lat = np.sin(lat_rad)

    # Meridional radius of curvature
    M = a * (1.0 - e2) / (1.0 - e2*sin_lat**2)**1.5

    # Prime-vertical radius of curvature
    N = a / np.sqrt(1.0 - e2*sin_lat**2)

    # Directional normal radius of curvature
    r_curv = (M * N) / (N*np.cos(asp)**2 + M*np.sin(asp)**2)

    # Relocation distance
    d_slp = R * np.sin(slp)

    # Earth-curvature correction
    dR = d_slp**2 / (2.0 * r_curv)

    # Direct method
    if mode == 'DM':

        h_cor = R - (R / np.cos(slp) + dR)
        h_echo = h + h_cor

        OFILEd = {
            ozvar: h_echo,
            'dist_cor': d_slp
        }

    # Relocation method
    elif mode == 'RM':

        h_cor = R - (R * np.cos(slp) + dR)
        h_echo = h + h_cor

        # Relocate reflection point along true upslope direction
        lon_echo, lat_echo, _ = geod.fwd(lon, lat, asp_deg, d_slp)

        OFILEd = {
            oxvar: lon_echo,
            oyvar: lat_echo,
            ozvar: h_echo,
            'dist_cor': d_slp
        }

    # Output filename
    path, fname = os.path.split(ifile)
    name, ext = os.path.splitext(fname)
    suffix = '_DM' if mode == 'DM' else '_RM'

    if opath is not None:
        path = opath
        os.makedirs(path, exist_ok=True)

    if ending:
        ofile = os.path.join(path, name + suffix + ext)
    else:
        ofile = os.path.join(path, name + ext)

    # Save
    with h5py.File(ifile, 'a') as f:
        for key, values in OFILEd.items():
            if key in f:
                del f[key]
            f.create_dataset(key, data=values)

    # Move file if output path/name changed
    if os.path.abspath(ifile) != os.path.abspath(ofile):
        os.replace(ifile, ofile)

    print(
        'output file:', ofile,
        'Average correction:', np.around(np.nanmean(h - h_echo), 2), 'm',
        'Average relocation:', np.around(np.nanmean(d_slp), 2), 'm'
    )


# Run
if njobs == 1:

    print('running sequential code ...')

    for f in files:
        main(f)

else:

    print('running parallel code (%d jobs) ...' % njobs)

    from joblib import Parallel, delayed, parallel_backend

    with parallel_backend('loky', inner_max_num_threads=1):
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

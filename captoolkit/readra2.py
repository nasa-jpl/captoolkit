#!/usr/bin/env python
#
# Written by Johan Nilsson, Jet Propulsion Laboratory 2018
#
# Load libraries
import os
import sys
import h5py
import pyproj
import pandas as pd
import numpy as np
from gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates

#
# FYI!
# Something seems to be off with time
# We are outputing A-R there is a height alreadly avaliable

# Missing values in the NetCDF files
FillValue = 2147483647


def geotiffread(ifile,metaData):
    """Read raster from file."""

    file = gdal.Open(ifile, GA_ReadOnly)

    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    proj = src.ExportToWkt()

    Nx = file.RasterXSize
    Ny = file.RasterYSize

    trans = file.GetGeoTransform()

    dx = trans[1]
    dy = trans[5]

    if metaData == "A":

        xp = np.arange(Nx)
        yp = np.arange(Ny)

        (Xp, Yp) = np.meshgrid(xp,yp)

        X = trans[0] + (Xp+0.5)*trans[1] + (Yp+0.5)*trans[2]  #FIXME: bottleneck!
        Y = trans[3] + (Xp+0.5)*trans[4] + (Yp+0.5)*trans[5]

    if metaData == "P":

        xp = np.arange(Nx)
        yp = np.arange(Ny)

        (Xp, Yp) = np.meshgrid(xp,yp)

        X = trans[0] + Xp*trans[1] + Yp*trans[2]  #FIXME: bottleneck!
        Y = trans[3] + Xp*trans[4] + Yp*trans[5]

    band = file.GetRasterBand(1)

    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def bilinear2d(xd,yd,data,xq,yq, **kwargs):
    """Bilinear interpolation from grid."""

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0,:]
    yd = yd[:,0]

    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1]-xd[0]), (yd[1]-yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq*ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq*ones(xq.size)

    xp = (xq-xd[0])*(nx-1)/(xd[-1]-xd[0])
    yp = (yq-yd[0])*(ny-1)/(yd[-1]-yd[0])

    coord = np.vstack([yp,xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq


def fillnans(A):
    """ Function for interpolating nan-values."""
    
    inds = np.arange(A.shape[0])
    
    good = np.where(np.isfinite(A))
    
    f = interp1d(inds[good], A[good],bounds_error=False)
    
    B = np.where(np.isfinite(A),A,f(inds))
    
    return B


def wrapTo360(lon):
    """Function for wrapping longitude to 0 to 360 degrees."""
    positiveInput = (lon > 0)
    lon = np.mod(lon, 360)
    lon[(lon == 0) & positiveInput] = 360
    return lon


def wrapTo180(lon):
    """Function for wrapping longitude to -180 to 180 degrees."""
    q = (lon < -180) | (180 < lon)
    lon[q] = wrapTo360(lon[q] + 180) - 180
    return lon


def list_files(path, endswith='.txt'):
    """List files recursively."""
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if f.endswith(endswith)]

def track_type(time, lat, tmax=1):
    """
        Determines ascending and descending tracks.
        Defines unique tracks as segments with time breaks > tmax,
        and tests whether lat increases or decreases w/time.
        """
    
    # Generate track segment
    tracks = np.zeros(lat.shape)
    
    # Set values for segment
    tracks[0:np.argmax(np.abs(lat))] = 1
    
    # Output index array
    i_asc = np.zeros(tracks.shape, dtype=bool)
    
    # Loop trough individual tracks
    for track in np.unique(tracks):
        
        # Get all points from an individual track
        i_track, = np.where(track == tracks)
        
        # Test tracks length
        if len(i_track) < 2:
            continue
        
        # Test if lat increases (asc) or decreases (des) w/time
        i_min = time[i_track].argmin()
        i_max = time[i_track].argmax()
        lat_diff = lat[i_track][i_max] - lat[i_track][i_min]
        
        # Determine track type
        if lat_diff > 0:
            i_asc[i_track] = True

    # Output index vector's
    return i_asc, np.invert(i_asc)


Rootdir = str(sys.argv[1])  # input dir
outdir  = sys.argv[2]       # output dir
fmask   = sys.argv[3]       # geotiff file with mask
proj    = str(sys.argv[4])  # epsg number
meta    = sys.argv[5]       # "A" or "P"
index   = int(sys.argv[6])  # mission reference (300=CS2,100=ICE etc)
njobs   = int(sys.argv[7])  # number of parallel jobs

# Generate file list
files = list_files(Rootdir, endswith='.nc')

print 'input dir:', Rootdir
print 'output dir:', outdir
print 'mask file:', fmask
print 'epsg num:', proj
print 'metadata:', meta
print 'njobs:', njobs
print '# files:', len(files)

# Track counter
k_iter = 0

# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")

# Make pyproj format
projection = '+init=EPSG:' + proj

# Projection - prediction grid
projGrd = pyproj.Proj(projection)

# If mask avaliable
if fmask != 'None':

    print 'Reading raster mask ....'
    
    # Read in masking grid
    (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(fmask, meta)

    print 'Reading raster mask done'

def main(file):
    
    # Access global variable
    global k_iter
    
    # Determine if the file is empty
    if os.stat(file).st_size == 0:
        return

    # Read netcdf file
    with h5py.File(file,'r') as data:

        # Load satellite parameters
        lat       = data['lat_20'][:] * 1e-6                           # Latitude (deg)
        lon       = data['lon_20'][:] * 1e-6                           # Longitude (deg)
        t_sec     = data['time_20'][:]                                 # Time (secs since 2000)
        r_ice1    = data['range_ice1_20_ku'][:] * 1e-4                 # Range ICE-1 retracker (m)
        a_sat     = data['alt_20'][:] * 1e-4                           # Altitude of satellite (m)
        bs_ice1   = data['sig0_ice1_20_ku'][:] * 1e-2                  # Backscatter of ICE-1 retracker (dB)
        lew_ice2  = data['width_leading_edge_ice2_20_ku'][:] * 1e-3    # Leading edge width from ICE-2 (m)
        tes_ice2  = data['slope_first_trailing_edge_ice2_20_ku'][:]    # Trailing edge slope from ICE-2 (1/m)
        qual_ice1 = data['retracking_ice1_qual_20_ku'][:]              # Retracking quality flag for ice 1
        h_ellip   = data['elevation_ice1_20_ku'][:] * 1e-2             # Elevation 20Hz ice 1
        h_dry     = data['mod_dry_tropo_cor_reanalysis_20'][:] * 1e-4  # Dry tropo cor
        h_wet     = data['mod_wet_tropo_cor_reanalysis_20'][:] * 1e-4  # Wet tropo cor
        h_geo_01  = data['pole_tide_01'][:] * 1e-4                     # Pole tide
        h_sol_01  = data['solid_earth_tide_01'][:] * 1e-4              # Solid tide
        h_ion_01  = data['iono_cor_gim_01_ku'][:] * 1e-4               # Ionospheric correction

        h_tide_eq_01 = data['ocean_tide_eq_01'][:] * 1e-4              # Tide-related corrections (below)
        h_tide_noneq_01 = data['ocean_tide_non_eq_01'][:] * 1e-4
        h_tide_sol1_01 = data['ocean_tide_sol1_01'][:] * 1e-4
        h_tide_sol2_01 = data['ocean_tide_sol2_01'][:] * 1e-4

    # Missing values -> NaNs
    h_dry[h_dry==FillValue] = np.nan
    h_wet[h_wet==FillValue] = np.nan
    h_geo_01[h_geo_01==FillValue] = np.nan
    h_sol_01[h_sol_01==FillValue] = np.nan
    h_ion_01[h_ion_01==FillValue] = np.nan
    h_tide_eq_01[h_tide_eq_01==FillValue] = np.nan
    h_tide_noneq_01[h_tide_noneq_01==FillValue] = np.nan
    h_tide_sol1_01[h_tide_sol1_01==FillValue] = np.nan
    h_tide_sol2_01[h_tide_sol2_01==FillValue] = np.nan

    # Create 20 Hz containers
    h_ion = np.empty((0,1))
    h_sol = np.empty((0,1))
    h_geo = np.empty((0,1))
    h_tide_eq = np.empty((0,1))
    h_tide_noneq = np.empty((0,1))
    h_tide_sol1 = np.empty((0,1))
    h_tide_sol2 = np.empty((0,1))

    # Make corrections to 20Hz
    for i in xrange(len(h_ion_01)):
        
        # Stack the correctons
        h_ion = np.vstack((h_ion, np.ones((20,1)) * h_ion_01[i]))
        h_geo = np.vstack((h_geo, np.ones((20,1)) * h_geo_01[i]))
        h_sol = np.vstack((h_sol, np.ones((20,1)) * h_sol_01[i]))
        h_tide_eq = np.vstack((h_tide_eq, np.ones((20,1)) * h_tide_eq_01[i]))
        h_tide_noneq = np.vstack((h_tide_noneq, np.ones((20,1)) * h_tide_noneq_01[i]))
        h_tide_sol1 = np.vstack((h_tide_sol1, np.ones((20,1)) * h_tide_sol1_01[i]))
        h_tide_sol2 = np.vstack((h_tide_sol2, np.ones((20,1)) * h_tide_sol2_01[i]))

    # Change dimensions
    h_ion = h_ion.reshape(lat.shape)
    h_geo = h_geo.reshape(lat.shape)
    h_sol = h_sol.reshape(lat.shape)
    h_tide_eq = h_tide_eq.reshape(lat.shape)
    h_tide_noneq = h_tide_noneq.reshape(lat.shape)
    h_tide_sol1 = h_tide_sol1.reshape(lat.shape)
    h_tide_sol2 = h_tide_sol2.reshape(lat.shape)

    # Wrap longitude to -180 to 180 degrees
    lon = wrapTo180(lon)

    # If mask avaliable
    if fmask != 'None':
        
        # Determine projection
        if proj != '4326':
            
            # Reproject coordinates
            (x, y) = pyproj.transform(projGeo, projGrd, lon, lat)

        else:
        
            # Keep lon/lat
            x, y = lon, lat
        
        # Interpolation of grid to points for masking
        Ii = bilinear2d(Xm, Ym, Zm, x.T, y.T, order=1)
        
        # Set all NaN's to zero
        Ii[np.isnan(Ii)] = 0
        
        # Get quality flag - keep valid records
        I_flag = (qual_ice1 == 0) & (Ii == 1)

    else:

        # Get quality flag - keep valid records
        I_flag = (qual_ice1 == 0)

    # Only keep valid records
    lon, lat, t_sec, r_ice1, a_sat, bs_ice1, lew_ice2, tes_ice2, h_ellip, \
            h_ion, h_geo, h_sol, h_dry, h_wet, h_tide_eq, h_tide_noneq, h_tide_sol1, h_tide_sol2 = \
                lon[I_flag], lat[I_flag], t_sec[I_flag], r_ice1[I_flag], a_sat[I_flag], \
                bs_ice1[I_flag], lew_ice2[I_flag], tes_ice2[I_flag], h_ellip[I_flag], \
                h_ion[I_flag], h_geo[I_flag], h_sol[I_flag], h_dry[I_flag], h_wet[I_flag], \
                h_tide_eq[I_flag], h_tide_noneq[I_flag], h_tide_sol1[I_flag], h_tide_sol2[I_flag]

    # If file is empty - skip file
    if len(h_ellip) == 0:
        print 'No data here!'
        return

    # Compute correct time - add back year 2000 in secs
    t_sec += 2000 * 365.25 * 24 * 3600.

    # Compute time in decimal years 
    t_year = t_sec / (365.25 * 24 * 3600.)

    # Compute time since 1970 - remove year 1970 in secs
    t_sec -= 1970 * 365.25 * 24 * 3600.

    # Separate tracks into asc/des orbits
    (i_asc, i_des) = track_type(t_sec, lat)
    
    # Create orbit number container
    orb = np.zeros(lat.shape)
    orb_type = np.zeros(lat.shape)

    # Set orbit numbers
    if len(lat[i_asc]) > 0:
    
        # Create independent track references
        orbit_num = np.char.add(str(index), str(k_iter)).astype('int')
        
        # Set vector to number
        orb[i_asc] = orbit_num
        
        # Set orbit type
        orb_type[i_asc] = 0
        
        # Increase counter
        k_iter += 1
    
    # Set orbit numbers
    if len(lat[i_des]) > 0:
        
        # Create independent track references
        orbit_num = np.char.add(str(index), str(k_iter)).astype('int')
        
        # Set vector to number
        orb[i_des] = orbit_num
        
        # Set orbit type
        orb_type[i_des] = 1
        
        # Increase counter
        k_iter += 1

    # Sum all corrections
    h_cor = h_ion + h_dry + h_wet + h_geo + h_sol

    # Correct range
    r_ice1_cor = (r_ice1 + h_cor)

    # Compute surface elevation
    h_ice1 = a_sat - r_ice1_cor
    
    # Test LeW for undefined numbers
    if np.any(lew_ice2[lew_ice2 == 0]):
        
        # Set values to NaN
        lew_ice2[lew_ice2 == 0] = np.nan
        
        # Interpolate any nan-values
        lew_ice2 = fillnans(lew_ice2)

    # Test TeS for undefined numbers
    if np.any(tes_ice2[tes_ice2 == 0]):
        
        # Set to NaN
        tes_ice2[tes_ice2 == 0] = np.nan
        
        # Interpolate any nan-values
        tes_ice2 = fillnans(tes_ice2)

    # Current file
    iFile = np.column_stack((
              orb, lat, lon, t_sec, h_ice1, t_year,
              bs_ice1, lew_ice2, tes_ice2, r_ice1_cor,
              h_ion, h_dry, h_wet, h_geo, h_sol, orb_type,
              h_tide_eq, h_tide_noneq, h_tide_sol1, h_tide_sol2))

    # Create varibales names
    fields = ['orbit', 'lat', 'lon', 't_sec', 'h_cor', 't_year',
              'bs', 'lew', 'tes', 'range',
              'h_ion', 'h_dry', 'h_wet', 'h_geo', 'h_sol', 'orb_type', 
              'h_tide_eq', 'h_tide_noneq', 'h_tide_sol1', 'h_tide_sol2']

    # Save ascending file
    if len(lat[i_asc]) > 0:
    
        # Create file ending
        str_orb = '_READ_A'
        
        # Change path/name of read file
        name, ext = os.path.splitext(os.path.basename(file))
        ofile = os.path.join(outdir, name + str_orb + ext)
        ofile = ofile.replace('.nc', '.h5')
        
        # Write to file
        with h5py.File(ofile, 'w') as f:
            [f.create_dataset(k, data=d) for k, d in zip(fields, iFile[i_asc].T)]

        # What file are we reading
        print ofile, len(h_ice1)

    # Save descending file
    if len(lat[i_des]) > 0:
    
        # Create file ending
        str_orb = '_READ_D'
        
        # Change path/name of read file
        name, ext = os.path.splitext(os.path.basename(file))
        ofile = os.path.join(outdir, name + str_orb + ext)
        ofile = ofile.replace('.nc', '.h5')
        
        # Write to file
        with h5py.File(ofile, 'w') as f:
            [f.create_dataset(k, data=d) for k, d in zip(fields, iFile[i_des].T)]

        # What file are we reading
        print ofile, len(h_ice1)


if njobs == 1:
    print 'running sequential code...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs)...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

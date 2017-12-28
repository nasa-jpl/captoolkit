#!/usr/bin/env python

# Usage:
#
#   python readenvi.py /mnt/devon-r0/nilssonj/Altimetry/EnviSat/Antarctica/cycles/ /mnt/devon-r0/shared_data/envisat/grounded/ /mnt/devon-r0/shared_data/masks/ANT_groundedice_240m.tif 3031 A 400 16
#
# Notes:
#   
#   Use 'shared_data/data/Masks/ANT_floatingice_240m.tif' for ice shelves.
#   Walks through an entire directory w/subfolders recursively.
#    

#
# Change log:
# Included A/D indexing (JN 2017/09/08)
# Removed extra columns (JN 2017/09/08)
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

save_to_hdf5 = True


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
outdir = sys.argv[2]        # output dir
fmask = sys.argv[3]         # geotiff file with mask
proj  = str(sys.argv[4])    # epsg number
meta  = sys.argv[5]         # "A" or "P"
index = int(sys.argv[6])    # mission reference (300=CS2,100=ICE etc)
njobs = int(sys.argv[7])    # number of parallel jobs

# Generate file list
files = list_files(Rootdir, endswith='.txt')

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

    # Read in masking grid
    (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(fmask, meta)

def main(file):
    
    # Access global variable
    global k_iter
    
    # Determine if the file is empty
    if os.stat(file).st_size == 0:
        return
    
    # Read CSV file
    data = pd.read_csv(file, engine="c", header=None, delim_whitespace=True)
    
    # Convert to numpy array and floats
    data = np.float64(pd.DataFrame.as_matrix(data))

    # Wrap longitude to -180 to 180 degrees
    data[:, 1] = wrapTo180(data[:, 1])
    
    # Get quality flag - keep valid records
    I_flag = (data[:, 12] == 0) & (data[:, 15] == 0)
    
    # Only keep valid records
    data = data[I_flag, :]
    
    # If mask avaliable
    if fmask != 'None':
        
        # Reproject coordinates
        (x, y) = pyproj.transform(projGeo, projGrd, data[:, 1], data[:, 0])

        # Interpolation of grid to points for masking
        Ii = bilinear2d(Xm, Ym, Zm, x.T, y.T, order=1)
    
        # Set all NaN's to zero
        Ii[np.isnan(Ii)] = 0
    
        # Convert to boolean
        Im = Ii == 1
    
        # Keep only data inside mask
        data = data[Im, :]
    
    # If file is empty - skip file
    if len(data) == 0:
        return
    
    # Load satellite parameters
    cyc      = data[:, 3]            # Cycle number
    rel      = data[:, 4]            # Relative orbit
    lat      = data[:, 0]            # Latitude (deg)
    lon      = data[:, 1]            # Longitude (deg)
    t_sec    = data[:, 2]            # Time (secs since 2000)
    r_ice1   = data[:,13] * 1e-3     # Range ICE-1 retracker (m)
    a_sat    = data[:, 5] * 1e-3     # Altitude of satellite (m)
    bs_ice1  = data[:,14] * 1e-2     # Backscatter of ICE-1 retracker (dB)
    lew_ice2 = data[:,19] * 1e-3     # Leading edge width from ICE-2 (m)
    tes_ice2 = data[:,20]            # Trailing edge slope fram ICE-2 (1/m)
    #orb_type = data[:,22]            # Orbit type A(0)/D(1)

    # Load geophysical corrections - (mm -> m)
    h_ion = data[:, 6] * 1e-3        # Ionospheric cor
    h_dry = data[:, 7] * 1e-3        # Dry tropo cor
    h_wet = data[:, 8] * 1e-3        # Wet tropo cor
    h_geo = data[:, 9] * 1e-3        # Pole tide
    h_sol = data[:,10] * 1e-3        # Solid tide
    
    # Compute correct time - add back year 2000 in secs
    t_sec += 2000 * 365.25 * 24 * 3600.

    # Compute time in decimal years 
    t_year = t_sec / (365.25 * 24 * 3600.)

    # Compute time since 1970 - remove year 1970 in secs
    t_sec -= 1970 * 365.25 * 24 * 3600.

    # Sum all corrections
    h_cor = h_ion + h_dry + h_wet + h_geo + h_sol
    
    # Correct range
    r_ice1_cor = (r_ice1 + h_cor)
    
    # Compute surface elevation
    h_ice1 = a_sat - r_ice1_cor

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

    # Current file
    iFile = np.column_stack((
              orb, lat, lon, t_sec, h_ice1, t_year,
              bs_ice1, lew_ice2, tes_ice2, r_ice1_cor,
              h_ion, h_dry, h_wet, h_geo, h_sol, orb_type))

    # Create varibales names
    fields = ['orbit', 'lat', 'lon', 't_sec', 'h_cor', 't_year',
              'bs', 'lew', 'tes', 'r_ice1_cor',
              'h_ion', 'h_dry', 'h_wet', 'h_geo', 'h_sol', 'orb_type']

    # Save ascending file
    if len(lat[i_asc]) > 0:
        
        # Create file ending
        str_orb = '_READ_A'
        
        # Change path/name of read file
        name, ext = os.path.splitext(os.path.basename(file))
        ofile = os.path.join(outdir, name + str_orb + ext)
        ofile = ofile.replace('.txt', '.h5')
        
        # Write to file
        with h5py.File(ofile, 'w') as f:
            [f.create_dataset(k, data=d) for k, d in zip(fields, iFile[i_asc].T)]
        
        # What file are we reading
        print ofile
    
    # Save descending file
    if len(lat[i_des]) > 0:
        
        # Create file ending
        str_orb = '_READ_D'
        
        # Change path/name of read file
        name, ext = os.path.splitext(os.path.basename(file))
        ofile = os.path.join(outdir, name + str_orb + ext)
        ofile = ofile.replace('.txt', '.h5')
        
        # Write to file
        with h5py.File(ofile, 'w') as f:
            [f.create_dataset(k, data=d) for k, d in zip(fields, iFile[i_des].T)]
        
        # What file are we reading
        print ofile

if njobs == 1:
    print 'running sequential code...'
    [main(f) for f in files]

else:
    print 'running parallel code (%d jobs)...' % njobs
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

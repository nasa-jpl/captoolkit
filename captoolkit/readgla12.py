#!/usr/bin/env python
"""
    Reads GLA12 Release 634 HDF5.

    Reads several files in parallel if njobs > 1 is specified.

    Extracts a subset of the data based on a mask.tif file.

    Example:

    python readgla.py /mnt/devon-r0/shared_data/icesat/GLAH12.034/ /mnt/devon-r0/shared_data/icesat/grounded/ /mnt/devon-r0/shared_data/masks/ANT_groundedice_240m.tif 3031 A 600 1

    See full GLA12 parameters at:

    http://nsidc.org/data/docs/daac/glas_altimetry/data-dictionary-glah12.html

    Notes:

    For previous releases the path of some fields have changed!

    Corrections applied by default (i.e. data come corrected):

    instrument corrections              - was applied
    atmospheric delays (wet/dry tropo)  - was applied
    tides and load                      - was applied
    GC offset                           - was applied

    saturation (d_satElevCorr)          - was NOT applied [1]
    inter-campaign bias                 - was NOT applied

    [1] If it is invalid, then the elevation should not be used.
    The saturation correction flag (sat_corr_flg) is an important
    flag to understand the possible quality of the elevation data.

    To REMOVE the tide and load cor, and APPLY saturation cor:

    elev_retide = d_elev + d_ocElv + d_ldElv + d_satElevCorr

"""

import os
import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
try:
    from gdalconst import *
except ImportError as e:
    from osgeo.gdalconst import *
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates


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


def wrap_to_180(lon):
    """Wrapps longitude to -180 to 180 degrees."""
    lon[lon>180] -= 360.
    return lon


def list_files(path, endswith='.h5'):
    """ List files in dir recursively. """
    return [os.path.join(dpath, f)
            for dpath, dnames, fnames in os.walk(path)
            for f in fnames if f.endswith(endswith)]

def track_type(time, lat):
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


indir = sys.argv[1]       # input dir
outdir = sys.argv[2]      # output dir
fmask = sys.argv[3]       # geotiff file with mask
proj  = str(sys.argv[4])  # epsg number
meta  = sys.argv[5]       # "A" or "P"
index = int(sys.argv[6])  # mission index
njobs = int(sys.argv[7])  # number of parallel jobs

# Generate file list
files = list_files(indir, endswith='.H5')

print(('input dir:', indir))
print(('output dir:', outdir))
print(('mask file:', fmask))
print(('epsg num:', proj))
print(('metadata:', meta))
print(('njobs:', njobs))
print(('# files:', len(files)))


# Projection - unprojected lat/lon
projGeo = pyproj.Proj("+init=EPSG:4326")

# Make pyproj format
projection = '+init=EPSG:' + proj

# Projection - prediction grid
projGrd = pyproj.Proj(projection)

iter = 1
index = 600

# Test for mask
if fmask != 'None':

    # Read in masking grid
    (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(fmask, meta)

def main(fname):

    print(('readg:', fname, '...'))

    global iter

    f = h5py.File(fname)

    d = {}  # Dictionary for input fields

    d['t_sec'] = f['Data_40HZ/Time/d_UTCTime_40'] # [secs since 2000-01-01 12:00:00 UTC]

    d['lat'] = f['Data_40HZ/Geolocation/d_lat']  # [deg]
    d['lon'] = f['Data_40HZ/Geolocation/d_lon']  # [deg]

    d['num_pk'] = f['Data_40HZ/Waveform/i_numPk']  # Num Peaks found in the Return
    d['gain'] = f['Data_40HZ/Waveform/i_gval_rcv']  # counts [unitless]
    d['rec_nrg'] = f['Data_40HZ/Reflectivity/d_RecNrgAll']  # [joules]
    d['tx_nrg'] = f['Data_40HZ/Transmit_Energy/d_TxNrg']  # [joules]

    d['h_sat'] = f['Data_40HZ/Elevation_Corrections/d_satElevCorr']  # saturation cor [m]
    d['h_gc'] = f['Data_40HZ/Elevation_Corrections/d_GmC']  # GC-offset cor [m]
    d['h_dry'] = f['Data_40HZ/Elevation_Corrections/d_dTrop']  # dry tropo [m]
    d['h_wet'] = f['Data_40HZ/Elevation_Corrections/d_wTrop']  # wet tropo [m]

    d['h_sol'] = f['Data_40HZ/Geophysical/d_erElv']  # solid tide [m]
    d['h_geo'] = f['Data_40HZ/Geophysical/d_poTide']  # geoc pole tide [m]
    d['h_equi'] = f['Data_40HZ/Geophysical/d_eqElv']  # equilib tide [m]
    d['h_ellip'] = f['Data_40HZ/Geophysical/d_deltaEllip']  # h_TP - h_WGS84 [m]
    d['h_tide'] = f['Data_40HZ/Geophysical/d_ocElv']  # ocean tide [m]
    d['h_load'] = f['Data_40HZ/Geophysical/d_ldElv']  # load tide [m]

    d['h_cor'] = f['Data_40HZ/Elevation_Surfaces/d_elev']  # corrected height [m]
    d['misfit'] = f['Data_40HZ/Elevation_Surfaces/d_IceSVar']  # gaussian misfit [volts] [2]

    d['rec_ndx'] = f['Data_40HZ/Time/i_rec_ndx']  # record index
    d['shot_count'] = f['Data_40HZ/Time/i_shot_count']  # shot index within record

    # Elevation quality flag: 0=valid, 1=not_valid
    d['use_flg'] = f['Data_40HZ/Quality/elev_use_flg']

    # Cloud contamination flag: 0=false, 1=true
    d['cloud_flg'] = f['Data_40HZ/Elevation_Flags/elv_cloud_flg']

    # Attitude quality flag: 0=good, 50=warning, 100=bad, 127=not_valid
    d['att_flg'] = f['Data_40HZ/Quality/sigma_att_flg']

    # Saturation Correction Flag:
    # 0=not_saturated, 1=inconsequential, 2=applicable 3=not_computed 4=not_applicable
    d['sat_flg'] = f['Data_40HZ/Quality/sat_corr_flg']

    # 1Hz Track
    track_01Hz = f['Data_1HZ/Geolocation/i_track'][:]

    # Get unique track numbers
    track_id = np.unique(track_01Hz)

    # Initialize vector
    track_40Hz = np.empty((0,1), dtype='int')

    # Construct 40 Hz track vector - IMPROVE! SLOW WAY OF DOING IT
    for i in range(len(track_01Hz)):

        # Create 40 Hz vector
        track_40Hz = np.vstack((track_40Hz, np.ones((40,1),dtype='int') * track_01Hz[i]))

    # Construct cycle vector
    #cycle = int(fname[fname.rfind('/') + 1:].split('_')[3]) * np.ones(track_40Hz.shape)

    # Induvidual track identifier
    #d['orbit'] = np.char.add(cycle.astype('int').astype('str'), track_40Hz.astype('int').astype('str')).astype('int')

    '''
        [2] For postprocessing: The RMS error converged to about 0.25 m after
        removing the data with the 5% highest waveform misfits in each campaign, so we
        adopted that as a data-editing threshold, retaining 95% of the original data.
        Also, filter out cloud-contaminated points using the 'cloud_flg' param.
    '''

    # Wrap longitude to -180/180 degrees
    d['lon'] = wrap_to_180(d['lon'][:])

    # Reproject coordinates
    lon, lat = d['lon'][:], d['lat'][:]

    # Converte to Stereographical coordinates
    (x, y) = pyproj.transform(projGeo, projGrd, lon, lat)

    # Test for mask
    if fmask != 'None':

        # Interpolation of grid to points for masking
        Ii = bilinear2d(Xm, Ym, Zm, x.T, y.T, order=1)

        # Set all NaN's to zero
        Ii[np.isnan(Ii)] = 0

        # Convert to boolean
        mask = Ii == 1

    else:

        # Select all data
        mask = np.ones(lat.shape, dtype='bool')

    # Parameters for selecting valid pts
    h_cor = d['h_cor'][:]
    h_sat = d['h_sat'][:]
    use_flg = d['use_flg'][:]
    sat_flg = d['sat_flg'][:]
    att_flg = d['att_flg'][:]
    num_pk = d['num_pk'][:]

    # Get index of valid pts
    idx, = np.where(
        (mask == 1) &
        (np.abs(h_cor) < 1e10) &
        (np.abs(h_sat) < 1e10) &
        (np.abs(lat) <= 90) &
        (np.abs(lon) <= 180) &
        (use_flg == 0) &
        (sat_flg <= 2) &
        (att_flg == 0) &
        (num_pk == 1))

     # Check if no valid pts
    if len(idx) == 0:
        print(('no valid pts:', fname))
        return

    # Keep only valid pts (and load to memory)
    for k in list(d.keys()):

        # Edit all the fields
        d[k] = d[k][:][idx]

    # Unapply tides (retide)
    d['h_cor'] += d['h_tide'] + d['h_load']

    # Apply saturation cor
    d['h_cor'] += d['h_sat']

    # Convert ellipsoid: h_TP -> h_WGS84
    d['h_cor'] -= d['h_ellip']

    #FIXME: THIS IS NOT ORBIT NUMBER (ONE ID FOR EACH TRACK)!!!
    # Combine rec_ndx and shot_count to uniquely identify each GLAS laser shot
    #d['orbit'] = np.char.add(d['rec_ndx'].astype('str'),
    #        d['shot_count'].astype('str')).astype('int')

    # Compute correct time - add back 'year 2000 + 12 hours' in secs
    d['t_sec'] += (2000 * 365.25 * 24 * 3600.) + (12 * 3600.)

    # Compute time in decimal years
    d['t_year'] = d['t_sec'] / (365.25 * 24 * 3600.)

    # Compute time since 1970 - remove year 1970 in secs
    d['t_sec'] -= 1970 * 365.25 * 24 * 3600.

    # Change path and/or name of read file
    name, ext = os.path.splitext(os.path.basename(fname))

    # Clip track vector
    tracks_40Hz = track_40Hz[idx]

    # Compute unique tracks
    tracks = np.unique(tracks_40Hz)

    # Create orbit array
    d['orbit'] = np.ones(d['lat'][:].shape) * np.nan

    # Select fields to save
    out = ['orbit',
           't_sec',
           't_year',
           'lon',
           'lat',
           'h_cor',
           'h_dry',
           'h_ellip',
           'h_equi',
           'h_gc',
           'h_geo',
           'h_sat',
           'h_sol',
           'h_wet',
           'gain',
           'misfit',
           'tx_nrg',
           'rec_nrg',
           'cloud_flg',]

    # Loop through tracks
    for i in range(len(tracks)):

        # Get index of induvidual tracks
        ind = (tracks_40Hz == tracks[i]).reshape(d['lat'][:].shape)

        # Set track datum identifier for each track
        (dec,year)=np.modf(d['t_year'][ind][0])
        month = np.round(dec * 12, decimals=0)
        day   = np.round(dec * 365.25, decimals=0)

        # Datum string
        date = str(int(year))+'_'+str(int(month)).zfill(2)+'_'+str(int(day)).zfill(3)

        # Separate tracks
        (i_asc, i_des) = track_type(d['t_sec'][ind], d['lat'][ind])

        # Save ascending track
        if len(d['lat'][ind][i_asc]) > 0:

            # Psudo orbit number generation
            d['orbit'][ind] = np.char.add(str(index), str(iter)).astype('int')

            # Orbit type identifier
            str_orb = 'READ_A'

            # Track number string
            str_trknum = '_'+str(int(iter)).zfill(6)+'_'

            # Fullname of output file
            outfile = os.path.join(outdir, name[0:7] + date + str_trknum + str_orb + ext)

            # Save data
            with h5py.File(outfile, 'w') as fout:
                [fout.create_dataset(k, data=d[k][ind][i_asc]) for k in out]

            # Update counter
            iter += 1
            print(('output file:', outfile))

        # Save descending track
        if len(d['lat'][ind][i_des]) > 0:

            # Psudo orbit number generation
            d['orbit'][ind] = np.char.add(str(index), str(iter)).astype('int')

            # Orbit type identifier
            str_orb = 'READ_D'

            # Track number string
            str_trknum = '_'+str(int(iter)).zfill(6)+'_'

            # Fullname of output file
            outfile = os.path.join(outdir, name[0:7] + date + str_trknum + str_orb + ext)

            # Save data
            with h5py.File(outfile, 'w') as fout:
                [fout.create_dataset(k, data=d[k][ind][i_des]) for k in out]

            # Update counter
            iter += 1

            print(('output file:', outfile))

    f.close()

if njobs == 1:
    print('running sequential code ...')
    [main(f) for f in files]

else:
    print(('running parallel code (%d jobs) ...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

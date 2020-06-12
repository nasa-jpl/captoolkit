#!/usr/bin/env python
"""
Reading of ICESat-2 ATL06 data into single track format

Reads the ATL06 file format and outputs separate files for each beam pair
and provides information in the file name if the orbit is ascending or
descending.

The observations output if they flagged valid by the "atl06_quality_summary"
flag, pass the segmentation filter (2 m tolerance) and are located inside the
ROI, defined by either a mask or a bounding box.

Mask can be passed as either a ".tif" file or ".h5" file containing 0 (off)
or 1 (on) masking values. If hdf5 used please use (X Y Z) as variable names,
as they are
hardcoded.

User can provide a blacklist (".txt" file) of ATL06 file names that should
not be processed.

Notes:
    The unique mission index ('-i') is used to separate different mission to
    avoid tracks having the same orbit number. The number provided is
    appended to the front of the current orbit number:
    orbit_new = str(unique) + str(orbit_i) = int('100') + int('1100')
    = int('1001100')

    Providing mission index and reading the files is parallel will
    unfortunately not provide a unique mission index. However, if applying a
    mission index is of interest the parallel option is significantly
    faster.

Example:

    python readatl06.py ./input/path/*.h5 ./output/path/*.h5 -f mask.tif -p
    3031 -i 100 -g 10 11 12 -n 1

"""
import os
import argparse

import h5py
import numpy as np
import pandas as pd
import pyproj
from astropy.time import Time
from gdalconst import GA_ReadOnly
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates


def segDifferenceFilter(dh_fit_dx, h_li, tol=2):
    """ Coded by Ben Smith University of Washington """
    dAT = 20.0

    if h_li.shape[0] < 3:
        mask = np.ones_like(h_li, dtype=bool)

        return mask

    EPplus = h_li + dAT * dh_fit_dx
    EPminus = h_li - dAT * dh_fit_dx

    segDiff = np.zeros_like(h_li)
    segDiff[0:-1] = np.abs(EPplus[0:-1] - h_li[1:])
    segDiff[1:] = np.maximum(segDiff[1:], np.abs(h_li[0:-1] - EPminus[1:]))

    mask = segDiff < tol

    return mask


def gps2dyr(time):
    """ Converte from GPS time to decimal years. """
    time = Time(time, format="gps")
    time = Time(time, format="decimalyear").value

    return time


def list_files(path, endswith=".h5"):
    """ List files in dir recursively."""

    return [
        os.path.join(dpath, f)

        for dpath, dnames, fnames in os.walk(path)

        for f in fnames

        if f.endswith(endswith)
    ]


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + proj1)
    proj2 = pyproj.Proj("+init=EPSG:" + proj2)
    return pyproj.transform(proj1, proj2, x, y)


def track_type(time, lat, tmax=1):
    """ Determines ascending and descending tracks.

    Defines unique tracks as segments with time breaks > tmax,
    and tests whether lat increases or decreases w/time.
    """

    # Generate track segment
    tracks = np.zeros(lat.shape)

    # Set values for segment
    tracks[0: np.argmax(np.abs(lat))] = 1

    # Output index array
    i_asc = np.zeros(tracks.shape, dtype=bool)

    # Loop trough individual tracks

    for track in np.unique(tracks):

        # Get all points from an individual track
        (i_track,) = np.where(track == tracks)

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


def geotiffread(ifile, metaData):
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

        (Xp, Yp) = np.meshgrid(xp, yp)

        X = (
            trans[0] + (Xp + 0.5) * trans[1] + (Yp + 0.5) * trans[2]
        )  # FIXME: bottleneck!
        Y = trans[3] + (Xp + 0.5) * trans[4] + (Yp + 0.5) * trans[5]

    if metaData == "P":

        xp = np.arange(Nx)
        yp = np.arange(Ny)

        (Xp, Yp) = np.meshgrid(xp, yp)

        X = trans[0] + Xp * trans[1] + Yp * trans[2]  # FIXME: bottleneck!
        Y = trans[3] + Xp * trans[4] + Yp * trans[5]

    band = file.GetRasterBand(1)

    Z = band.ReadAsArray()

    dx = np.abs(dx)
    dy = np.abs(dy)

    return X, Y, Z, dx, dy, proj


def bilinear2d(xd, yd, data, xq, yq, **kwargs):
    """Bilinear interpolation from grid."""

    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)

    xd = xd[0, :]
    yd = yd[:, 0]

    nx, ny = xd.size, yd.size
    # (x_step, y_step) = (xd[1] - xd[0]), (yd[1] - yd[0])

    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq * np.ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq * np.ones(xq.size)

    xp = (xq - xd[0]) * (nx - 1) / (xd[-1] - xd[0])
    yp = (yq - yd[0]) * (ny - 1) / (yd[-1] - yd[0])

    coord = np.vstack([yp, xp])

    zq = map_coordinates(data, coord, **kwargs)

    return zq


# Output description of solution
description = "Program for reading ICESat ATL06 data."

# Define command-line arguments
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "ifiles",
    metavar="ifile",
    type=str,
    nargs="+",
    help="path for ifile(s) to read (.h5).",
)
parser.add_argument(
    "ofiles",
    metavar="ofile",
    type=str,
    nargs="+",
    help="path for ofile(s) to save (.h5).",
)
parser.add_argument(
    "-f",
    metavar=("fmask"),
    dest="fmask",
    type=str,
    nargs=1,
    help="name of raster file mask ('.tif' or 'hdf5')",
    default=[None],
)
parser.add_argument(
    "-l",
    metavar=("efile"),
    dest="efile",
    type=str,
    nargs=1,
    help="blacklist file if available ('.txt')",
    default=[None],
)
parser.add_argument(
    "-b",
    metavar=("w", "e", "s", "n"),
    dest="bbox",
    type=float,
    nargs=4,
    help=("bounding box for geographical region (deg)"),
    default=[None],
)
parser.add_argument(
    "-n",
    metavar=("njobs"),
    dest="njobs",
    type=int,
    nargs=1,
    help="number of cores to use for parallel processing",
    default=[1],
)
parser.add_argument(
    "-p",
    metavar=("epsg"),
    dest="proj",
    type=str,
    nargs=1,
    help=("proj for mask: EPSG number (AnIS=3031, GrIS=3413)"),
    default=["3031"],
)
parser.add_argument(
    "-i",
    metavar=("index"),
    dest="index",
    type=int,
    nargs=1,
    help=("unique mission index (appended to original)"),
    default=[None],
)
parser.add_argument(
    "-g",
    metavar=("granual"),
    dest="granual",
    type=str,
    nargs="+",
    help=("select specific granuals"),
    default=None,
)

# Parser argument to variable
args = parser.parse_args()

# Read input from terminal
ifiles = args.ifiles
opath = args.ofiles[0]
bbox = args.bbox
njobs = args.njobs[0]
fmask = args.fmask[0]
proj = args.proj[0]
index = args.index[0]
gran = args.granual
bfile = args.efile[0]

# Beam names
group = ["./gt1l", "./gt1r", "./gt2l", "./gt2r", "./gt3l", "./gt3r"]

# Beam indices
beams = [1, 2, 3, 4, 5, 6]

# Create beam index container
orb_i = 0

# Select the granule

if gran is not None:
    gran = np.asarray(gran).astype(int)

# Raster mask

if fmask is not None:

    print("Reading raster mask ....")

    if fmask.endswith(".tif"):

        # Read in masking grid
        (Xm, Ym, Zm, dX, dY, Proj) = geotiffread(fmask, "A")

    else:

        # Read Hdf5 from memory
        Fmask = h5py.File(fmask, "r")

        # Get variables
        Xm = Fmask["X"][:]
        Ym = Fmask["Y"][:]
        Zm = Fmask["Z"][:]

# Test for black list file

if bfile is not None:

    # Read blacklist file
    blacklist = pd.read_csv(bfile).values


def main(ifile, n=""):

    # Get index of where filename starts
    nidx = ifile.rfind("/") + 1

    # Check if we should process

    if bfile is not None:
        if np.any(ifile[nidx:] == blacklist):
            print("Rejected by blacklist")

            return

    # Check if we are using granules

    if gran is not None:

        # Get string
        gran_str = ifile.split("_")[-3]

        # Calculate the length of the string
        n_str = len(gran_str)

        # Convert to integers
        gran_num = int(gran_str[-2:n_str])

        # Check granule number

        if np.all(gran_num != gran):
            return

    # Access global variable
    global orb_i

    # Check if we already processed the file

    if ifile.endswith("_A.h5") or ifile.endswith("_D.h5"):
        return

    # Beam flag error
    flg_read_err = False

    # Loop trough beams

    for k in range(len(group)):

        # Load full data into memory (only once)
        with h5py.File(ifile, "r") as fi:

            # Try to read vars
            try:

                # Read in varibales of interest (more can be added!)
                dac = fi[group[k] + "/land_ice_segments/geophysical/dac"][:]
                lat = fi[group[k] + "/land_ice_segments/latitude"][:]
                lon = fi[group[k] + "/land_ice_segments/longitude"][:]
                h_li = fi[group[k] + "/land_ice_segments/h_li"][:]
                s_li = fi[group[k] + "/land_ice_segments/h_li_sigma"][:]
                t_dt = fi[group[k] + "/land_ice_segments/delta_time"][:]
                flag = fi[group[k] + "/land_ice_segments/atl06_quality_summary"][:]
                s_fg = fi[group[k] + "/land_ice_segments/fit_statistics/signal_selection_source"][:]
                snr = fi[group[k] + "/land_ice_segments/fit_statistics/snr_significance"][:]
                h_rb = fi[group[k] + "/land_ice_segments/fit_statistics/h_robust_sprd"][:]
                f_sn = fi[group[k] + "/land_ice_segments/geophysical/bsnow_conf"][:]
                tref = fi["/ancillary_data/atlas_sdp_gps_epoch"][:]
                rgt = fi["/orbit_info/rgt"][:] * np.ones(len(lat))
                dh_fit_dx = fi[group[k] + "/land_ice_segments/fit_statistics/dh_fit_dx"][:]
                cycle = fi["/orbit_info/cycle_number"][:] * np.ones(len(lat))

                # Read needed attribues
                spot_number = fi[group[k]].attrs["atlas_spot_number"].decode()
                beam_type = fi[group[k]].attrs["atlas_beam_type"].decode()

                # Tide corrections
                tide_earth = fi[group[k] + "/land_ice_segments/geophysical/tide_earth"][:]
                tide_load = fi[group[k] + "/land_ice_segments/geophysical/tide_load"][:]
                tide_ocean = fi[group[k] + "/land_ice_segments/geophysical/tide_ocean"][:]
                tide_pole = fi[group[k] + "/land_ice_segments/geophysical/tide_pole"][:]

            except IOError:
                raise

                flg_read_err = True
                pass

        # Continue to next beam

        if flg_read_err:
            print(("could not read file:", ifile))

            return

        # Set beam type for file

        if beam_type == "strong":
            # Strong beam
            beam = np.ones(lat.shape)
        else:
            # Weak beam
            beam = np.zeros(lat.shape)

        # Creating array of spot numbers
        spot = float(spot_number) * np.ones(lat.shape)

        # Make sure its a 64 bit float
        dh_fit_dx = np.float64(dh_fit_dx)

        # Apply bounding box

        if bbox[0]:

            # Extract bounding box
            (lonmin, lonmax, latmin, latmax) = bbox

            # Select data inside bounding box
            ibox = (lon >= lonmin) & (lon <= lonmax) & \
                   (lat >= latmin) & (lat <= latmax)

            # Set mask container
            i_m = np.zeros(lat.shape)

            # Set to so we keep data inside box
            i_m[ibox] = 1

        # Raster mask
        elif fmask is not None:

            # Determine projection

            if proj != "4326":

                # Reproject coordinates
                (x, y) = transform_coord("4326", proj, lon, lat)

            else:

                # Keep lon/lat
                x, y = lon, lat

            # Interpolation of grid to points for masking
            i_m = bilinear2d(Xm, Ym, Zm, x.T, y.T, order=1)

            # Set all NaN's to zero
            i_m[np.isnan(i_m)] = 0

        else:

            # Select all boolean
            ibox = np.ones(lat.shape, dtype=bool)

            # Set mask container
            i_m = np.ones(lat.shape)

        # Compute segment difference mask
        mask = segDifferenceFilter(dh_fit_dx, h_li, tol=2)

        # Copy original flag
        q_flag = flag.copy()

        # Quality flag, only keep good data and data inside box
        flag = (flag == 0) & (np.abs(h_li) < 10e3) & (i_m > 0) & mask

        # Only keep good data (update variables)
        (
            lat,
            lon,
            h_li,
            s_li,
            t_dt,
            h_rb,
            s_fg,
            snr,
            q_flag,
            f_sn,
            tide_earth,
            tide_load,
            tide_ocean,
            tide_pole,
            dac,
            rgt,
            cycle,
            beam,
            spot,
        ) = (
            lat[flag],
            lon[flag],
            h_li[flag],
            s_li[flag],
            t_dt[flag],
            h_rb[flag],
            s_fg[flag],
            snr[flag],
            q_flag[flag],
            f_sn[flag],
            tide_earth[flag],
            tide_load[flag],
            tide_ocean[flag],
            tide_pole[flag],
            dac[flag],
            rgt[flag],
            cycle[flag],
            beam[flag],
            spot[flag],
        )

        # Test for no data

        if len(h_li) == 0:
            return

        # Time in decimal years
        t_li = gps2dyr(t_dt + tref)

        # Time in GPS seconds
        t_gps = t_dt + tref

        # Determine track type
        (i_asc, i_des) = track_type(t_li, lat)

        # Determine if to use the index

        if index is not None:

            # Add unique mission identifier
            orb_unique = np.char.add(str(index), str(orb_i)).astype("int")

            # Create orbit number
            orb = np.full(t_gps.shape, orb_unique)

        else:

            # Create orbit number
            orb = np.full(t_gps.shape, orb_i)

        # Construct output name and path
        name, ext = os.path.splitext(os.path.basename(ifile))
        ofile = os.path.join(
            opath, name + "_" + group[k][2:] + "_spot" + spot_number + ext
        )

        # Save track as ascending

        if len(lat[i_asc]) > 1:

            with h5py.File(ofile.replace(".h5", "_A.h5"), "w") as fa:

                fa["orbit"] = orb[i_asc][:]
                fa["lon"] = lon[i_asc][:]
                fa["lat"] = lat[i_asc][:]
                fa["h_elv"] = h_li[i_asc][:]
                fa["s_elv"] = s_li[i_asc][:]
                fa["t_year"] = t_li[i_asc][:]
                fa["h_rb"] = h_rb[i_asc][:]
                fa["s_fg"] = s_fg[i_asc][:]
                fa["snr"] = snr[i_asc][:]
                fa["q_flg"] = q_flag[i_asc][:]
                fa["f_sn"] = f_sn[i_asc][:]
                fa["t_sec"] = t_gps[i_asc][:]
                fa["tide_load"] = tide_load[i_asc][:]
                fa["tide_ocean"] = tide_ocean[i_asc][:]
                fa["tide_pole"] = tide_pole[i_asc][:]
                fa["tide_earth"] = tide_earth[i_asc][:]
                fa["dac"] = dac[i_asc][:]
                fa["rgt"] = rgt[i_asc][:]
                fa["cycle"] = cycle[i_asc][:]
                fa["beam"] = beam[i_asc][:]
                fa["spot"] = spot[i_asc][:]

                ostr = "_A.h5"

        # Save track as desending

        if len(lat[i_des]) > 1:

            with h5py.File(ofile.replace(".h5", "_D.h5"), "w") as fd:

                fd["orbit"] = orb[i_des][:]
                fd["lon"] = lon[i_des][:]
                fd["lat"] = lat[i_des][:]
                fd["h_elv"] = h_li[i_des][:]
                fd["s_li"] = s_li[i_des][:]
                fd["t_year"] = t_li[i_des][:]
                fd["h_rb"] = h_rb[i_des][:]
                fd["s_fg"] = s_fg[i_des][:]
                fd["snr"] = snr[i_des][:]
                fd["q_flg"] = q_flag[i_des][:]
                fd["f_sn"] = f_sn[i_des][:]
                fd["t_sec"] = t_gps[i_des][:]
                fd["tide_load"] = tide_load[i_des][:]
                fd["tide_ocean"] = tide_ocean[i_des][:]
                fd["tide_pole"] = tide_pole[i_des][:]
                fd["tide_earth"] = tide_earth[i_des][:]
                fd["dac"] = dac[i_des][:]
                fd["rgt"] = rgt[i_des][:]
                fd["cycle"] = cycle[i_des][:]
                fd["beam"] = beam[i_des][:]
                fd["spot"] = spot[i_des][:]

                ostr = "_D.h5"

        # Name of file
        try:
            print((ofile.replace(".h5", ostr)))
        except IOError:
            print("Not processed!")

        # Update orbit number
        orb_i += 1


# Run main program

if njobs == 1:

    print("running in serial ...")
    [main(f) for f in ifiles]

else:

    print(("running in parallel (%d jobs) ..." % njobs))
    from joblib import Parallel, delayed

    Parallel(n_jobs=njobs, verbose=5)(
            delayed(main)(f, n) for n, f in enumerate(ifiles)
    )

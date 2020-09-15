#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computes and applies the ocean or load tide correction to height data.
It interpolates values from tidal constituents for a given model
    https://github.com/tsutterley/pyTMD

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
or Global Tide Model (GOT) solutions provided by Richard Ray at GSFC

Make sure the default parameters are set properly in the code (below).

Output:
    (two options)
    1. Applies tide correction and saves the cor as additional variable.
    2. Generates external file with correction for each point (x,y,t,cor).

Requires:
    numpy: Scientific Computing Tools For Python
        http://www.numpy.org
        http://www.scipy.org/NumPy_for_Matlab_Users
    scipy: Scientific Tools for Python
        http://www.scipy.org/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        http://h5py.org
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

Dependencies:
    calc_astrol_longitudes.py: computes the basic astronomical mean longitudes
    calc_delta_time.py: calculates difference between universal and dynamic time
    convert_xy_ll.py: convert lat/lon points to and from projected coordinates
    infer_minor_corrections.py: return corrections for 16 minor constituents
    load_constituent.py: loads parameters for a given tidal constituent
    load_nodal_corrections.py: load the nodal corrections for tidal constituents
    predict_tide_drift.py: predict tidal elevations using harmonic constants
    read_tide_model.py: extract tidal harmonic constants from OTIS tide models
    read_netcdf_model.py: extract tidal harmonic constants from netcdf models
    read_GOT_model.py: extract tidal harmonic constants from GSFC GOT models

Example:
    python corrtide.py '/path/to/data/*.h5' \
            -D /path/to/tidedirectory -T CATS2008 \
            -v lon lat t_sec h_cor -a
"""
import argparse
import datetime as dt
import os
import sys
from collections import OrderedDict
from glob import glob

import h5py
import numpy as np
import pytz

from calc_astrol_longitudes import calc_astrol_longitudes
from calc_delta_time import calc_delta_time
from convert_xy_ll import convert_xy_ll
from infer_minor_corrections import infer_minor_corrections
from load_constituent import load_constituent
from load_nodal_corrections import load_nodal_corrections
from predict_tide_drift import predict_tide_drift
from read_tide_model import extract_tidal_constants
from read_netcdf_model import extract_netcdf_constants
from read_GOT_model import extract_GOT_constants

""" Default parameters. """

# Default variable names of x/y/t/z in the HDF5 files
XVAR = "lon"
YVAR = "lat"
TVAR = "t_sec"
ZVAR = "h_cor"

# Default column numbers of x/y/t/z in the ASCII files
XCOL = 0
YCOL = 1
TCOL = 2
ZCOL = 3

# Default reference epoch of input (height) time in seconds
EPOCH = (1970, 1, 1, 0, 0, 0)


def get_parser():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Computes and apply the ocean or load tide correction."
    )
    parser.add_argument(
        "file",
        metavar="file",
        type=str,
        nargs="+",
        help="ASCII or HDF5 file(s) to process",
    )
    parser.add_argument(
        "-T",
        metavar="tide",
        dest="tide",
        type=str,
        help=("tide model to use"),
        default='CATS2008',
    )
    parser.add_argument(
        "-D",
        metavar="directory",
        dest="directory",
        type=str,
        help=("path to tide directory"),
        default=os.getcwd,
    )
    parser.add_argument(
        "-v",
        metavar=("x", "y", "t", "h"),
        dest="vnames",
        type=str,
        nargs=4,
        help=("variable names of lon/lat/time/height in HDF5 file"),
        default=[XVAR, YVAR, TVAR, ZVAR],
    )
    parser.add_argument(
        "-c",
        metavar=("0", "1", "2"),
        dest="cols",
        type=int,
        nargs=3,
        help=("column positions of lon/lat/time/height in ASCII file"),
        default=[XCOL, YCOL, TCOL, ZCOL],
    )
    parser.add_argument(
        "-e",
        metavar=("Y", "M", "D", "h", "m", "s"),
        dest="epoch",
        type=int,
        nargs=6,
        help=("reference epoch of input time in secs"),
        default=EPOCH,
    )
    parser.add_argument(
        "-a",
        dest="apply",
        action="store_true",
        help=("apply tide correction instead of saving to separate file"),
        default=False,
    )

    return parser


def sec_to_days(
    secs, epoch1=(1970, 1, 1, 0, 0, 0), epoch2=None, tzinfo=pytz.UTC
):
    """
    Convert seconds since epoch1 to days since epoch2.

    If epoch2 is None, keeps epoch1 as the reference time.
    """
    epoch1 = dt.datetime(*epoch1, tzinfo=tzinfo)
    epoch2 = (
        dt.datetime(*epoch2, tzinfo=tzinfo) if epoch2 is not None else epoch1
    )
    secs_btw_epochs = (epoch2 - epoch1).total_seconds()

    return (secs - secs_btw_epochs) / 86400.0  # subtract time diff


def get_xyt_txt(fname, xcol, ycol, tcol):
    """Read x,y,t columns from ASCII file."""

    return np.loadtxt(fname, usecols=(xcol, ycol, tcol), unpack=True)


def get_xyt_h5(fname, xvar, yvar, tvar):
    """Read x,y,t variables from HDF5 file."""
    with h5py.File(fname, "r") as f:
        return f[xvar][:], f[yvar][:], f[tvar][:]


def get_xyt(fname, xvar, yvar, tvar):
    """
    Read x/y/t data from ASCII or HDF5 file.

    x/y/t can be column number or variable names.
    """

    if isinstance(xvar, str):
        return get_xyt_h5(fname, xvar, yvar, tvar)
    else:
        return get_xyt_txt(fname, xvar, yvar, tvar)


def saveh5(outfile, data):
    """ Save data in a dictionary to HDF5 (1d arrays). """
    with h5py.File(outfile, "w") as f:
        [f.create_dataset(key, data=val) for key, val in list(data.items())]
        f.close()

def main():
    # Get command-line args
    args = get_parser().parse_args()
    files = args.file[:]
    vnames = args.vnames[:]
    cols = args.cols[:]
    epoch = args.epoch[:]
    apply_ = args.apply
    model = args.tide
    tide_dir = os.path.expanduser(args.directory)

    # verify model before running program
    model_list = ['CATS0201','CATS2008','CATS2008_load','TPXO9-atlas',
        'TPXO9-atlas-v2','TPXO9.1','TPXO8-atlas','TPXO7.2','TPXO7.2_load',
        'AODTM-5','AOTIM-5','AOTIM-5-2018','GOT4.7','GOT4.7_load','GOT4.8',
        'GOT4.8_load','GOT4.10','GOT4.10_load']
    assert model in model_list, 'Unlisted tide model'

    # In case a string is passed to avoid "Argument list too long"

    if len(files) == 1:
        files = glob(files[0])

    # Check extension of input files

    if files[0].endswith((".h5", ".hdf5", ".hdf", ".H5")):
        print("input is HDF5")
        xvar, yvar, tvar, zvar = vnames
    else:
        print("input is ASCII")
        xvar, yvar, tvar, zvar = cols

    print("parameters:")

    for arg in vars(args).items():
        print(arg)

    print("# of input files:", len(files))

    # select between tide models
    if (model == 'CATS0201'):
        grid_file = os.path.join(tide_dir,'cats0201_tmd','grid_CATS')
        model_file = os.path.join(tide_dir,'cats0201_tmd','h0_CATS02_01')
        reference = 'https://mail.esr.org/polar_tide_models/Model_CATS0201.html'
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (model == 'CATS2008'):
        grid_file = os.path.join(tide_dir,'CATS2008','grid_CATS2008')
        model_file = os.path.join(tide_dir,'CATS2008','hf.CATS2008.out')
        reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/cats2008/')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = 'CATS2008'
        TYPE = 'z'
    elif (model == 'CATS2008_load'):
        grid_file = os.path.join(tide_dir,'CATS2008a_SPOTL_Load','grid_CATS2008a_opt')
        model_file = os.path.join(tide_dir,'CATS2008a_SPOTL_Load','h_CATS2008a_SPOTL_load')
        reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/cats2008/')
        variable = 'tide_load'
        long_name = "Load Tide"
        description = "Local displacement due to Ocean Loading (-6 to 0 cm)"
        model_format = 'OTIS'
        EPSG = 'CATS2008'
        TYPE = 'z'
    elif (model == 'TPXO9-atlas'):
        model_directory = os.path.join(tide_dir,'TPXO9_atlas')
        grid_file = 'grid_tpxo9_atlas.nc.gz'
        model_files = ['h_q1_tpxo9_atlas_30.nc.gz','h_o1_tpxo9_atlas_30.nc.gz',
            'h_p1_tpxo9_atlas_30.nc.gz','h_k1_tpxo9_atlas_30.nc.gz',
            'h_n2_tpxo9_atlas_30.nc.gz','h_m2_tpxo9_atlas_30.nc.gz',
            'h_s2_tpxo9_atlas_30.nc.gz','h_k2_tpxo9_atlas_30.nc.gz',
            'h_m4_tpxo9_atlas_30.nc.gz','h_ms4_tpxo9_atlas_30.nc.gz',
            'h_mn4_tpxo9_atlas_30.nc.gz','h_2n2_tpxo9_atlas_30.nc.gz']
        reference = 'http://volkov.oce.orst.edu/tides/tpxo9_atlas.html'
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'netcdf'
        TYPE = 'z'
        SCALE = 1.0/1000.0
    elif (model == 'TPXO9.1'):
        grid_file = os.path.join(tide_dir,'TPXO9.1','DATA','grid_tpxo9')
        model_file = os.path.join(tide_dir,'TPXO9.1','DATA','h_tpxo9.v1')
        reference = 'http://volkov.oce.orst.edu/tides/global.html'
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (model == 'TPXO8-atlas'):
        grid_file = os.path.join(tide_dir,'tpxo8_atlas','grid_tpxo8atlas_30_v1')
        model_file = os.path.join(tide_dir,'tpxo8_atlas','hf.tpxo8_atlas_30_v1')
        reference = 'http://volkov.oce.orst.edu/tides/tpxo8_atlas.html'
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'ATLAS'
        EPSG = '4326'
        TYPE = 'z'
    elif (model == 'TPXO7.2'):
        grid_file = os.path.join(tide_dir,'TPXO7.2_tmd','grid_tpxo7.2')
        model_file = os.path.join(tide_dir,'TPXO7.2_tmd','h_tpxo7.2')
        reference = 'http://volkov.oce.orst.edu/tides/global.html'
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (model == 'TPXO7.2_load'):
        grid_file = os.path.join(tide_dir,'TPXO7.2_load','grid_tpxo6.2')
        model_file = os.path.join(tide_dir,'TPXO7.2_load','h_tpxo7.2_load')
        reference = 'http://volkov.oce.orst.edu/tides/global.html'
        variable = 'tide_load'
        long_name = "Load Tide"
        description = "Local displacement due to Ocean Loading (-6 to 0 cm)"
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (model == 'AODTM-5'):
        grid_file = os.path.join(tide_dir,'aodtm5_tmd','grid_Arc5km')
        model_file = os.path.join(tide_dir,'aodtm5_tmd','h0_Arc5km.oce')
        reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aodtm-5/')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (model == 'AOTIM-5'):
        grid_file = os.path.join(tide_dir,'aotim5_tmd','grid_Arc5km')
        model_file = os.path.join(tide_dir,'aotim5_tmd','h_Arc5km.oce')
        reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aotim-5/')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (model == 'AOTIM-5-2018'):
        grid_file = os.path.join(tide_dir,'Arc5km2018','grid_Arc5km2018')
        model_file = os.path.join(tide_dir,'Arc5km2018','h_Arc5km2018')
        reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aotim-5/')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (model == 'GOT4.7'):
        model_directory = os.path.join(tide_dir,'GOT4.7','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'GOT'
        SCALE = 1.0/100.0
    elif (model == 'GOT4.7_load'):
        model_directory = os.path.join(tide_dir,'GOT4.7','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_load'
        long_name = "Load Tide"
        description = "Local displacement due to Ocean Loading (-6 to 0 cm)"
        model_format = 'GOT'
        SCALE = 1.0/1000.0
    elif (model == 'GOT4.8'):
        model_directory = os.path.join(tide_dir,'got4.8','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'GOT'
        SCALE = 1.0/100.0
    elif (model == 'GOT4.8_load'):
        model_directory = os.path.join(tide_dir,'got4.8','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_load'
        long_name = "Load Tide"
        description = "Local displacement due to Ocean Loading (-6 to 0 cm)"
        model_format = 'GOT'
        SCALE = 1.0/1000.0
    elif (model == 'GOT4.10'):
        model_directory = os.path.join(tide_dir,'GOT4.10c','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_ocean'
        long_name = "Ocean Tide"
        description = ("Ocean Tides including diurnal and semi-diurnal "
            "(harmonic analysis), and longer period tides (dynamic and "
            "self-consistent equilibrium).")
        model_format = 'GOT'
        SCALE = 1.0/100.0
    elif (model == 'GOT4.10_load'):
        model_directory = os.path.join(tide_dir,'GOT4.10c','grids_loadtide')
        model_files = ['q1load.d.gz','o1load.d.gz','p1load.d.gz','k1load.d.gz',
            'n2load.d.gz','m2load.d.gz','s2load.d.gz','k2load.d.gz',
            's1load.d.gz','m4load.d.gz']
        c = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
        reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        variable = 'tide_load'
        long_name = "Load Tide"
        description = "Local displacement due to Ocean Loading (-6 to 0 cm)"
        model_format = 'GOT'
        SCALE = 1.0/1000.0

    for infile in files:

        # Get data points to interpolate (lon, lat, secs)
        x, y, t = get_xyt(infile, xvar, yvar, tvar)
        npts = len(t)

        # Convert input data time to tide time (days since 1992-01-01)
        print("converting secs to days since 1992-01-01...")
        tide_time = sec_to_days(t, epoch1=epoch, epoch2=(1992, 1, 1, 0, 0, 0))

        # read tidal constants and interpolate to grid points
        if model_format in ('OTIS','ATLAS'):
            amp,ph,D,c = extract_tidal_constants(x, y, grid_file, model_file,
                EPSG, TYPE=TYPE, METHOD='spline', GRID=model_format)
            deltat = np.zeros_like(tide_time)
        elif (model_format == 'netcdf'):
            amp,ph,D,c = extract_netcdf_constants(x, y, model_directory,
                grid_file, model_files, TYPE=TYPE, METHOD='spline', SCALE=SCALE)
            deltat = np.zeros_like(tide_time)
        elif (model_format == 'GOT'):
            amp,ph = extract_GOT_constants(x, y, model_directory, model_files,
                METHOD='spline', SCALE=SCALE)
            # interpolate delta times from calendar dates to tide time
            delta_file = os.path.join(tide_dir,'deltat.data')
            deltat = calc_delta_time(delta_file, tide_time)

        # calculate complex phase in radians for Euler's
        cph = -1j*ph*np.pi/180.0
        # calculate constituent oscillation
        hc = amp*np.exp(cph)

        # predict tidal elevations at time and infer minor corrections
        tide = np.ma.empty((npts),fill_value=np.nan)
        tide.mask = np.any(hc.mask,axis=1)
        tide.data[:] = predict_tide_drift(tide_time, hc, c,
            DELTAT=deltat, CORRECTIONS=model_format)
        minor = infer_minor_corrections(tide_time, hc, c,
            DELTAT=deltat, CORRECTIONS=model_format)
        tide.data[:] += minor.data[:]
        # replace masked and nan values with fill value
        invalid, = np.nonzero(np.isnan(tide.data) | tide.mask)
        tide.data[invalid] = tide.fill_value
        tide.mask[invalid] = True

        if 1:  # FIXME: Modify commented at some point <<<<<<<<<<<<<<<<<<<

            # Apply and save correction (as variable)
            with h5py.File(infile, "a") as f:
                if apply_:
                    f[zvar][:] = f[zvar][:] - tide

                try:
                    f[variable][:] = tide  # update correction
                except KeyError:
                    f[variable] = tide  # create correction
                    # add HDF5 variable attributes
                    f[variable].attrs['reference'] = reference
                    f[variable].attrs['long_name'] = long_name
                    f[variable].attrs['description'] = description
                    f[variable].attrs['model'] = model

            outfile = "{0}_{1}.h5".format(os.path.splitext(infile)[0],model)
            os.rename(infile, outfile)

        """
        else:

            # Save corrections to separate file (x, y, t, tide)
            d = OrderedDict([(xvar, x), (yvar, y),
                             (tvar, t_orig), (variable, tide)])

            if isinstance(xvar, str):
                outfile = "{0}_{1}.h5".format(os.path.splitext(infile)[0],model)
                saveh5(outfile, d)

            else:
                outfile = "{0}_{1}.txt".format(os.path.splitext(infile)[0],model)
                np.savetxt(outfile, np.column_stack(d.values()), fmt='%.6f')
        """

        print("input  <-", infile)
        print("output ->", outfile)


if __name__ == "__main__":
    main()

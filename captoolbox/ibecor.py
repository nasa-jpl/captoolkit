#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computes and applies the inverse barometer correction (IBE) to height data.

It interpolates values from IBE data (a 3d-array) generated from ERA-Interim
sea-level pressure. The 3d IBE data set can be generated using: 

    slp2ibe.py

Make sure the default parameters are set properly in the code (below).

It assumes time in the IBE data is "hours since 1900-1-1" (from Era-Int).
If not, change the time transformation in the code.

Output:
    (two options)
    1. Applies IBE correction and saves the cor as additional variable.
    2. Generates external file with correction for each point (x,y,t,cor).

Example:
    a. To convert ERA-Interim sea-level pressure [Pa] to the
    inverse barometer correction [m]:

        python slp2ibe.py -a -b eraint_sea_level.nc

    b. To apply the IB correction to an ASCII file with x,y,t
    in columns 0,1,2:

        python ibecor.py -a -b file.txt

Notes:
    - For ERA-Interim the point interval on the native Gaussian grid is
      about 0.75 degrees.
    - On sufficiently long time scales and away from coastal effects, the
      ocean’s isostatic response is ~1 cm depression of sea level for a
      1 hecto-Pascal (hPa) or milibar (mbar) increase in P_air (Gill, 1982;
      Ponte and others, 1991; Ponte, 1993).
    - The frequency band 0.03<w<0.5 cpd (T=2-33 days) (a.k.a. the "weather
      band") contains most of the variance in P_air. At higher frequencies,
      tides and measurement noise dominate h, and at lower frequencies,
      seasonal and climatological changes in the ice thickness and the
      underlying ocean state dominate the variability. 
    - The IBE correction has generaly large spatial scales.
    - There can be significant trends in P_air on time scales of 1-3 yr.
    - ERA-Interim MSL pressure has time units: hours since 1900-01-01 00:00:0.0

    The sea level increases (decreases) by approximately 1 cm when air
    pressure decreases (increases) by approximately 1 mbar. The inverse
    barometer correction (IBE) that must be subtracted from the sea surface
    height is simply given by:

        h_ibe = (-1/rho g) * (P - P_ref)

    where P_ref is the global "mean" pressure (reference pressure) over the
    ocean (rho is sea water density and g gravity). For most applications,
    P_ref is assumed to be a constant (e.g., 1013.3 mbar).

    See Dorandeu and Le Traon, 1999:

        https://goo.gl/zSCptr

    Our correction uses P_ref spatially variant:

        h_ibe(x,y,t) = (-1/rho g) * [P(x,y,t) - P_ref(x,y)]

    where P_ref(x,y) is the climatological mean at each location.

    Several refereces here:

        https://link.springer.com/chapter/10.1007/978-3-662-04709-5_88

    The IBE correction should be applied as:

        h_cor = h - h_ibe

    If the IBE data cube is global (-90 to 90), subset it to speed up I/O!

Test:
    To test if the generated IBE data is correct, uncomment the 'Test'
    section in the code and compare the plots with the ones provided.

Download:
    Download the latest Era-Int MSLP and generate a new IBE product.
    See how on ibe/README.txt and ibe/geteraint.py.

"""
import os
import sys
import h5py
import argparse
import numpy as np
import datetime as dt
import seaborn as sns
from glob import glob
from scipy import ndimage
from collections import OrderedDict
import matplotlib.pyplot as plt


""" Default parameters. """

# Default location of IBE file (HDF5).
# If passed as command-line arg, this will be ignored.
IBEFILE = 'IBE_antarctica_3h_19900101_20170331.h5'

# Default variable names of x/y/t/z in the IBE file
#NOTE: It assumes IBE time is 'hours since 1900' (from ERA-Int) 
XIBE = 'lon'
YIBE = 'lat'
TIBE = 'time'
ZIBE = 'ibe'

# Default variable names of x/y/t/z in the HDF5 files
XVAR = 'lon'
YVAR = 'lat'
TVAR = 't_sec'
ZVAR = 'h_cor'

# Default column numbers of x/y/t/z in the ASCII files
XCOL = 0
YCOL = 1
TCOL = 2
ZCOL = 3

# Default reference epoch of input (height) time in seconds
EPOCH = (1970,1,1,0,0,0)


def get_parser():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Computes and apply the inverse barometer correction.')

    parser.add_argument(
            'file', metavar='file', type=str, nargs='+',
            help='ASCII or HDF5 file(s) to process')

    parser.add_argument(
            '-b', metavar='ibefile', dest='ibefile', type=str, nargs=1,
            help=('path to (HDF5) IBE file (created with slp2ibe.py)'),
            default=[IBEFILE],)

    parser.add_argument(
            '-v', metavar=('x','y','t','h'), dest='vnames', type=str, nargs=4,
            help=('variable names of lon/lat/time/height in HDF5 file'),
            default=[XVAR,YVAR,TVAR,ZVAR],)

    parser.add_argument(
            '-c', metavar=('0','1','2'), dest='cols', type=int, nargs=3,
            help=('column positions of lon/lat/time/height in ASCII file'),
            default=[XCOL,YCOL,TCOL,ZCOL],)

    parser.add_argument(
            '-e', metavar=('Y','M','D','h','m','s'), dest='epoch',
            type=int, nargs=6,
            help=('reference epoch of input time in secs'),
            default=EPOCH,)

    parser.add_argument(
            '-t', metavar=('t1','t2'), dest='tspan',
            type=float, nargs=2,
            help=('time span for subsetting IBE (in dec years)'),
            default=[],)

    parser.add_argument(
            '-a', dest='apply', action='store_true',
            help=('apply IBE cor instead of saving to separate file'),
            default=False)

    return parser


def secs_to_hours(secs, epoch1=(1970,1,1,0,0,0), epoch2=None):
    """
    Convert seconds since epoch1 to hours since epoch2.

    If epoch2 is None, keeps epoch1 as the reference time.
    """
    epoch1 = dt.datetime(*epoch1)
    epoch2 = dt.datetime(*epoch2) if epoch2 is not None else epoch1
    secs_btw_epochs = (epoch2 - epoch1).total_seconds()
    return (secs - secs_btw_epochs) / 3600.  # subtract time diff


def get_xyt_txt(fname, xcol, ycol, tcol):
    """Read x,y,t columns from ASCII file."""
    return np.loadtxt(fname, usecols=(xcol,ycol,tcol), unpack=True)


def get_xyt_h5(fname, xvar, yvar, tvar):
    """Read x,y,t variables from HDF5 file."""
    with h5py.File(fname, 'r') as f:
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
    with h5py.File(outfile, 'w') as f:
        [f.create_dataset(key, data=val) for key, val in data.items()]
        f.close()


def interp3d(x, y, z, v, xi, yi, zi, **kwargs):
    """
    Fast 3D interpolation.
    
    Given a 3d-array (a cube) "v" with pixel coordinates "x","y","z"
    (0-, 1-, 2-axis), interpolate values "xi","yi","zi" using linear
    interpolation.

    Additional kwargs are passed on to ``scipy.ndimage.map_coordinates``.

    Note that in the case of "real-world" coordinates, we might have:
    x=time (0-axis), y=latitude (1-axis), z=longitude (2-axis) or
    x=bands, y=rows, z=cols. Example:
    
        interp_pts = interp3d(time, lat, lon, grid, t_pts, y_pts, x_pts)

    See:
        https://goo.gl/KmfnPk
        https://goo.gl/g7APAf
    """
    def interp_pixels(grid_coords, interp_coords):
        """ Map interpolation coordinates to pixel locations. """
        grid_pixels = np.arange(len(grid_coords))
        if np.all(np.diff(grid_coords) < 0):
            grid_coords, grid_pixels = grid_coords[::-1], grid_pixels[::-1]
        return np.interp(interp_coords, grid_coords, grid_pixels)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)  # to ensure float output
    coords = [interp_pixels(*item) for item in zip([x, y, z], [xi, yi, zi])]
    ndimage.map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def wrap_to_180(lon):
    """ Wrapps longitude to -180 to 180 degrees. """
    lon[lon>180] -= 360.
    return lon


def main():

    # Get command-line args
    args = get_parser().parse_args()
    files = args.file[:]
    vnames = args.vnames[:]
    cols = args.cols[:]
    epoch = args.epoch[:]
    tspan = args.tspan[:]
    apply_ = args.apply
    ibefile = args.ibefile[0]

    # In case a string is passed to avoid "Argument list too long"
    if len(files) == 1:
        files = glob(files[0])

    # Check extension of input files
    if files[0].endswith(('.h5', '.hdf5', '.hdf', '.H5')):
        print 'input is HDF5'
        xvar, yvar, tvar, zvar = vnames
    else:
        print 'input is ASCII'
        xvar, yvar, tvar, zvar = cols

    print 'parameters:'
    for arg in vars(args).iteritems(): print arg

    print '# of input files:', len(files)

    # Get the IBE data (3d array), outside main loop (load only once!)
    print 'loading ibe cube ...'
    f = h5py.File(ibefile, 'r')
    x_ibe = f[XIBE][:]  # [deg]
    y_ibe = f[YIBE][:]  # [deg]
    t_ibe = f[TIBE][:]  # [hours since 1900-1-1]
    z_ibe = f[ZIBE]#[:] # ibe(time,lat,lon) [m].  ##NOTE: WARNING: large dataset!

    # Subset IBE datset for speed up

    if tspan:
        
        print 'subsetting ibe ...'
        t1, t2 = tspan

        # Filter time
        t_year = (t_ibe/8760.) + 1900  # hours since 1900 -> years
        k, = np.where((t_year >= t1) & (t_year <= t2))
        k1, k2 = k[0], k[-1]+1

        # Subset
        t_ibe = t_ibe[k1:k2]
        z_ibe = z_ibe[k1:k2,:,:]

        #--- Test (plot for testing) -----------------------

        if 0:
            import pandas as pd
            import matplotlib.pyplot as plt
            from mpl_toolkits.basemap import Basemap

            t_year = t_year[k1:k2]
            t_year = (t_year-2007) * 365.25 - 26
            # 26(25) Leap days from 1900 to 2007(2000)

            find_nearest = lambda arr, val: (np.abs(arr-val)).argmin()

            if 1:
               # Single grid cell centered on Larsen-C
                j = find_nearest(y_ibe, -67.5)
                i = find_nearest(x_ibe, 297.5-360)
                p = z_ibe[:,j,i]
            else:
                # Single grid cell centered on Brunt
                j = find_nearest(y_ibe, -75.6)
                i = find_nearest(x_ibe, 333.3-360)
                p = z_ibe[:,j,i]

            plt.plot(t_year, p, linewidth=2)
            plt.show()

            sys.exit()

        if 0:
            # Map of Antarctica
            fig = plt.figure()
            ax = plt.gca()

            m = Basemap(projection='spstere', boundinglat=-60, lon_0=180)

            xx, yy = np.meshgrid(x_ibe, y_ibe)
            xx, yy = m(xx, yy)

            # plot data
            grid = z_ibe[10] - z_ibe[10].mean()
            c = m.pcolormesh(xx, yy, grid, edgecolor='k')

            # Plot ice-shelf boundaries
            FILE_SHELF = ('/Users/paolofer/data/masks/scripps/'
                          'scripps_iceshelves_v1_geod.txt')

            FILE_COAST = ('/Users/paolofer/data/masks/scripps/'
                          'scripps_coastline_v1_geod.txt')

            x, y = np.loadtxt(FILE_SHELF, usecols=(0,1),
                              unpack=True, comments='%')

            x, i_uni, i_inv = np.unique(x, return_index=True,
                                        return_inverse=True)
            y = y[i_uni]
            x, y = m(x, y)
            plt.scatter(x, y, s=5, c='.5', facecolor='.5',
                        lw=0, rasterized=True, zorder=2)

            # Plot location of time series
            px, py = m(297.5, -67.5)
            plt.scatter(px, py, s=50, c='r', facecolor='.5',
                        lw=0, rasterized=True, zorder=10)

            plt.show()
            sys.exit()

    else:

        z_ibe = z_ibe[:]


    for infile in files:

        # Get data points to interpolate 
        x, y, t = get_xyt(infile, xvar, yvar, tvar)

        t_orig = t.copy()

        # Convert input data time to IBE time (hours since 1900-1-1)
        print 'converting secs to hours ...'
        t = secs_to_hours(t, epoch1=epoch, epoch2=(1900,1,1,0,0,0))

        # Assert lons are consistent
        x_ibe = wrap_to_180(x_ibe)
        x = wrap_to_180(x)

        # Interpolate x/y/t onto IBE (3d-array)
        print 'interpolating x/y/t onto IBE cube ...'
        h_ibe = interp3d(t_ibe, y_ibe, x_ibe, z_ibe, t, y, x)

        if apply_:

            # Apply and save correction (as h_ibe)
            with h5py.File(infile, 'a') as f:
                f[zvar][:] = f[zvar][:] - h_ibe
                f['h_ibe'] = h_ibe

            outfile = os.path.splitext(infile)[0] + '_IBE.h5'  # HDF5
            os.rename(infile, outfile)

        else:

            # Save corrections to separate file (x, y, t, h_ibe)
            d = OrderedDict([(xvar, x), (yvar, y), (tvar, t_orig), ('h_ibe', h_ibe)])

            if isinstance(xvar, str):
                outfile = os.path.splitext(infile)[0] + '_IBE.h5'  # HDF5
                saveh5(outfile, d)

            else:
                outfile = os.path.splitext(infile)[0] + '_IBE.txt'  # ASCII
                np.savetxt(outfile, np.column_stack(d.values()), fmt='%.6f')

        print 'input  <-', infile
        print 'output ->', outfile


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Computes satellite altimeter crossovers.

Calculates location and values at orbit-intersection points by the means of
linear or cubic interpolation between the two/four closest records to the
crossover location for ascending and descending orbits.

Notes: 
    The program needs to have an input format according as followed:
    orbit nr., latitude (deg), longitude (deg), time (yr), surface height (m).

    The crossover location is rejected if any of the four/eight closest
    records has a distance larger than an specified threshold, provided by
    the user. To reduce the crossover computational time the input data can
    be divided into tiles, where each tile is solved independently of the
    others. This reduced the number of obs. in the inversion to calculate the
    crossing position. The input data can further be reprojected, give user
    input, to any wanted projection. Changing the projection can help to
    improve the estimation of the crossover location, due to improved
    geometry of the tracks (straighter). For each crossover location the
    closest records are linearly interpolated using 1D interpolation, by
    using the first location in each track (ascending and descending) as a
    distance reference. The user can also provide a resolution parameter used
    to determine how many point (every n:th point) to be used for solving for
    the crossover intersection to improve computation speed. However, the
    closest points are still used for interpolation of the records to the
    crossover location. The program can also be used for inter-mission
    elevation change rate and bias estimation, which requires the mission to
    provide a extra column with satellite mission indexes.

Args:
    ifile: Name of input file, if ".npy" data is read as binary.
    ofile: Name of output file, if ".npy" data is saved as binary
    icols: String of indexes of input columns: "orb lat lon t h".
    radius: Cut-off radius (from crossing pt) for accepting crossover (m).
    proj: EPSG projection number (int).
    dxy: Tile size (km).
    nres: Track resolution: Use n:th for track intersection (int).
    buff: Add buffer around tile (km)
    mode: Interpolation mode "linear" or "cubic".
    mission: Inter-mission rate or bias estimation "True", "False"

Example:
    xover2.py ~/data/xover/vostok/unc/RA2_D_ICE_A.h5 \
            -v orbit orb_type lon lat t_year h_res -t 2003 2009 -d 50 -p 3031

"""
import os
import sys
import numpy as np
import pyproj
import h5py
import argparse
import warnings
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

# Ignore all warnings
warnings.filterwarnings("ignore")


def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Program for computing satellite/airborne crossovers.')
    parser.add_argument(
            'input', metavar='ifile', type=str, nargs='+',
            help='name of input file(s) (HDF5, ASCII, Numpy)')
    parser.add_argument(
            '-o', metavar='ofile', dest='output', type=str, nargs=1,
            help='name of output file (HDF5, ASCII, Numpy)',
            default=[None])
    parser.add_argument(
            '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
            help='maximum interpolation distance from crossing location (m)',
            default=[350],)
    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
            default=['4326'],)
    parser.add_argument(
            '-d', metavar=('tile_size'), dest='dxy', type=int, nargs=1,
            help='tile size (km)',
            default=[0],)
    parser.add_argument(
            '-k', metavar=('subsample'), dest='nres', type=int, nargs=1,
            help='along-track subsampling every k:th point (for speed up)',
            default=[1],)
    parser.add_argument(
            '-b', metavar=('buffer'), dest='buff', type=int, nargs=1,
            help=('tile buffer (km)'),
            default=[0],)
    parser.add_argument(
            '-m', metavar=None, dest='mode', type=str, nargs=1,
            help='interpolation method, "linear" or "cubic"',
            choices=('linear', 'cubic'), default=['linear'],)
    parser.add_argument(
            '-i', metavar=('satid'), dest='satid', type=str, nargs=1,
            help='sat id for multi-mission mode: var_name or col_num',
            default=[None],)
    parser.add_argument(
            '-c', metavar=('0','1','2','3','4','5'), dest='cols', type=int, nargs=6,
            help='col# of orbit/type/lon/lat/t/h in the ASCII',
            default=[0,2,1,3,4,5],)
    parser.add_argument(
            '-v', metavar=('o','a','x','y','t','h'), dest='vnames', type=str, nargs=6,
            help=('name of orbit/type/lon/lat/t/h in the HDF5'),
            default=[None],)
    parser.add_argument(
            '-t', metavar=('t1','t2'), dest='tspan', type=float, nargs=2,
            help='only compute crossovers for given time span',
            default=[None],)
    parser.add_argument(
            '-n', metavar=('njobs'), dest='njobs', type=int, nargs=1,
            help="for parallel processing of individual tiles, optional",
            default=[1],)
    return parser.parse_args()


def orbit_type(lat, orbits):
    """ Determine if ascending or descending orbit. """

    # Unique orbits
    tracks = np.unique(orbits)

    # Output arrays
    Ia = np.full(lat.shape, np.nan)
    Id = np.full(lat.shape, np.nan)

    # Loop trough orbits
    for i in xrange(len(tracks)):

        # Get individual orbits
        I = orbits == tracks[i]

        # Get two first coordinates
        coord = np.abs(lat[I][0:2])
        #coord = np.abs(lat[I][[0,-1]])  #FIXME: Use this!!!

        # Test tracks length
        if len(coord) < 2:
            continue

        # Determine orbit type
        if coord[0] < coord[1]:

            # Save orbit type to ascending vector
            Ia[i] = tracks[i]

        else:

            # Save orbit type to descending vector
            Id[i] = tracks[i]

    # Output index vector's
    return Ia[~np.isnan(Ia)], Id[~np.isnan(Id)]


def intersect(x_down, y_down, x_up, y_up):
    """ Find orbit crossover locations. """
    
    p = np.column_stack((x_down, y_down))

    q = np.column_stack((x_up, y_up))

    (p0, p1, q0, q1) = p[:-1], p[1:], q[:-1], q[1:]

    rhs = q0 - p0[:, np.newaxis, :]

    mat = np.empty((len(p0), len(q0), 2, 2))

    mat[..., 0] = (p1 - p0)[:, np.newaxis]

    mat[..., 1] = q0 - q1

    mat_inv = -mat.copy()

    mat_inv[..., 0, 0] = mat[..., 1, 1]

    mat_inv[..., 1, 1] = mat[..., 0, 0]

    det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]

    mat_inv /= det[..., np.newaxis, np.newaxis]

    import numpy.core.umath_tests as ut

    params = ut.matrix_multiply(mat_inv, rhs[..., np.newaxis])

    intersection = np.all((params >= 0) & (params <= 1), axis=(-1, -2))

    p0_s = params[intersection, 0, :] * mat[intersection, :, 0]

    return p0_s + p0[np.where(intersection)[0]]


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def get_bboxs(x, y, dxy):
    """
    Define blocks (bbox) for speeding up the processing. 

    Args:
        x/y: must be in grid projection, e.g. stereographic (m).
        dxy: grid-cell size.
    """
    # Number of tile edges on each dimension 
    Nns = int(np.abs(y.max() - y.min()) / dxy) + 1
    New = int(np.abs(x.max() - x.min()) / dxy) + 1

    # Coord of tile edges for each dimension
    xg = np.linspace(x.min(), x.max(), New)
    yg = np.linspace(y.min(), y.max(), Nns)

    # Vector of bbox for each cell
    bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg

    return bboxs


# Read in parameters
args = get_args()
ifiles  = args.input[:]
ofile_ = args.output[0]
radius = args.radius[0]
proj   = args.proj[0]
dxy    = args.dxy[0]
nres   = args.nres[0]
buff   = args.buff[0]
mode   = args.mode[0]
satid  = args.satid[0]
icols  = args.cols[:]
vnames = args.vnames[:]
tspan = args.tspan[:]
njobs = args.njobs[0]

print tspan

print 'parameters:'
for arg in vars(args).iteritems(): print arg


# If satid is a column number, str -> int
if satid and satid.isdigit():
    satid = int(satid)

# Get column numbers
(co, ca, cx, cy, ct, cz) = icols

# Get variable names
ovar, avar, xvar, yvar, tvar, zvar = vnames 

# Test for stereographic
if proj != "4326":

    # Convert to meters
    dxy *= 1e3


def main(ifile):
    """ Find and compute crossover values. """

    print 'processing file:', ifile, '...'

    # Determine input file type
    if ifile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):

        # Load all 1d variables needed
        with h5py.File(ifile, 'r') as fi:
            orbit = fi[ovar][:]
            otype = fi[avar][:]
            lon = fi[xvar][:]
            lat = fi[yvar][:]
            time = fi[tvar][:]
            height = fi[zvar][:]
            mid = fi[satid][:] if satid else satid
    else:

        # Load data points - ascii
        F = pd.read_csv(ifile, header=None, delim_whitespace=True).as_matrix()

        orbit = F[:,co]
        otype = F[:,ca]
        lon = F[:,cx]
        lat = F[:,cy]
        time = F[:,ct]
        height = F[:,cz]
        mid = F[:,satid] if satid else satid

    # If time span given, filter out invalid data
    if len(tspan) > 1:
        t1, t2 = tspan
        idx, = np.where((time >= t1) & (time <= t2))
        orbit = orbit[idx]
        otype = otype[idx]
        lon = lon[idx]
        lat = lat[idx]
        time = time[idx]
        height = height[idx]
        mid = mid[idx] if satid else satid

    # Transform to wanted coordinate system
    (xp, yp) = transform_coord(4326, proj, lon, lat)

    # Time limits (yr)
    tmin = time.min()
    tmax = time.max()

    # Interpolation type and number of needed points
    if mode == "linear":
        # Linear interpolation
        nobs  = 2
        order = 1

    else:
        # Cubic interpolation
        nobs  = 4
        order = 3

    # Tiling option - "on" or "off"
    if dxy != 0:
        bboxs = get_bboxs(xp, yp, dxy)

    else:
        bboxs = [(-1e16, 1e16, -1e16, 1e16)]  #NOTE: Double check this.

    print 'number of sub-tiles:', len(bboxs)

    # Initiate output container (much larger than it needs to be)
    out = np.full((len(orbit), 10), np.nan)

    # Plot for testing
    if 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(lon[otype==0], lat[otype==0], '.')
        plt.figure()
        plt.plot(lon[otype==1], lat[otype==1], '.')
        plt.show()

    # Initiate xover counter
    i_xover = 0

    # Loop through each sub-tile
    for k,bbox in enumerate(bboxs):

        print 'tile #', k

        # Bounding box of grid cell
        xmin, xmax, ymin, ymax = bbox

        # Get the tile indices
        idx, = np.where( (xp >= xmin - buff) & (xp <= xmax + buff) & 
                         (yp >= ymin - buff) & (yp <= ymax + buff) )

        # Extract tile data
        orbits = orbit[idx]
        otypes = otype[idx]
        lons = lon[idx]
        lats = lat[idx]
        x = xp[idx]
        y = yp[idx]
        h = height[idx]
        t = time[idx]
        m = mid[idx] if satid else []

        # Get unique identifiers of asc/des points (full tracks)
        oa = np.unique(orbits[otypes==0])
        od = np.unique(orbits[otypes==1])

        # Test if tile is empty
        if len(orbits) == 0:
            continue

        # Loop through ascending tracks
        for ka in xrange(len(oa)):

            # Index for single ascending orbit
            Ia = orbits == oa[ka]

            # Extract single orbit
            xa = x[Ia]
            ya = y[Ia]
            ta = t[Ia]
            ha = h[Ia]
            ma = m[Ia][0] if satid else np.nan

            # Loop through descending tracks
            for kd in xrange(len(od)):

                # Index for single descending orbit
                Id = orbits == od[kd]

                # Extract single orbit
                xd = x[Id]
                yd = y[Id]
                td = t[Id]
                hd = h[Id]
                md = m[Id][0] if satid else np.nan

                # Test length of vector
                if len(xa) < 3 or len(xd) < 3:
                    continue
                
                # Initial crossing test -  start and end points
                cxy_intial = intersect(xa[[0, -1]], ya[[0, -1]], xd[[0, -1]], yd[[0, -1]])
                
                # Test for crossing
                if len(cxy_intial) == 0:
                    continue

                # Compute exact crossing - full set of observations, or every n:th point
                cxy_main = intersect(xa[::nres], ya[::nres], xd[::nres], yd[::nres])

                # Test again for crossing
                if len(cxy_main) == 0:
                    continue

                # Extract crossing coordinates
                xi = cxy_main[0][0]
                yi = cxy_main[0][1]

                # Get start coordinates of orbits
                xa0 = xa[0]
                ya0 = ya[0]
                xd0 = xd[0]
                yd0 = yd[0]

                # Compute distance from crossing node to each arc
                da = (xa - xi) * (xa - xi) + (ya - yi) * (ya - yi)
                dd = (xd - xi) * (xd - xi) + (yd - yi) * (yd - yi)

                # Sort according to distance
                Ida = np.argsort(da)
                Idd = np.argsort(dd)

                # Sort arrays - asc
                xa = xa[Ida]
                ya = ya[Ida]
                ta = ta[Ida]
                ha = ha[Ida]
                da = da[Ida]

                # Sort arrays - des
                xd = xd[Idd]
                yd = yd[Idd]
                td = td[Idd]
                hd = hd[Idd]
                dd = dd[Idd]

                # Get distance of four closest observations
                dad = np.vstack((da[[0, 1]], dd[[0, 1]]))

                # Test if any point is too far away
                if np.any(np.sqrt(dad) > radius):
                    continue

                # Test if enough obs. are available for interpolation
                if (len(xa) < nobs) or (len(xd) < nobs):
                    continue

                # Compute distance again from the furthest point
                da0 = (xa - xa0) * (xa - xa0) + (ya - ya0) * (ya - ya0)
                dd0 = (xd - xd0) * (xd - xd0) + (yd - yd0) * (yd - yd0)

                # Compute distance again from the furthest point
                dai = (xi - xa0) * (xi - xa0) + (yi - ya0) * (yi - ya0)
                ddi = (xi - xd0) * (xi - xd0) + (yi - yd0) * (yi - yd0)

                ##TODO: Try interpolation with np.interp1d()!

                # Interpolate height to crossover location
                Fhai = InterpolatedUnivariateSpline(da0[0:nobs], ha[0:nobs], k=order)
                Fhdi = InterpolatedUnivariateSpline(dd0[0:nobs], hd[0:nobs], k=order)

                # Interpolate time to crossover location
                Ftai = InterpolatedUnivariateSpline(da0[0:nobs], ta[0:nobs], k=order)
                Ftdi = InterpolatedUnivariateSpline(dd0[0:nobs], td[0:nobs], k=order)
                
                # Get interpolated values - height
                hai = Fhai(dai)
                hdi = Fhdi(ddi)

                # Get interpolated values - time
                tai = Ftai(dai)
                tdi = Ftdi(ddi)
                
                # Test interpolate time values
                if (tai > tmax) or (tai < tmin) or (tdi > tmax) or (tdi < tmin):
                    continue
                
                # Compute differences and save parameters
                out[i_xover,0] = xi
                out[i_xover,1] = yi
                out[i_xover,2] = hai - hdi
                out[i_xover,3] = tai - tdi
                out[i_xover,4] = tai
                out[i_xover,5] = tdi
                out[i_xover,6] = hai
                out[i_xover,7] = hdi
                out[i_xover,8] = ma
                out[i_xover,9] = md

                # Increment counter
                i_xover += 1

    # Remove invalid rows 
    out = out[~np.isnan(out[:,2]),:]

    # Test if output container is empty 
    if len(out) == 0:
        print 'no crossovers found!'
        return

    # Remove the two id columns if they are empty 
    out = out[:,:-2] if np.isnan(out[:,-1]).all() else out

    # Transform coords back to lat/lon
    out[:,0], out[:,1] = transform_coord(proj, '4326', out[:,0], out[:,1])

    # Name of each variable
    fields = ['lon', 'lat', 'dh', 'dt', 'h_asc', 'h_des', 't_asc', 't_des']

    if satid:
        fields.extend(['m_asc', 'm_des'])

    # Create output file name if not given 
    if ofile_ is None:
        path, ext = os.path.splitext(ifile)
        ofile = path + '_xover' + ext
    else:
        ofile = ofile_
        
    # Determine data format
    if ofile.endswith('.npy'):
        
        # Save as binary file
        np.save(ofile, out)

    elif ofile.endswith(('.h5', '.H5', '.hdf', '.hdf5')):

        # Create h5 file
        with h5py.File(ofile, 'w') as f:
            
            # Loop through fields
            [f.create_dataset(k, data=d) for k,d in zip(fields, out.T)]

    else:

        # Save data to ascii file
        np.savetxt(ofile, out, delimiter="\t", fmt="%8.5f")

    print 'ouput ->', ofile


if __name__ == '__main__':

    if njobs == 1:
        print 'running sequential code ...'
        [main(f) for f in ifiles]

    else:
        print 'running parallel code (%d jobs) ...' % njobs
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in ifiles)

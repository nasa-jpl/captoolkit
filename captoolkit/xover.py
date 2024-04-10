#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import pyproj
import h5py
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from datetime import datetime
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""

Program for computing satellite crossovers from ascending and 
descending orbits using linear or cubic interpolation. 

The program takes as input two files separated into asc. and des. orbits 
with a minimum a number of variables: orbit number, lon ,lat, time and 
height. The user can provide three extra variables if needed. For radar 
this might be waveform parameters needed to perform account for changed in 
the scattering regime.

The software uses internal tiling to speed up computation of the crossovers,
with dimensions specified by the user. It can further process data divided 
into external tiles provided by "tile.py". To further speed up processing the 
user can downsample the tracks. This allows for a faster computation of the 
crossing point, but the difference is computed using the native sampling. 

Notes:
    
    The software currently works best using polar-stereo coordinates (3031 oe 3413)
    so please use these for now!
    
    For external tile processing please use "tile.py" with the same extent 
    for the A and D files. This as the program uses the tile numbering to 
    determine which of the tiles should be crossed together.
    
    When running in external tile-mode the saved file with crossovers
    will be appended with "_XOVERS_AD/DA". Please use "_A" or "_D" in the
    filename to indicate Asc or Des tracks when running in tile mode.
    
    At each crossover location the surface slope is also computed using the 
    difference between height and distance from the two closest points used 
    for the crossover interpolation for each track. The slopes are saved in 
    units of degrees. 
    
    If you are using external tile-mode "-f" please provide input path as a 
    strings: './tiles/*_a.h5' './tiles/*_d.h5'
    
Example:

    python xover.py a.h5 d.h5 -o xover.h5 -r 350 -p 3031 -d 100 -k 1 1\
    -m linear -v orb lon lat time height dum dum dum -b 10
    
    python xover.py './tiles/*_a.h5' './tiles/*_d.h5' -r 350 -p 3031 -d 100 -k\
    1 1 -m linear -v orb lon lat time height dum dum dum -f -b 10
    
Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Fernando Paolo (paolofer@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""

# Ignore all warnings
warnings.filterwarnings("ignore")

def get_args():
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='Program for computing satellite/airborne crossovers.')
    parser.add_argument(
            'input', metavar='ifile', type=str, nargs=2,
            help='name of two input files to cross (HDF5)')
    parser.add_argument(
            '-o', metavar='ofile', dest='output', type=str, nargs=1,
            help='name of output file (HDF5)',
            default=[None])
    parser.add_argument(
            '-r', metavar=('radius'), dest='radius', type=float, nargs=1,
            help='maximum interpolation distance from crossing (m)',
            default=[350],)
    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('projection: EPSG number (AnIS=3031, GrIS=3413)'),
            default=['4326'],)
    parser.add_argument(
            '-d', metavar=('tile_size'), dest='dxy', type=int, nargs=1,
            help='tile size (km)',
            default=[None],)
    parser.add_argument(
            '-k', metavar=('na','nd'), dest='nres', type=int, nargs=2,
            help='along-track sample every n:th pnt for each file',
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
            '-v', metavar=('o','x','y','t','h','b','l','t'), dest='vnames', type=str, nargs=8,
            help=('main vars: names if HDF5, orbit/lon/lat/time/height/bs/lew/tes'),
            default=[None],)
    parser.add_argument(
            '-t', metavar=('t1','t2'), dest='tspan', type=float, nargs=2,
            help='only compute crossovers for given time span',
            default=[None,None],)
    parser.add_argument(
            '-f', dest='tile', action='store_true',
            help=('run in tile mode'),
            default=False)
    parser.add_argument(
            '-q', dest='plot', action='store_true',
            help=('plot for inspection'),
            default=False)
    parser.add_argument(
            '-i', dest='diff', action='store_true',
            help=('do not interpolate vars just take diff'),
            default=False)
            
    return parser.parse_args()


def intersect(x_down, y_down, x_up, y_up):
    """ Find orbit crossover locations.
        Link to algorithm

        https://stackoverflow.com/questions/
        17928452/find-all-intersections-of-
        xy-data-point-graph-with-numpy

    """
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


def get_bboxs_old(xmin, xmax, ymin, ymax, dxy):
    """
    Define blocks (bbox) for speeding up the processing. 

    Args:
        xmin/xmax/ymin/ymax: must be in grid projection: stereographic (m).
        dxy: grid-cell size.
    """
    # Number of tile edges on each dimension 
    Nns = int(np.abs(ymax - ymin) / dxy) + 1
    New = int(np.abs(xmax - xmin) / dxy) + 1

    # Coord of tile edges for each dimension
    xg = np.linspace(xmin, xmax, New)
    yg = np.linspace(ymin, ymax, Nns)

    # Vector of bbox for each cell
    bboxs = [(w,e,s,n) for w,e in zip(xg[:-1], xg[1:]) 
                       for s,n in zip(yg[:-1], yg[1:])]
    del xg, yg

    return bboxs


def get_bboxs(x, y, xmin, xmax, ymin, ymax, dxy, buff):
    """
        Define blocks (bbox) for speeding up the processing.
        
        Args:
        xmin/xmax/ymin/ymax: must be in grid projection: stereographic (m).
        dxy: grid-cell size.
    """
    
    # Number of tile edges on each dimension
    Nns = int(np.abs(ymax - ymin) / dxy) + 1
    New = int(np.abs(xmax - xmin) / dxy) + 1
    
    # Coord of tile edges for each dimension
    xg = np.linspace(xmin-buff, xmax+buff, New)
    yg = np.linspace(ymin-buff, ymax+buff, Nns)
    
    # Indicies for each grid-cell
    bboxs = stats.binned_statistic_2d(x, y, np.ones(x.shape),'count', bins=[xg,yg]).binnumber
    
    return bboxs

def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def interp1D(x, y, xi, n=1):
    """ 1D interpolation """
    
    # Sort data
    idx = np.argsort(x)
    
    # Sort arrays
    x, y = x[idx], y[idx]
    
    # Create interpolator
    Fi = InterpolatedUnivariateSpline(x, y, k=n)
    
    # Interpolated value
    yi = Fi(xi)
    
    return yi

def tile_num(fname):
    """ Extract tile number from file name. """
    l = os.path.splitext(fname)[0].split('_')  # fname -> list
    i = l.index('tile')
    return int(l[i+1])

def match_tiles(str1,str2,key):
    """ Matches tile indices """

    # Get file names
    files1 = glob.glob(str1)
    files2 = glob.glob(str2)

    # Create output list
    f1out = []
    f2out = []

    # Loop trough files-1
    for file1 in files1:

        # Get tile index
        f1 = tile_num(file1)

        # Loop trough files-2
        for file2 in files2:

            # Get tile index
            f2 = tile_num(file2)

            # Check if tiles have same index
            if f1 == f2:

                # Save if true
                f1out.append(file1)
                f2out.append(file2)
                break

    return f1out, f2out

# Read in parameters
args   = get_args()
ifiles = args.input[:]
ofile_ = args.output[0]
radius = args.radius[0]
proj   = args.proj[0]
dxy    = args.dxy[0]
nres_a = args.nres[0]
nres_d = args.nres[1]
buff   = args.buff[0]
mode   = args.mode[0]
vnames = args.vnames[:]
tspan  = args.tspan[:]
tile   = args.tile
plot   = args.plot
diff   = args.diff

print('parameters:')
for arg in vars(args).items(): print(arg)

# Get variable names
ovar, xvar, yvar, tvar, zvar, bvar, lvar, svar = vnames

# Create output names
oovar_a = ovar+'_1'
oovar_d = ovar+'_2'
oxvar_x = xvar
oyvar_x = yvar
otvar_a = tvar+'_1'
otvar_d = tvar+'_2'
otvar_x = 'd'+tvar
ozvar_a = zvar+'_1'
ozvar_d = zvar+'_2'
ozvar_x = 'd'+zvar

obvar_a = bvar+'_1'
obvar_d = bvar+'_2'
obvar_x = 'd'+bvar
olvar_a = lvar+'_1'
olvar_d = lvar+'_2'
olvar_x = 'd'+lvar
osvar_a = svar+'_1'
osvar_d = svar+'_2'
osvar_x = 'd'+svar

# Test names
if ozvar_x == osvar_x or ozvar_x == obvar_x or ozvar_x == olvar_x:
    print("****************************************************************")
    print("Extra parameters can't have the same name as the main parameter!")
    print("****************************************************************")
    sys.exit()

# Test for stereographic
if proj != "4326":

    # Convert to meters
    dxy *= 1e3
else:
    print("Warning you are using EPSG 4326 which can have issues! Please Polar Stereographic Projections.")
    
def main(ifile1, ifile2):
    """ Find and compute crossover values. """
    
    # Start time of program
    startTime = datetime.now()

    print('crossing files:', ifile1, ifile2, '...')

    # Load all 1d variables needed
    with h5py.File(ifile1, 'r') as f1, \
         h5py.File(ifile2, 'r') as f2:
             
        # File 1
        orbit1  = f1[ovar][:]
        lon1    = f1[xvar][:]
        lat1    = f1[yvar][:]
        time1   = f1[tvar][:]
        height1 = f1[zvar][:]
        bs1     = f1[bvar][:] if bvar in f1 else np.zeros(lon1.shape)
        lew1    = f1[lvar][:] if lvar in f1 else np.zeros(lon1.shape)
        tes1    = f1[svar][:] if svar in f1 else np.zeros(lon1.shape)
        
        # File 2
        orbit2  = f2[ovar][:]
        lon2    = f2[xvar][:]
        lat2    = f2[yvar][:]
        time2   = f2[tvar][:]
        height2 = f2[zvar][:]
        bs2     = f2[bvar][:] if bvar in f2 else np.zeros(lon2.shape)
        lew2    = f2[lvar][:] if lvar in f2 else np.zeros(lon2.shape)
        tes2    = f2[svar][:] if svar in f2 else np.zeros(lon2.shape)
        
        # File 1
        orbit1  = orbit1[np.isfinite(height1)]
        lon1    = lon1[np.isfinite(height1)]
        lat1    = lat1[np.isfinite(height1)]
        time1   = time1[np.isfinite(height1)]
        bs1     = bs1[np.isfinite(height1)]
        lew1    = lew1[np.isfinite(height1)]
        tes1    = tes1[np.isfinite(height1)]
        height1 = height1[np.isfinite(height1)]
        
        # File 2
        orbit2  = orbit2[np.isfinite(height2)]
        lon2    = lon2[np.isfinite(height2)]
        lat2    = lat2[np.isfinite(height2)]
        time2   = time2[np.isfinite(height2)]
        bs2     = bs2[np.isfinite(height2)]
        lew2    = lew2[np.isfinite(height2)]
        tes2    = tes2[np.isfinite(height2)]
        height2 = height2[np.isfinite(height2)]
        
        # Set flags for extra parameters
        if np.all(bs1 == 0):
            flag_bs = False
        else:
            flag_bs = True
        if np.all(lew1 == 0):
            flag_le = False
        else:
            flag_le = True
        if np.all(tes1 == 0):
            flag_ts = False
        else:
            flag_ts = True

    # If time span given, filter out invalid data
    if tspan[0] != None:

        t1, t2 = tspan

        idx, = np.where((time1 >= t1) & (time1 <= t2))
        orbit1 = orbit1[idx]
        lon1 = lon1[idx]
        lat1 = lat1[idx]
        time1 = time1[idx]
        height1 = height1[idx]
        bs1 = bs1[idx]
        lew1 = lew1[idx]
        tes1 = tes1[idx]
        
        idx, = np.where((time2 >= t1) & (time2 <= t2))
        orbit2 = orbit2[idx]
        lon2 = lon2[idx]
        lat2 = lat2[idx]
        time2 = time2[idx]
        height2 = height2[idx]
        bs2 = bs2[idx]
        lew2 = lew2[idx]
        tes2 = tes2[idx]
        
        if len(time1) < 3 or len(time2) < 3:
            print('no points within time-span!')
            sys.exit()

    # Transform to wanted coordinate system
    (xp1, yp1) = transform_coord(4326, proj, lon1, lat1)
    (xp2, yp2) = transform_coord(4326, proj, lon2, lat2)
    
    # Time limits: the largest time span (yr)
    tmin = min(np.nanmin(time1), np.nanmin(time2))
    tmax = max(np.nanmax(time1), np.nanmax(time2))

    # Boundary limits: the smallest spatial domain (m)
    xmin = max(np.nanmin(xp1), np.nanmin(xp2))
    xmax = min(np.nanmax(xp1), np.nanmax(xp2))
    ymin = max(np.nanmin(yp1), np.nanmin(yp2))
    ymax = min(np.nanmax(yp1), np.nanmax(yp2))

    # Interpolation type and number of needed points
    if mode == "linear":

        # Linear interpolation
        nobs  = 2
        order = 1

    else:

        # Cubic interpolation
        nobs  = 6
        order = 3

    # Tiling option - "on" or "off"
    if dxy:
        
        print('tileing asc/des data...')
        
        # Get bounding box
        bboxs1 = get_bboxs(xp1, yp1, xmin, xmax, ymin, ymax, dxy, buff*1e3)
        bboxs2 = get_bboxs(xp2, yp2, xmin, xmax, ymin, ymax, dxy, buff*1e3)

        # Copy box for conviniance
        bboxs = bboxs1

    else:
        
        # Get bounding box from full domain
        bboxs = [(xmin, xmax, ymin, ymax)]

    # Start time of program
    startTime = datetime.now()

    # Initiate output container
    out = []
    
    # Initiate xover counter
    i_xover = 0

    # Counter
    ki = 0

    # Unique boxes
    ibox = np.unique(bboxs)

    print('computing crossovers ...')
    
    # Loop through each sub-tile
    for k in ibox:
        
        # Get the tile indices
        idx1, = np.where(bboxs1 == k)
        idx2, = np.where(bboxs2 == k)

        # Extract tile data from each set
        orbits1 = orbit1[idx1]
        lons1 = lon1[idx1]
        lats1 = lat1[idx1]
        x1 = xp1[idx1]
        y1 = yp1[idx1]
        h1 = height1[idx1]
        t1 = time1[idx1]
        b1 = bs1[idx1]
        l1 = lew1[idx1]
        s1 = tes1[idx1]
        
        orbits2 = orbit2[idx2]
        lons2 = lon2[idx2]
        lats2 = lat2[idx2]
        x2 = xp2[idx2]
        y2 = yp2[idx2]
        h2 = height2[idx2]
        t2 = time2[idx2]
        b2 = bs2[idx2]
        l2 = lew2[idx2]
        s2 = tes2[idx2]

        # Get unique orbits
        orb_ids1 = np.unique(orbits1)
        orb_ids2 = np.unique(orbits2)

        # Test if tile has no crossovers
        if len(orbits1) == 0 or len(orbits2) == 0:
            
            # Go to next track
            continue

        # Loop through orbits from file #1
        for orb_id1 in orb_ids1:
            
            # Index for single ascending orbit
            i_trk1 = orbits1 == orb_id1 

            # Extract points from single orbit (a track)
            xa = x1[i_trk1]
            ya = y1[i_trk1]
            ta = t1[i_trk1]
            ha = h1[i_trk1]
            ba = b1[i_trk1]
            la = l1[i_trk1]
            sa = s1[i_trk1]
            
            # Loop through tracks from file #2
            for orb_id2 in orb_ids2:

                # Index for single descending orbit
                i_trk2 = orbits2 == orb_id2

                # Extract single orbit
                xb = x2[i_trk2]
                yb = y2[i_trk2]
                tb = t2[i_trk2]
                hb = h2[i_trk2]
                bb = b2[i_trk2]
                lb = l2[i_trk2]
                sb = s2[i_trk2]
                
                # Test length of vector
                if len(xa) < 3 or len(xb) < 3: continue

                # Compute exact crossing location
                cxy_main = intersect(xa[::nres_a], ya[::nres_a], \
                                     xb[::nres_d], yb[::nres_d])
                
                # Test again for crossing
                if len(cxy_main) == 0: continue

                """
                    SUPPORT SHOULD BE ADDED FOR MULTIPLE CROSSOVERS FOR SAME TRACK!

                """

                # Extract crossing coordinates
                xi = cxy_main[0][0]
                yi = cxy_main[0][1]
                
                # Get start coordinates of orbits
                xa0 = xa[0]
                ya0 = ya[0]
                xb0 = xb[0]
                yb0 = yb[0]

                # Compute distance from crossing node to each arc
                da = (xa - xi) * (xa - xi) + (ya - yi) * (ya - yi)
                db = (xb - xi) * (xb - xi) + (yb - yi) * (yb - yi)

                # Sort according to distance
                Ida = np.argsort(da)
                Idb = np.argsort(db)

                # Sort arrays - A
                xa = xa[Ida]
                ya = ya[Ida]
                ta = ta[Ida]
                ha = ha[Ida]
                da = da[Ida]
                ba = ba[Ida]
                la = la[Ida]
                sa = sa[Ida]
                
                # Sort arrays - B
                xb = xb[Idb]
                yb = yb[Idb]
                tb = tb[Idb]
                hb = hb[Idb]
                db = db[Idb]
                bb = bb[Idb]
                lb = lb[Idb]
                sb = sb[Idb]
                
                # Get distance of four closest observations
                dab = np.vstack((da[[0, 1]], db[[0, 1]]))

                # Test if any point is too far away
                if np.any(np.sqrt(dab) > radius):
                    continue
                # Test if enough obs. are available for interpolation
                elif (len(xa) < nobs) or (len(xb) < nobs):
                    continue
                else:
                    # Accepted
                    pass
            
                # Compute distance again from the furthest point
                da0 = (xa - xa0) * (xa - xa0) + (ya - ya0) * (ya - ya0)
                db0 = (xb - xb0) * (xb - xb0) + (yb - yb0) * (yb - yb0)

                # Compute distance again from the furthest point
                dai = (xi - xa0) * (xi - xa0) + (yi - ya0) * (yi - ya0)
                dbi = (xi - xb0) * (xi - xb0) + (yi - yb0) * (yi - yb0)
                
                # Interpolate height to crossover location
                hai = interp1D(da0[0:nobs], ha[0:nobs], dai, order)
                hbi = interp1D(db0[0:nobs], hb[0:nobs], dbi, order)
                
                # Interpolate time to crossover location
                tai = interp1D(da0[0:nobs], ta[0:nobs], dai, order)
                tbi = interp1D(db0[0:nobs], tb[0:nobs], dbi, order)

                # Distance between to closes points
                d_a = np.sqrt((xa[0]-xa[1])**2 + (ya[0]-ya[1])**2)
                d_b = np.sqrt((xa[0]-xa[1])**2 + (ya[0]-ya[1])**2)

                # Height difference between closest points
                h_a = np.abs(ha[0] - ha[1])
                h_b = np.abs(hb[0] - hb[1])

                # Slope in degrees from closest points
                s_a = np.arctan(h_a / d_a) * 180 / np.pi
                s_b = np.arctan(h_b / d_b) * 180 / np.pi

                # Test interpolate time values
                if (tai > tmax) or (tai < tmin) or \
                       (tbi > tmax) or (tbi < tmin):
                    continue
                
                # Create output array
                out_i = np.full(22, np.nan)
                
                # Compute differences and save parameters
                out_i[0]  = xi
                out_i[1]  = yi
                out_i[2]  = hai - hbi
                out_i[3]  = tai - tbi
                out_i[4]  = tai
                out_i[5]  = tbi
                out_i[6]  = hai
                out_i[7]  = hbi
                out_i[8]  = (hai - hbi) / (tai - tbi)
                out_i[18] = orb_id1
                out_i[19] = orb_id2
                out_i[20] = s_a
                out_i[21] = s_b

                # Test for more parameters to difference
                if flag_bs:
                    
                    if diff is True:
                        
                        # Save paramters
                        bai = ba[0]
                        bbi = bb[0]
                        
                        # Save difference
                        out_i[9]  = bai - bbi
                        out_i[10] = bai
                        out_i[11] = bbi

                    else:
                        
                        # Interpolate sigma0 to crossover location
                        bai = interp1D(da0[0:nobs], ba[0:nobs], order)
                        bbi = interp1D(db0[0:nobs], bb[0:nobs], order)
                        
                        # Save difference
                        out_i[9]  = bai - bbi
                        out_i[10] = bai
                        out_i[11] = bbi

                if flag_le:
                    
                    if diff is True:

                        # Get parameters
                        lai = la[0]
                        lbi = lb[0]
                        
                        # Save difference
                        out_i[12] = lai - lbi
                        out_i[13] = lai
                        out_i[14] = lbi
               
                    else:
                        
                        # Interpolate leading edge width to crossover location
                        lai = interp1D(da0[0:nobs], la[0:nobs], order)
                        lbi = interp1D(db0[0:nobs], lb[0:nobs], order)
                        
                        # Save difference
                        out_i[12] = lai - lbi
                        out_i[13] = lai
                        out_i[14] = lbi

                if flag_ts:
                    
                    if diff is True:
                        
                        # Get parameters
                        sai = sa[0]
                        sbi = sb[0]
                        
                        # Save difference
                        out_i[15] = sai - sbi
                        out_i[16] = sai
                        out_i[17] = sbi

                    else:
                        
                        # Interpolate trailing edge slope to crossover location
                        sai = interp1D(da0[0:nobs], sa[0:nobs], order)
                        sbi = interp1D(db0[0:nobs], sb[0:nobs], order)
                        
                        # Save difference
                        out_i[15] = sai - sbi
                        out_i[16] = sai
                        out_i[17] = sbi
                        
                # Add to list
                out.append(out_i)
                
        # Operating on current tile
        # print('tile:', ki, len(ibox))

        # Update counter
        ki += 1

    # Change back to numpy array
    out = np.asarray(out)

    # Remove invalid rows
    out = out[~np.isnan(out[:,2]),:]

    # Test if output container is empty 
    if len(out) == 0:
        print('no crossovers found!')
        return

    # Remove the two id columns if they are empty 
    out = out[:,:-2] if np.isnan(out[:,-1]).all() else out

    # Copy ouput arrays 
    xs, ys = out[:,0].copy(), out[:,1].copy()

    # Transform coords back to lat/lon
    out[:,0], out[:,1] = transform_coord(proj, '4326', out[:,0], out[:,1])

    # Create output file name if not given
    if ofile_ is None:
        path, ext = os.path.splitext(ifile1)
        if tile:
            tilenum = str(tile_num(ifile1))
        else:
            tilenum = '' 
        if ifile1.find('_A_') > 0:
            fnam = '_XOVERS_AD_' 
        else:
            fnam = '_XOVERS_DA_'
        ofile = path + fnam + tilenum + ext
    else:
        ofile = ofile_

    # Create h5 file
    with h5py.File(ofile, 'w') as f:
        
        # Add standard parameters
        f[oxvar_x] = out[:,0]
        f[oyvar_x] = out[:,1]
        f[ozvar_x] = out[:,2]
        f[otvar_x] = out[:,3]
        f[otvar_a] = out[:,4]
        f[otvar_d] = out[:,5]
        f[ozvar_a] = out[:,6]
        f[ozvar_d] = out[:,7]
        f[oovar_a] = out[:,18]
        f[oovar_d] = out[:,19]
        f['dhdt']  = out[:,8]
        f['slp_1'] = out[:,20]
        f['slp_2'] = out[:,21]

        # Add extra parameters
        if flag_bs:
            f[obvar_x] = out[:,9]
            f[obvar_a] = out[:,10]
            f[obvar_d] = out[:,11]
        if flag_le:
            f[olvar_x] = out[:,12]
            f[olvar_a] = out[:,13]
            f[olvar_d] = out[:,14]
        if flag_ts:
            f[osvar_x] = out[:,15]
            f[osvar_a] = out[:,16]
            f[osvar_d] = out[:,17]
                
    # Stat. variables
    dh   = out[:,2]
    dt   = out[:,3]
    dhdt = out[:,8]

    # Compute statistics
    med0 = np.around(np.median(dh[np.abs(dt)<=1./12]),3)
    std0 = np.around(mad_std(dh[np.abs(dt)<=1./12]),3)
    med1 = np.around(np.median(dhdt),3)
    std1 = np.around(mad_std(dhdt),3)
    
    # Print some statistics to screen
    print('')
    print('execution time: ' + str(datetime.now() - startTime))
    print('number of crossovers found:',str(len(out)))
    print('statistics -> mean:',med0,'std.dev:',std0, '(m) (dt<30d)')
    print('statistics -> mean:',med1,'std.dev:',std1, '(dvar/yr)')
    print('ofile name ->', ofile)

    if plot:

        # Some light filtering for plotting
        io = np.abs(dh) < 3.0*mad_std(dh)
        
        gridsize = (3, 2)
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid(gridsize, (2, 0))
        ax3 = plt.subplot2grid(gridsize, (2, 1))
        # Plot main difference variable
        ax1.scatter(xs*1e-3,ys*1e-3,s=10,c=dh,cmap='jet',vmin=-20,vmax=20)
        #plt.subplot(211)
        ax2.plot(dt[io], dh[io], '.')
        ax2.set_ylabel(ozvar_x)
        ax2.set_xlabel(otvar_x)
        #plt.subplot(212)
        ax3.hist(dh[io],50)
        ax3.set_ylabel('Frequency')
        ax3.set_xlabel(ozvar_x)
        #plt.subplots_adjust(hspace=.5)
        plt.show()

#
# Running main program!
#

# Read file names
str1, str2 = ifiles

# Check for tile mode
if tile:
        
    # Get matching tiles
    files1, files2 = match_tiles(str1, str2, 'tile')

    # Loop trough tiles
    for i in range(len(files1)):
            
        # Run main program
        main(files1[i], files2[i])

# Run as single files
else:
        
    # File names
    file1, file2 = str1, str2
            
    # Run main program
    main(file1, file2)











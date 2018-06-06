#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial Optimal Interpolation using modeled covariance function.

Example:
     python ointerp.py ~/data/ers1/floating/filt_scat_det/joined_pts_a.h5_ross -d 3 3 -n 100 -r 5 -e .1 -v t_year lon lat h_res None

"""

import sys
import h5py
import pyproj
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
from numba import jit, int32, float64
from scipy.spatial.distance import cdist, pdist, squareform

import matplotlib.pyplot as plt

#-----------------------------------------------------------

""" Covariance models. """

# Modeled parameters
#s, R = [0.831433532, 1536.21967]  # gauss
s, R = [0.85417087, 655.42437697]  # markov
#s, R = [0.720520448, 1084.11029]  # generic

tau = 0.25

def gauss(r, s, R):
    return s**2 * np.exp(-r**2/R**2)

def markov(r, s, R):
    return s**2 * (1 + r/R) * np.exp(-r/R)

def generic(r, s, R):
    return s**2 * (1 + (r/R) - 0.5 * (r/R)**2) * np.exp(-r/R)

def exp(t, tau):
    return np.exp(-t**2/tau**2)

def covxt(r, t, s, R, tau):
    """ C(r,t) = C(r) * C(t). """
    return markov(r, s, R) * exp(t, tau) 

covmodel = markov
#covmodel = covxt

#----------------------------------------------------------


def get_args():
    """ Get command-line arguments. """

    des = 'Optimal Interpolation of spatial data'
    parser = argparse.ArgumentParser(description=des)

    parser.add_argument(
            'ifile', metavar='ifile', type=str, nargs='+',
            help='name of i-file, numpy binary or ascii (for binary ".npy")')

    parser.add_argument(
            '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
            help='name of o-file, numpy binary or ascii (for binary ".npy")',
            default=['ointerp.h5']) #FIXME

    parser.add_argument(
            '-i', metavar=('w','e','s','n'), dest='bbox', type=float, nargs=4,
            help=('bounding box for geograph. region (deg or m), optional'),
            default=[],)

    parser.add_argument(
            '-d', metavar=('dx','dy'), dest='dxy', type=float, nargs=2,
            help=('spatial resolution for grid (deg or km)'),
            default=[1, 1],)

    parser.add_argument(
            '-n', metavar='nobs', dest='nobs', type=int, nargs=1,
            help=('number of obs. for each quadrant'),
            default=[1],)

    parser.add_argument(
            '-r', metavar='radius', dest='radius', type=float, nargs=1,
            help=('maximum search radius (km)'),
            default=[1],)

    parser.add_argument(
            '-a', metavar='alpha', dest='alpha', type=float, nargs=1,
            help=('correlation length (km)'),
            default=[1],)

    parser.add_argument(
            '-e', metavar='sigma', dest='sigma', type=float, nargs=1,
            help=('rms noise of obs. (m)'),
            default=[0],)

    parser.add_argument(
            '-f', metavar=('min','max','tol','thrs'), dest='filt', type=float, nargs=4,
            help=('reject obs: obs_min, obs_max, tolerance (pct), N*rms'),
            default=[-9999,9999,5,3],)

    parser.add_argument(
            '-v', metavar=('time', 'lon','lat', 'obs', 'err'), dest='vnames',
            type=str, nargs=5,
            help=('name of t/x/y/z/e variables in the HDF5. If err=None, skip'),
            default=[None], required=True)

    parser.add_argument(
            '-p', metavar=('epsg_num'), dest='proj', type=str, nargs=1,
            help=('EPSG proj number (AnIS=3031, GrIS=3413)'),
            default=['3031'],)

    parser.add_argument(
            '-s', metavar=None, dest='select', type=str, nargs=1,
            help=('sampling mode: random (r) or distance (s).'),
            choices=('r', 's'), default=['s'],)

    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def iterfilt(x, xmin, xmax, tol, alpha):
    """ Iterative outlier filter """
    
    # Set default value
    tau = 100.0
    
    # Remove data outside selected range
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan
    
    # Initiate counter
    k = 0
    
    # Outlier rejection loop
    while tau > tol:
        
        # Compute initial rms
        rmse_b = mad_std(x)
        
        # Compute residuals
        dh_abs = np.abs(x - np.nanmedian(x))
        
        # Index of outliers
        io = dh_abs > alpha * rmse_b
        
        # Compute edited rms
        rmse_a = mad_std(x[~io])
        
        # Determine rms reduction
        tau = 100.0 * (rmse_b - rmse_a) / rmse_a
        
        # Remove data if true
        if tau > tol or k == 0:
            
            # Set outliers to NaN
            x[io] = np.nan
            
            # Update counter
            k += 1
    
    return x


def rand(x, n):
    """Draws random samples from array"""

    # Determine data density
    if len(x) > n:

        # Draw random samples from array
        I = np.random.choice(np.arange(len(x)), n, replace=False)
    
    else:

        # Output boolean vector - true
        I = np.ones(len(x), dtype=bool)

    return I


def sort_dist(d, n):
    """ Sort array by distance"""
    
    # Determine if sorting needed
    if len(d) >= n:
        
        # Sort according to distance
        I = np.argsort(d)
    
    else:
        
        # Output boolean vector - true
        I = np.ones(len(x), dtype=bool)

    return I


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Stereo AnIS (x/y):  3031
        Stereo GrIS (x/y):  3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


def get_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Generate a regular grid. """

    # Setup grid dimensions
    Nx = int((np.abs(xmax - xmin)) / dx) + 1
    Ny = int((np.abs(ymax - ymin)) / dy) + 1

    # Initiate lat/lon vectors for grid
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)

    # Construct output grid-coordinates
    return np.meshgrid(x, y)


def segment_number(time, tmax=1):
    """
    Partition time array into segments with breaks > tmax.

    Returns an array w/unique identifiers for each segment.
    """
    n = 0
    trk = np.zeros(time.shape)
    for k in xrange(1, len(time)):
        if np.abs(time[k]-time[k-1]) > tmax:
            n += 1
        trk[k] = n
    return trk


""" Compiled functions. """

@jit(nopython=True)
def add_offdiag_error(A, B, C, err):
    """
    Add correlated (off-diagonal) errors to C.

    If i,j belong to the same track (aij == bij)
    and they are not in the diagonal (i != j), then:
        cij += sigma
    """
    M, N = A.shape
    for i in range(M):
        for j in range(N):
            aij = A[i,j]
            bij = B[i,j]
            if i != j and aij == bij:
                C[i,j] += err

#-------------

""" Helper functions. """

def filter_domain(t, x, y, z, e, b):
    """ Reduce domain size for testing purposes. """
    xx = (x > -160) & (x < 0)          # space
    tt = (t > 1995) & (t < 1995.5)  # time
    ii, = np.where(xx & tt)
    #ii, = np.where(tt)
    return t[ii], x[ii], y[ii], z[ii], e[ii], b[ii]


def filter_invalid(t, x, y, z, e, b, step=1):
    """ Mask NaNs and Zeros. """
    bb = (b != 0) & ~np.isnan(b)
    kk = (z != 0) & ~np.isnan(z)
    ii, = np.where(bb & kk)
    return t[ii], x[ii], y[ii], z[ii], e[ii], b[ii]


# Parser argument to variable
args = get_args() 

# Read input from terminal
ifile = args.ifile[0]
ofile = args.ofile[0]
bbox = args.bbox
dx = args.dxy[0] * 1e3
dy = args.dxy[1] * 1e3
proj = args.proj[0]
nobs = args.nobs[0]
radius = args.radius[0] * 1e3
alpha = args.alpha[0] * 1e3
sigma = args.sigma[0]
tvar = args.vnames[0]
xvar = args.vnames[1]
yvar = args.vnames[2]
zvar = args.vnames[3]
evar = args.vnames[4]
zmin = args.filt[0]
zmax = args.filt[1]
tol = args.filt[2]
thres = args.filt[3]
selec = args.select[0]

# Print parameters to screen
print_args(args)

# Start timing of script
startTime = datetime.now()

print "reading input file ..."

with h5py.File(ifile, 'r') as f:

    step = 2
    time = f[tvar][::step]
    lon = f[xvar][::step]
    lat = f[yvar][::step]
    obs = f[zvar][::step]
    err = f[evar][::step] if evar != 'None' else np.full_like(obs, None)
    scat = f['h_bs'][::step]


# Only for time-dependent data
if 1:
    time, lon, lat, obs, err, scat = filter_domain(time, lon, lat, obs, err, scat)
    time, lon, lat, obs, err, scat = filter_invalid(time, lon, lat, obs, err, scat)


# Remove mean (background field)
#obs -= np.nanmean(obs)  ##FIXME

# Convert to stereo coordinates
xp, yp = transform_coord(4326, proj, lon, lat)

# Extract observations
zp = obs

# Assign a track ID to each data point
tsec = time * 3.154e7  # year -> sec
trk = segment_number(tsec, tmax=100)

# Plot
if 0:
    '''
    plt.scatter(xp, yp, c=obs, s=1, rasterized=True,
            vmin=-np.nanstd(obs)/2., vmax=np.nanstd(obs)/2.,
            cmap=plt.cm.RdBu)
    plt.colorbar()
    '''
    trk_unique = np.unique(trk)
    for k in trk_unique:
        ii, = np.where(k == trk)
        obs = obs
        xp_ = xp[ii]
        yp_ = yp[ii]
        plt.plot(xp_, yp_, '.', rasterized=True)

    plt.show()
    sys.exit()


#FIXME: Check what this is doing!
# Test for different types of input
if len(bbox) == 6:
    # Extract bounding box elements
    (xmin, xmax, ymin, ymax) = bbox
else:
    # Create bounding box limits
    (xmin, xmax, ymin, ymax) = (xp.min() - 10.0*dx), (xp.max() + \
            10.0*dx), (yp.min() - 10.0*dy), (yp.max() + 10.0*dy)

# Generate prediction grid     
Xi, Yi = get_grid(xmin, xmax, ymin, ymax, dx, dy)

# Flatten prediction grid
xi, yi = Xi.ravel(), Yi.ravel()

# Geographical projection
if np.abs(ymax) < 100:
    
    # Convert to stereographic coord.
    (xi, yi) = pyproj.transform(projGeo, projGrd, xi, yi)

print "setting up kd-tree ..."

# Construct cKDTree - points
TreeP = cKDTree(zip(xp, yp))


# Compute noise variance
sigma2 = sigma * sigma

# Compute Long-wavelength (along-track) error
sigma2_L = sigma2 * 0.5

# Output vectors
zi = np.ones(len(xi)) * np.nan
ei = np.ones(len(xi)) * np.nan
ni = np.ones(len(xi)) * np.nan


##TODO: Make a parallel version here!
print "looping grid nodes ..."

# Enter prediction loop
for i in xrange(len(xi)):
    
    # Get indexes from Tree 
    idx = TreeP.query_ball_point((xi[i], yi[i]), radius)
    
    # Compute distance between prediction (grid) pt and obs within search radius
    dxj = np.sqrt((xp[idx] - xi[i]) * (xp[idx] - xi[i]) + (yp[idx] - yi[i]) * (yp[idx] - yi[i]))
                    
    # Test for empty cell
    if len(zp[idx]) == 0: continue
    #if len(zp[idx]) < 100: continue  ##FIXME: Only for testing (with plot below)
    
    # Obs. before editing
    nb = len(zp[idx])

    # Quick outlier editing
    Io = ~np.isnan(iterfilt(zp[idx].copy(), zmin, zmax, tol, thres))

    ##HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Parameters
    x = xp[idx][Io]
    y = yp[idx][Io]
    z = zp[idx][Io]
    k = trk[idx][Io]

    # Plot data within search radius
    if 0:
        plt.figure()
        plt.scatter(x, y, c=z, s=50, cmap=plt.cm.RdBu)

        plt.figure()
        k_unique = np.unique(k)
        for ki in k_unique:
            ii, = np.where(ki == k)
            plt.plot(np.hypot(x[ii], y[ii]), z[ii], 'o')
            plt.xlabel('Distance along track (m)')

        plt.show()
        continue
    
    # Distance editing
    dxj = dxj[Io]
                    
    # Obs. after editing
    na = len(z)

    # Outliers removed
    dn = nb - na

    # Test for empty cell
    if len(z) == 0: continue

    """ Build Error matrix. """

    # Build error matrix 
    if np.isnan(err).all():

        # Provide all obs. the same sigma
        c = np.full(x.shape[0], sigma2)

    else:

        # Set all obs. errors < sigma2 to sigma2
        c = err[Io]**2
        c[c<sigma2] = sigma2

    # Select data within search radius with sampling strategy
    if 0:

        """
        # Compute angle to data points
        theta = (180.0 / np.pi) * np.arctan2(y - yi[i], x - xi[i]) + 180

        # Get index for data in 8-sectors
        IQ1 = (theta > 0) & (theta < 45)
        IQ2 = (theta > 45) & (theta < 90)
        IQ3 = (theta > 90) & (theta < 135)
        IQ4 = (theta > 135) & (theta < 180)
        IQ5 = (theta > 180) & (theta < 225)
        IQ6 = (theta > 225) & (theta < 270)
        IQ7 = (theta > 270) & (theta < 315)
        IQ8 = (theta > 315) & (theta < 360)

        # Merge all data to sectors
        Q1 = np.vstack((x[IQ1], y[IQ1], z[IQ1], c[IQ1], dxj[IQ1])).T
        Q2 = np.vstack((x[IQ2], y[IQ2], z[IQ2], c[IQ2], dxj[IQ2])).T
        Q3 = np.vstack((x[IQ3], y[IQ3], z[IQ3], c[IQ3], dxj[IQ3])).T
        Q4 = np.vstack((x[IQ4], y[IQ4], z[IQ4], c[IQ4], dxj[IQ4])).T
        Q5 = np.vstack((x[IQ5], y[IQ5], z[IQ5], c[IQ5], dxj[IQ5])).T
        Q6 = np.vstack((x[IQ6], y[IQ6], z[IQ6], c[IQ6], dxj[IQ6])).T
        Q7 = np.vstack((x[IQ7], y[IQ7], z[IQ7], c[IQ7], dxj[IQ7])).T
        Q8 = np.vstack((x[IQ8], y[IQ8], z[IQ8], c[IQ8], dxj[IQ8])).T
        
        # Determine sampling strategy
        if selec == 'r':
            
            # Draw random samples from each sector
            I1 = rand(Q1[:, 0], nobs)
            I2 = rand(Q2[:, 0], nobs)
            I3 = rand(Q3[:, 0], nobs)
            I4 = rand(Q4[:, 0], nobs)
            I5 = rand(Q5[:, 0], nobs)
            I6 = rand(Q6[:, 0], nobs)
            I7 = rand(Q7[:, 0], nobs)
            I8 = rand(Q8[:, 0], nobs)

        else:
            
            # Draw closest samples from each sector
            I1 = rand(Q1[:, 4], nobs)
            I2 = rand(Q2[:, 4], nobs)
            I3 = rand(Q3[:, 4], nobs)
            I4 = rand(Q4[:, 4], nobs)
            I5 = rand(Q5[:, 4], nobs)
            I6 = rand(Q6[:, 4], nobs)
            I7 = rand(Q7[:, 4], nobs)
            I8 = rand(Q8[:, 4], nobs)

        # Stack the data
        Q18 = np.vstack((Q1[I1, :], Q2[I2, :], Q3[I3, :], Q4[I4, :],
            Q5[I5, :], Q6[I6, :], Q7[I7, :], Q8[I8, :]))
        """

    # Use all the data within search radius w/o specific sampling
    else:

        X = np.column_stack((x, y, z, c, dxj, k))

    # Information of points within inversino cell
    xc = X[:,0]  # x-coord of pts
    yc = X[:,1]  # y-coord of pts
    zc = X[:,2]  # value of each pt
    ec = X[:,3]  # error of each pt
    dc = X[:,4]  # distance to cell center
    kc = X[:,5]  # track id of each pt


    # Estimate local median (robust) and local variance of data
    m0 = np.median(zc)
    c0 = np.var(zc) 


    if 0:
        pass

    else:

        # Scaling factor to convert: global cov -> local cov
        scale = c0/covmodel(0, s, R)

        # Distance from grid node (model) to data => vector
        Dxj = X[:, 4]

        # Covariance vector: model-data 
        Cxj = covmodel(Dxj, s, R) * scale

        # Compute pair-wise distance => matrix
        Dij = cdist(zip(xc, yc), zip(xc, yc), "euclidean")
        
        # Covariance matrix: data-data 
        Cij = covmodel(Dij, s, R) * scale

    ######

    if 0:
        plt.plot(Dxj, Cxj, 'o')
        plt.show()
        continue

    n_obs = zc.shape[0]

    # Uncorrelated errors
    # (diagonal: variance of uncorrelated white noise)
    Nij = np.diag(ec)  

    # Matrices with track id for each data point
    Kx, Ky = np.meshgrid(kc, kc)

    # Plot error matrix w/diagonal only
    if 0:
        plt.figure()
        plt.imshow(Nij)

    # Correlated errors
    # (off-diagonal: variance of along-track long-wavelength error)
    if 1:
        add_offdiag_error(Kx, Ky, Nij, sigma2_L)

    # Plot error matrix w/off-diagonal entries
    if 0:
        plt.figure()
        plt.imshow(Nij)
        plt.show()
        continue

    # Augmented data-cov matrix w/errors
    Aij = Cij + Nij

    if 1:

        # Matrix inversion of: Cxj * Aij^(-1)
        CxjAiji = np.linalg.solve(Aij.T, Cxj.T)

        # Predicted value
        zi[i] = np.dot(CxjAiji, zc) + (1 - np.sum(CxjAiji)) * m0
        
        # Predicted error
        ei[i] = np.sqrt(np.abs(c0 - np.dot(CxjAiji, Cxj.T)))
        
        # Number of data used for prediction    
        ni[i] = len(zc)

    else:
        ##NOTE: Need to doulbe check this is correct.

        L = np.linalg.cholesky(Aij)

        B = np.linalg.solve(L, Cxj.T)
        #B = np.dot(np.linalg.inv(L), Cxj.T)

        y = np.linalg.solve(L, zc)
        #y = np.dot(np.linalg.inv(L), zc)

        zi[i] = np.dot(B.T, y) + (1 - np.sum(B)) * m0
        ei[i] = s - np.dot(B.T, B) 

        ni[i] = len(zc)

    # Print progress to terminal
    if (i % 500) == 0:
        
        # N-predicted values
        print str(i) + '/' + str(len(xi)) \
                + ' Pred: ' + str(np.around(zi[i],2)) \
                + '  Nsol: '+ str(ni[i]) \
                + '  Dmax: ' + str(np.around(1e-3 * dxj.max(),2))

# Convert back to arrays
Zi = np.flipud(zi.reshape(Xi.shape))
Ei = np.flipud(ei.reshape(Xi.shape))
Ni = np.flipud(ni.reshape(Xi.shape))

# Flip coordinates
Xi = np.flipud(Xi)
Yi = np.flipud(Yi)

# Save prediction
if 0:
    with h5py.File(ofile, 'w') as f:
        f['x'] = Xi
        f['y'] = Yi
        f['z'] = Zi
        f['e'] = Ei
        f['n'] = Ni

#################################

from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage as ni

Zi = np.ma.masked_invalid(Zi)

if 1: # smooth and interpolate before regridding
    #gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    #Zi = convolve(Zi, gauss_kernel, boundary='wrap', normalize_kernel=True)
    Zi = ni.median_filter(Zi, 3)

vmin = -np.nanstd(Zi) * .5
vmax = np.nanstd(Zi) * .5

plt.figure()
plt.pcolormesh(Xi, Yi, Zi, vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)

plt.figure()
plt.contourf(Xi, Yi, Zi, 60, vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)

plt.show()

"""
# Set output names
OFILE_1 = ofile[:-4]+'_PRED'+'.tif'
OFILE_2 = ofile[:-4]+'_RMSE'+'.tif'
OFILE_3 = ofile[:-4]+'_NSOL'+'.tif'

print "saving data ..."

# Write data to geotiff-format
geotiffwrite(OFILE_1, Xi, Yi, Zi, dx, dy, int(proj), "float")
geotiffwrite(OFILE_2, Xi, Yi, Ei, dx, dy, int(proj), "float")
geotiffwrite(OFILE_3, Xi, Yi, Ni, dx, dy, int(proj), "float")
"""

# Print execution time of script
print 'Execution time: '+ str(datetime.now()-startTime)

"""

	Helper functions for altimetry algorithms for captoolkit

"""
import xarray as xr
import h5py
import numpy as np
import pyproj
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.ndimage import map_coordinates
from scipy import signal
from scipy.linalg import solve
from affine import Affine
from scipy.interpolate import interp1d
from numba import jit
from numpy.linalg import pinv
from gdalconst import *
from osgeo import gdal, osr

################################################################################
#	Function for iterative weighted least squares					           #
################################################################################
def lstsq(A, y, w=None, n_iter=None, n_sigma=None, ylim=None, cov=False, weight=False):
	"""
	Iterative weighted least-squares

	:param A: design matrix (NxM)
	:param y: observations (N)
	:param w: weights (N)
	:param n_iter: number of iterations
	:param n_sigma: outlier threshold (i.e. 3-sigma)
	:return: model coefficients
	:return: index of outliers
	"""

	i = 0

	if n_sigma is None:
		n_iter = 1

	if weight is True:
		W = np.diag(w)
		A = np.dot(W,A)
		y = np.dot(y,W)

	x   = np.ones(len(A.T)) * np.nan
	e   = np.ones(len(A.T)) * np.nan
	bad =  np.ones(y.shape, dtype=bool)

	while i <= n_iter:

		good = np.isfinite(y)
		
		if len(y[good]) < len(A.T): break

		try:
			x = np.linalg.lstsq(A[good,:], y[good], rcond=None)[0]
		except:
			break

		if n_sigma is not None:

			n0 = len(y[~good])

			r = y - np.dot(A, x)

			y[np.abs(r) > n_sigma * mad_std(r)] = np.nan

			if ylim is not None: y[np.abs(r) > ylim] = np.nan

			good = ~np.isnan(y)

			n1 = len(y[~good])

			if np.abs(n1 - n0) == 0: break

		i += 1

	if cov is True:
		try:
			s = np.nanvar(y - np.dot(A,x))
			e = np.sqrt(s * np.diag(pinv(A.T.dot(A))))
		except:
			pass

	bad = ~good

	return x, e, bad

################################################################################
#	Function for reading tif files without GDAL							       #
################################################################################
def tiffread(file):
	"""
	Read geotiff file

	:param file: input file name
	:return x: coordinate array(2D)
	:return y: coordinate array(2D)
	:return z: value array (2D)
	:return dx: x-resolution
	:return dy: y-resolution
	:return proj: crs string
	"""

	ds = xr.open_rasterio(file)

	dx, dy =  ds.attrs['res']

	nx, ny = ds.sizes["x"], ds.sizes["y"]

	transform = Affine(*ds.attrs["transform"])

	x, y = transform * np.meshgrid(np.arange(nx), np.arange(ny))

	z = ds.variable.data[0]

	proj = ds.attrs['crs']

	return x, y, z, dx, dy, proj

################################################################################
#	Function for reprojectind coordinates 									   #
################################################################################
def transform_coord(proj1, proj2, x, y):
    """
	Transform coordinates from proj1 to proj2
	usgin EPSG number

    :param proj1: current projection (4326)
    :param proj2: target projection (3031)
    :param x: x-coord in current proj1
    :param y: y-coord in current proj1
    :return: x and y now in proj2
    """

    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))

    return pyproj.transform(proj1, proj2, x, y)

################################################################################
#	Function for raster interpolation 							   			   #
################################################################################
def interp2d(x, y, z, xi, yi, **kwargs):
    """
    Raster to point interpolation based on
    scipy.ndimage import map_coordinates

    :param x: x-coord. in 2D (m)
    :param y: x-coord. in 2D (m)
    :param z: values in 2D
    :param xi: interp. point in x (m)
    :param yi: interp. point in y (m)
    :param kwargs: see map_coordinates
    :return: array of interp. values
    """

    x = np.flipud(x)
    y = np.flipud(y)
    z = np.flipud(z)

    x = x[0,:]
    y = y[:,0]

    nx, ny = x.size, y.size

    x_s, y_s = x[1] - x[0], y[1] - y[0]

    if np.size(xi) == 1 and np.size(yi) > 1:
        xi = xi * ones(yi.size)
    elif np.size(yi) == 1 and np.size(xi) > 1:
        yi = yi * ones(xi.size)

    xp = (xi - x[0]) * (nx - 1) / (x[-1] - x[0])
    yp = (yi - y[0]) * (ny - 1) / (y[-1] - y[0])

    coords = np.vstack([yp, xp])

    zi = map_coordinates(z, coords, mode='nearest', **kwargs)

    return zi

################################################################################
#	Function for 2D binning of data 							   			   #
################################################################################
def binning(x, y, xmin=None, xmax=None, dx=1 / 12., window=3 / 12.,
			median=False):
	"""
	Time-series binning (w/overlapping windows) using the mean/median.
	Handles NaN values in computation
	:param x: reference values (time)
	:param y: values to bin (elevation)
	:param xmin: min-value of reference
	:param xmax: max-value of reference
	:param dx: step-size of referece values
	:param window: size of binning window
	:param median: median instead of mean for binning
	:return xb: reference value (center of bin)
	:return yb: binned value
	:return eb: standard deviation of bin
	:return nb: number of values in bin
	:return sb: sum of values in bin
	"""

	if xmin is None:
		xmin = np.nanmin(x)
	if xmax is None:
		xmax = np.nanmax(x)

	steps = np.arange(xmin, xmax, dx)  # time steps
	bins = [(ti, ti + window) for ti in steps]  # bin limits

	N = len(bins)
	yb = np.full(N, np.nan)
	xb = np.full(N, np.nan)
	eb = np.full(N, np.nan)
	nb = np.full(N, np.nan)
	sb = np.full(N, np.nan)

	for i in range(N):

		t1, t2 = bins[i]
		idx, = np.where((x >= t1) & (x <= t2))

		if len(idx) == 0:
			xb[i] = 0.5 * (t1 + t2)
			continue

		ybv = y[idx]

		if median:
			yb[i] = np.nanmedian(ybv)
		else:
			yb[i] = np.nanmean(ybv)

		xb[i] = 0.5 * (t1 + t2)
		eb[i] = mad_std(ybv)
		nb[i] = np.sum(~np.isnan(ybv))
		sb[i] = np.sum(ybv)

	return xb, yb, eb, nb, sb

################################################################################
#	Function for wrapping longitude to 0-360 degress 						   #
################################################################################
def wrapTo360(arr):
	"""
	Wrapping array of values in degrees to 0-360 degrees

	:param arr: value in degress
	:return warr: wrapped value
	"""
	warr = arr.copy()
	positiveInput = (warr > 0)
	warr = np.mod(warr, 360)
	warr[(warr == 0) & positiveInput] = 360
	return warr

################################################################################
#	Function for wrapping longitude to -180 to 180 degress 					   #
################################################################################
def wrapTo180(arr):
	"""
	Wrapping array of values in degrees to -180 to 180 degrees

	:param arr: value in degress
	:return warr: wrapped value
	"""
	warr = arr.copy()
	idx = (warr < -180.) | (180. < warr)
	warr[idx] = wrapTo360(warr[idx] + 180.) - 180.
	return warr

################################################################################
#	Function for estimaging std.dev based on Absolute Median Deviation		   #
################################################################################
def mad_std(x, axis=None):
    """
    Robust std.dev using median absolute deviation

    :param x: data values
    :param axis: target axis for computation
    :return: std.dev (MAD)
    """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)

################################################################################
#	Function for estimating standard error based on MAD	   					   #
################################################################################
def mad_se(x, axis=None):
	"""
	Robust Robust standard error (using MAD)
	:param x: data values
	:param axis: target axis for computation
	:return: standard error based on MAD
	"""
	return mad_std(x, axis=axis) / np.sqrt(np.sum(~np.isnan(x, axis=axis)))

################################################################################
#	Function for filtering data based on MAD	   							   #
################################################################################
def median_filter(x, n=3):
	"""
	Remove values greater than n * MAD (set to NaN)
	:param x: data values
	:param n: integer for editing (3*MAD)
	:return: edited x-values (contains NaN's)
	"""
	x[np.abs(x - np.nanmedian(x)) > n_median * mad_std(x)] = np.nan
	return x

################################################################################
#	Function for constructing 2D or 1D grids for e.g interpolation			   #
################################################################################
def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=True):
    """
    Construct 2D-grid given input boundaries

    :param xmin: x-coord. min
    :param xmax: x-coord. max
    :param ymin: y-coors. min
    :param ymax: y-coord. max
    :param dx: x-resolution
    :param dy: y-resolution
    :param return_2d: if true return grid otherwise vector
    :return: 2D grid or 1D vector
    """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)

    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi

################################################################################
#	Function for removing outliers inside a defined spatial boundiing box	   #
################################################################################
def spatial_filter(x, y, z, dx, dy, n_sigma=3.0):
    """
    Spatial outlier editing filter

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param dx: filter res. in x (m)
    :param dy: filter res. in y (m)
    :param n_sigma: cutt-off value
    :param thres: max absolute value of data
    :return: filtered array containing nan-values
    """

    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    f_bin = stats.binned_statistic_2d(x, y, x, bins=(Ne, Nn))

    index = f_bin.binnumber

    ind = np.unique(index)

    zo = z.copy()

    for i in range(len(ind)):

        # index for each bin
        idx, = np.where(index == ind[i])

        zb = z[idx]

        if len(zb[~np.isnan(zb)]) == 0:
            continue

        dh = zb - np.nanmedian(zb)

        foo = np.abs(dh) > n_sigma * np.nanstd(dh)

        zb[foo] = np.nan

        zo[idx] = zb

    return zo
################################################################################
#	Function for peak detection in data (based on MATLAB version)	  	       #
################################################################################
def findpeaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False,valley=False):

    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        index of the peaks in `x`.

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak

        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

################################################################################
#	Function for filling non-defined data points	  	                       #
################################################################################
def fillnans(x):
	"""
	Interpolates and fills NaN data using linear interpolation

	:param x: vector contaning NaN's
	:return: vector with filled NaN's
	"""
	idx = np.arange(x.shape[0])

	good = np.where(np.isfinite(x))

	f = interp1d(idx[good], x[good],\
                 kind='linear', fill_value='extrapolate')

	return np.where(np.isfinite(x), x, f(idx))

################################################################################
#	Function for filtering/interpolation time series 	  	                       #
################################################################################
def window_filter(x, y, dx):

	yf = y.copy()

	for i in range(len(x)):

		dxx = np.abs(x - x[i])

		idx = np.argwhere(dxx < dx)

		yf[i] = np.nanmedian(y[idx])

	return yf

################################################################################
#	Function for smoothing time series 	  	                                   #
################################################################################
@jit(nopython=True)
def box_filter1d(x, k):
	"""
	box-filter for smoothing data using mean kernel

	:param x: vector of data to filter
	:param k: size of filter kernel/box
	:return: vector of filtered values
	"""

	n = len(x)
	y = x.copy()

	for i in range(n):
		y[i] = np.nanmean(x[i-k:i+k+1])
	return y

################################################################################
#    Function for spatial filtering using surface model                        #
################################################################################
def spatial_filter_param(x, y, z, dx, dy, niter=None, sigma=None, thres=None):
    """
    spatial parametric filter using bi-quadratic surface model
    to and edits residuals. should accept lat/lon as coords.
    :param x : x-coord
    :param y : y-coord
    :param z : vector of data to filter
    :param dx: size of box x-direction
    :param dy: size of box y-direction
    :param niter: number of least-squares iterations
    :param sigma: outlier threshold for residuals
    :param thres: absolut threshold for residuals
    :return: vector of filtered values
    """
    
    # Grid dimensions
    Nn = int((np.abs(y.max() - y.min())) / dy) + 1
    Ne = int((np.abs(x.max() - x.min())) / dx) + 1

    # Bin data
    f_bin = stats.binned_statistic_2d(x, y, x, bins=(Ne,Nn))

    # Get bin numbers for the data
    index = f_bin.binnumber

    # Unique indexes
    ind = np.unique(index)

    # Create output
    zo = z.copy()

    # Number of unique index
    for i in range(len(ind)):

        # index for each bin
        idx, = np.where(index == ind[i])

        # Get data
        xb = x[idx]
        yb = y[idx]
        zb = z[idx]

        # Centering of coordinates
        dxb, dyb = xb - xb.mean(), yb - yb.mean()

        # Design matrix
        Ab = np.vstack((np.ones(xb.shape), dxb, dyb,\
                        dxb*dyb, dxb**2, dyb**2,
                        dyb*dxb**2, dxb*dyb**2,
                        (dxb**2)*(dyb**2))).T

        # Iterative least-squares fit of data
        ibad = lstsq(Ab.copy(), zb.copy(), n_iter=niter, n_sigma=sigma, ylim=thres)[2]

        # Set to NaN again
        zb[ibad] = np.nan
        
        # Replace data
        zo[idx] = zb
    
    return zo

################################################################################
#    Function for spatial interpoaltion using median                           #
################################################################################
def interpmed(x, y, z, Xi, Yi, n, d):
    """
    2D median interpolation of scattered data

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param Xi: x-coord. grid (2D)
    :param Yi: y-coord. grid (2D)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :return: 1D array of interpolated values
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]

        zi[i] = np.median(zc)

    return zi

################################################################################
#    Function for spatial interpoaltion using gaussian kernel                  #
################################################################################
def interpgaus(x, y, z, s, Xi, Yi, n, d, a):
    """
    2D interpolation using a gaussian kernel
    weighted by distance and error

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. errors
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param n: number of nearest neighbours
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :return: 1D vec. of prediction, sigma and nobs
    """

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])

    if np.all(np.isnan(s)): s = np.ones(s.shape)

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)

        if n == 1:
            pass
        elif dxy.min() > d:
            continue
        else:
            pass

        zc = z[idx]
        sc = s[idx]
        
        if len(zc[~np.isnan(zc)]) == 0: continue
        
        # Weights
        wc = (1./sc**2) * np.exp(-(dxy**2)/(2*a**2))
        
        # Avoid singularity
        wc += 1e-6
        
        # Predicted value
        zi[i] = np.nansum(wc * zc) / np.nansum(wc)

        # Weighted rmse
        sigma_r = np.nansum(wc * (zc - zi[i])**2) / np.nansum(wc)

        # Obs. error
        sigma_s = 0 if np.all(s == 1) else np.nanmean(sc)

        # Prediction error
        ei[i] = np.sqrt(sigma_r ** 2 + sigma_s ** 2)

        # Number of points in prediction
        ni[i] = 1 if n == 1 else len(zc)

    return zi, ei, ni

################################################################################
#    Function for spatial interpoaltion using collocation/ordinary kriging     #
################################################################################
def interpkrig(x, y, z, s, Xi, Yi, d, a, n):
    """
    2D interpolation using ordinary kriging/collocation
    with second-order markov covariance model.

    :param x: x-coord (m)
    :param y: y-coord (m)
    :param z: values
    :param s: obs. error added to diagonal
    :param Xi: x-coord. interp. point(s) (m)
    :param Yi: y-coord. interp. point(s) (m)
    :param d: maximum distance allowed (m)
    :param a: correlation length in distance (m)
    :param n: number of nearest neighbours
    :return: 1D vec. of prediction, sigma and nobs
    """

    n = int(n)

    # Check
    if n == 1:
        print('n > 1 needed!')
        return

    xi = Xi.ravel()
    yi = Yi.ravel()

    zi = np.zeros(len(xi)) * np.nan
    ei = np.zeros(len(xi)) * np.nan
    ni = np.zeros(len(xi)) * np.nan

    tree = cKDTree(np.c_[x, y])
    
    # Convert to meters
    a *= 0.595 * 1e3
    d *= 1e3

    for i in range(len(xi)):

        (dxy, idx) = tree.query((xi[i], yi[i]), k=n)
        
        # Check if closest point is to far away
        if dxy.min() > d: continue

        xc = x[idx]
        yc = y[idx]
        zc = z[idx]
        sc = s[idx]
        
        # Need at least two for computation
        if len(zc) < 2: continue
        
        # Compute centroid/varibility for data
        m0 = np.median(zc)
        c0 = np.var(zc)
        
        # Covariance function for Dxy
        Cxy = c0 * (1 + (dxy / a)) * np.exp(-dxy / a)
        
        # Compute pair-wise distance
        dxx = cdist(np.c_[xc, yc], np.c_[xc, yc], "euclidean")
        
        # Covariance function Dxx
        Cxx = c0 * (1 + (dxx / a)) * np.exp(-dxx / a)
        
        # Measurement noise matrix
        N = np.eye(len(Cxx)) * sc * sc
        
        # Solve for the inverse
        CxyCxxi = np.linalg.solve((Cxx + N).T, Cxy.T)
        
        # Predicted value
        zi[i] = np.dot(CxyCxxi, zc) + (1 - np.sum(CxyCxxi)) * m0
        
        # Predicted error
        ei[i] = np.sqrt(np.abs(c0 - np.dot(CxyCxxi, Cxy.T)))
        
        # Number of points in prediction
        ni[i] = len(zc)

    return zi, ei, ni

################################################################################
#    Function for writing tif files with GDAL                                  #
################################################################################
def tiffwrite(ofile, X, Y, Z, dx, dy, proj, otype='float'):
    """
    Writing raster to a tif-file

    :param ofile: name of ofile
    :param X: x-coord of raster (2D)
    :param Y: y-coord of raster (2D)
    :param Z: values (2D)
    :param dx: grid-spacing x
    :param dy: grid-spacing y
    :param proj: projection (epsg number)
    :param dtype: save as 'int' or 'float'
    :return: written file to memory
    """

    proj = int(proj)

    N, M = Z.shape

    driver = gdal.GetDriverByName("GTiff")

    if otype == 'int':
        datatype = gdal.GDT_Int32

    if otype == 'float':
        datatype = gdal.GDT_Float32

    ds = driver.Create(ofile, M, N, 1, datatype)

    src = osr.SpatialReference()

    src.ImportFromEPSG(proj)

    ulx = np.min(np.min(X)) - 0.5 * dx

    uly = np.max(np.max(Y)) + 0.5 * dy

    geotransform = [ulx, dx, 0, uly, 0, -dy]

    ds.SetGeoTransform(geotransform)

    ds.SetProjection(src.ExportToWkt())

    ds.GetRasterBand(1).SetNoDataValue(np.nan)

    ds.GetRasterBand(1).WriteArray(Z)

    ds = None

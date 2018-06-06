import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt

tvar = 't_year'
xvar = 'lon'
yvar = 'lat'
zvar = 'h_res'

#fname = 'ANT_RA2_ISHELF_2002_2010_READ_A_RM_TOPO_IBE_TIDE_PARAMS.h5_subset'
fname = sys.argv[1]


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)


with h5py.File(fname, 'r') as f:
    step = 2
    time = f[tvar][::step] 
    lon = f[xvar][::step]
    lat = f[yvar][::step]
    height = f[zvar][::step]
    asc = f['asc'][::step]

    #trk_id = f['trk_id'][::step]
    trk_id = np.empty_like(height)

    '''
    #xcal = f['h_cal'][::step]  ##NOTE: h_cal2 was saved with step=2 already!
    xcal = f['h_cal_old'][::step]
    #xcal = f['h_cal2'][::]  ##NOTE: h_cal2 was saved with step=2 already!
    '''

# Apply cross-cal offset
#height -= xcal

'''
idx, = np.where(asc == 0)

time = time[idx]
lon = lon[idx]
lat = lat[idx]
height = height[idx]
'''

t1, t2 = 1992.34, 1992.36
idx, = np.where( (time > t1) & (time < t2) )

time = time[idx]
lon = lon[idx]
lat = lat[idx]
height = height[idx]
trk_id = trk_id[idx]


x, y = transform_coord(4326, 3031, lon, lat)


if 0:
    idx, = np.where( (x > -301000) & (x < -261000) & (y > -1081000) & (y < -1041000) )

    time = time[idx]
    lon = lon[idx]
    lat = lat[idx]
    height = height[idx]
    trk_id = trk_id[idx]
    x = x[idx]
    y = y[idx]


if 0:
    for trk in np.unique(trk_id):
        idx, = np.where(trk_id == trk)

        if len(idx) > 1:
            dx = np.hypot(x[idx], y[idx])
            height[idx] = np.gradient(height[idx], dx)
        else:
            height[idx] = np.nan

    #sys.exit()


# Remove fixed value
median = 0.0781279270892
height -= median
#height -= np.nanmedian(height)


nstd = mad_std(height) * 2.5
#zmin, zmax = -nstd, nstd
zmin, zmax = -.8, .8 

plt.figure(figsize=(10,5))
plt.scatter(x, y, c=height, s=.5, vmin=zmin, vmax=zmax,
        cmap=plt.cm.RdBu, rasterized=True)
plt.title('%.2f - %.2f, residual h (m)' % (t1, t2))

plt.colorbar(orientation='vertical', shrink=.7)

plt.show()
sys.exit()



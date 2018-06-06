import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

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


# Time series locations (center of search radius)
locations = [
             (-158.71, -78.7584),   # Ross
             (-124.427, -74.4377),  # Getz
             (-100.97, -75.1478),   # PIG
             ]

locations = [transform_coord(4326, 3031, lo, la) for lo, la in locations]

print 'loading data ...'

with h5py.File(fname, 'r') as f:
    step = 2
    lon = f['lon'][::step]
    lat = f['lat'][::step]
    h_res = f['h_res'][::step]
    h_bs = f['h_bs'][::step]
    t_year = f['t_year'][::step]

print 'subsetting ...'

# Subset dataset
ii, = np.where(lon < -90)
lon = lon[ii]
lat = lat[ii]
t_year = t_year[ii]
h_res = h_res[ii]
h_bs = h_bs[ii]

print 'converting coords ...'

x, y = transform_coord(4326, 3031, lon, lat)

# Plot location of time series
if 1:
    plt.plot(x, y, '.', rasterized=True)
    xx = [i for i, j in locations]
    yy = [j for i, j in locations]
    plt.plot(xx, yy, 'o')
    plt.show()

# Plot maps 
#plt.figure(figsize=(20,13))

print 'building kd-tree ...'

Tree = cKDTree(zip(x, y))

r = 2.5 * 1000  # search radius (m)

print 'querying ...'

for k, (xi, yi) in enumerate(locations):

    # Query the Tree with data coordinates
    idx = Tree.query_ball_point((xi, yi), r)
    
    if len(idx) == 0:
        continue

    t_year_ = t_year[idx]
    h_res_ = h_res[idx]
    h_bs_ = h_bs[idx]
    h_tide_ = h_tide[idx]
    h_load_ = h_load[idx]
    h_ibe_ = h_ibe[idx]
    h_unc_ = h_res_ + h_bs_

    idx, = np.where(~np.isnan(h_res_) & ~np.isnan(h_unc_))
    t_year_ = t_year_[idx]
    h_res_ = h_res_[idx]
    h_bs_ = h_bs_[idx]
    h_unc_ = h_unc_[idx]

    std1 = np.nanstd(h_unc_, ddof=1)
    std2 = np.nanstd(h_res_, ddof=1)
    print 'std_unc: %.4f' % std1
    print 'std_cor: %.4f' % std2
    print 'change : %.1f%%' % (100*(std2-std1)/std1)

    trn1 = np.polyfit(t_year_-t_year_.mean(), h_unc_-h_unc_.mean(), 1)[0]
    trn2 = np.polyfit(t_year_-t_year_.mean(), h_res_-h_res_.mean(), 1)[0]
    print 'trn_unc: %.4f' % trn1
    print 'trn_cor: %.4f' % trn2
    print 'change : %.1f%%' % (100*(trn2-trn1)/trn1)
    print ''

    plt.subplot(len(locations),1,k+1)
    plt.plot(t_year_, h_unc_, '.')
    plt.plot(t_year_, h_res_, '.')
    #plt.title('Correlation with Bs')

plt.show()


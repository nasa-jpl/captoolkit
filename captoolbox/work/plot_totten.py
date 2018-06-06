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


# Time series locations (center of serach radius)
locations = [(-158.71, -78.7584),   # Ross
             (-124.427, -74.4377),  # Getz
             (-100.97, -75.1478)]   # PIG

locations = [transform_coord(4326, 3031, lo, la) for lo, la in locations]

print 'loading data ...'

with h5py.File(fname, 'r') as f:
    lon = f['lon'][:]
    lat = f['lat'][:]
    t_year = f['t_year'][:]
    h_res = f['h_res'][:]

x, y = transform_coord(4326, 3031, lon, lat)

plt.plot(t_year, h_res, '.', rasterized=True)
#plt.plot(x, y, '.', rasterized=True)
#xx = [i for i, j in locations]
#yy = [j for i, j in locations]
#plt.plot(xx, yy, 'o')
plt.show()
sys.exit()

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
    h_unc_ = h_res_ + h_bs_

    print 'std_unc:', np.nanstd(h_unc_, ddof=1)
    print 'std_cor:', np.nanstd(h_res_, ddof=1)

    ii, = np.where(~np.isnan(h_res_) & ~np.isnan(h_unc_))

    print 'trd_unc:', np.polyfit(t_year_[ii]-t_year_[ii].mean(), h_unc_[ii]-h_unc_[ii].mean(), 1)[0]
    print 'trd_cor:', np.polyfit(t_year_[ii]-t_year_[ii].mean(), h_res_[ii]-h_res_[ii].mean(), 1)[0]
    print ''

    plt.subplot(3,1,k+1)
    plt.plot(t_year_, h_unc_, '.')
    plt.plot(t_year_, h_res_, '.')
    #plt.title('Correlation with Bs')

plt.show()


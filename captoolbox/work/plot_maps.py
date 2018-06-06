import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt

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
    step = 50
    lon = f['lon'][::step]
    lat = f['lat'][::step]
    r_bs = f['r_bs'][::step]
    r_lew = f['r_lew'][::step]
    r_tes = f['r_tes'][::step]
    s_bs = f['s_bs'][::step]
    s_lew = f['s_lew'][::step]
    s_tes = f['s_tes'][::step]
    p_std = f['p_std'][::step]
    d_std = f['d_std'][::step]
    p_trend = f['p_trend'][::step]
    d_trend = f['d_trend'][::step]
    r2 = f['r2'][::step]

    lon, lat = transform_coord(4326, 3031, lon, lat)

    lon /= 1000
    lat /= 1000

'''
#ii, = np.where((lon < -600000) | (lon > -200000))
#ii, = np.where((lat < -600000))
lon[ii] = np.nan
lat[ii] = np.nan
r_bs[ii] = np.nan
r_lew[ii] = np.nan
r_tes[ii] = np.nan
s_bs[ii] = np.nan
s_lew[ii] = np.nan
s_tes[ii] = np.nan
p_std[ii] = np.nan
d_std[ii] = np.nan
p_trend[ii] = np.nan
d_trend[ii] = np.nan
r2[ii] = np.nan
'''


# Plot histogram
if 0:
    for p in [d_std, d_trend, r2]:
        p[p==0] = np.nan
        plt.figure()
        plt.hist(p[~np.isnan(p)], bins=.100)
        plt.xlim(-mad_std(p)*3, mad_std(p)*3)

    plt.show()
    sys.exit()


# Plot maps 
plt.figure(figsize=(20,13))

'''
plt.subplot(4,3,1)
plt.scatter(lon, lat, c=r_bs, s=.1, vmin=-.5, vmax=.5, cmap=plt.cm.bwr, rasterized=True)
plt.title('Correlation with Bs')
plt.colorbar()

plt.subplot(4,3,2)
plt.scatter(lon, lat, c=r_lew, s=.1, vmin=-.5, vmax=.5, cmap=plt.cm.bwr, rasterized=True)
plt.title('Correlation with LeW')
plt.colorbar()

plt.subplot(4,3,3)
plt.scatter(lon, lat, c=s_tes, s=.1, vmin=-.5, vmax=.5, cmap=plt.cm.bwr, rasterized=True)
plt.title('Correlation with TeS')
plt.colorbar()
'''

ax = plt.subplot(3,3,1)
sigma = mad_std(s_bs) * 2
plt.scatter(lon, lat, c=s_bs, s=.1, vmin=-.3, vmax=.3, cmap=plt.cm.bwr, rasterized=True)
plt.title('Sensitivity to Bs')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,2)
sigma = mad_std(s_lew) * 2
plt.scatter(lon, lat, c=s_lew, s=.1, vmin=-.3, vmax=.3, cmap=plt.cm.bwr, rasterized=True)
plt.title('Sensitivity to LeW')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,3)
sigma = mad_std(s_tes) * 2
plt.scatter(lon, lat, c=s_tes, s=.1, vmin=-.3, vmax=.3, cmap=plt.cm.bwr, rasterized=True)
plt.title('Sensitivity to TeS')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,4)
plt.scatter(lon, lat, c=p_std, s=.1, vmin=-.2, vmax=0, rasterized=True)
plt.title('Percentage std change')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,5)
plt.scatter(lon, lat, c=p_trend, s=.1, vmin=-.75, vmax=.75, cmap=plt.cm.PiYG, rasterized=True)
plt.title('Percentage trend change')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,6)
sigma = mad_std(r2) * 2
plt.scatter(lon, lat, c=r2, s=.1, vmin=.05, vmax=.2, cmap=plt.cm.plasma_r, rasterized=True)
plt.title('r-squared of multivariate fit')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,7)
sigma = mad_std(d_std) * 3
plt.scatter(lon, lat, c=d_std, s=.1, vmin=-sigma, vmax=0, rasterized=True)
plt.title('Magnitude std change (m)')
plt.colorbar()
ax.set_rasterization_zorder(0)

ax = plt.subplot(3,3,8)
sigma = mad_std(p_trend) * .2
plt.scatter(lon, lat, c=d_trend, s=.1, vmin=-.06, vmax=.06, cmap=plt.cm.PiYG, rasterized=True)
plt.title('Magnitude trend change (m/yr)')
plt.colorbar()
ax.set_rasterization_zorder(0)


plt.show()

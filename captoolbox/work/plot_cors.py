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


def get_trend(x, y, order=1):
    coef = np.polyfit(x, y, order)
    return np.polyval(coef, x)


# Time series locations (center of search radius)
locations = [
             # Ross
             #(-158.40, -78.80), 
             #(-178.40, -78.80), 
             #(-188.00, -77.95),
             #(-160.00, -80.40), 
             #(-178.40, -80.60), 
             #(-190.40, -80.60), 

             # Ronne
             #(-61.09, -75.94),
             #(-57.78, -77.17),
             #(-52.82, -78.02),
             #(-67.55, -77.73),
             #(-66.06, -79.67),
             #(-55.46, -79.33),

             # Larsen-C
             #(-61.01, -66.50),
             #(-61.75, -67.68),
             #(-61.75, -68.47),
             #(-62.91, -66.67),
             #(-63.24, -67.60),
             #(-63.74, -68.00),

             # Amery
             (72.30, -69.30),
             (70.80, -69.40),
             (70.30, -69.90),
             (70.40, -70.41),
             (70.50, -70.99),
             (69.50, -71.61),
             ]


#scale = [.85, .8, .8, .85, .7, .7]  # Ross
#scale = [.85, .8, .8, .85, .7, .7]  # Ronne
#scale = [.85, .8, .8, .85, .7, .7]  # Larsen-C
#scale = [.8, .7, .9, .65, .85, .9]  # Amery fully uncor
scale = [.55, .8, .9, .6, .75, .75]  # Amery partially uncor
#scale = [1., 1., 1., 1., 1., 1.]  # Amery


locations = [transform_coord(4326, 3031, lo, la) for lo, la in locations]

print 'loading data ...'

with h5py.File(fname, 'r') as f:
    step = 2
    lon = f['lon'][::step]
    lat = f['lat'][::step]
    h_res = f['h_res'][::step]
    h_res_unf = f['h_res_unfilt'][::step]
    h_bs = f['h_bs'][::step]
    h_tide = f['h_tide'][::step]
    h_load = f['h_load'][::step]
    h_ibe = f['h_ibe'][::step]
    t_year = f['t_year'][::step]

print 'subsetting ...'

# Subset dataset
if 0:
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
    xi = [i for i,j in locations]
    yi = [j for i,j in locations]
    plt.plot(xi, yi, 'o')
    for k, (i,j) in enumerate(locations):
        plt.text(i+2.0, j, str(k+1))
    plt.title('Ross Ice Shelf')
    plt.show()

print 'building kd-tree ...'

Tree = cKDTree(zip(x, y))

r = 5 * 1000  # search radius (m)

print 'querying ...'

#plt.figure(figsize=(6.5,13.5))
plt.figure(figsize=(5.2,10.8))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
oran = colors[1]

for k, (xi,yi) in enumerate(locations):

    ss = scale[k]  # scale for tide cor

    # Query the Tree with data coordinates
    idx = Tree.query_ball_point((xi, yi), r)
    
    if len(idx) == 0:
        continue

    t_year_ = t_year[idx]
    h_res_ = h_res[idx]
    h_res_unf_ = h_res_unf[idx]
    h_bs_ = h_bs[idx]
    h_tide_ = h_tide[idx]
    h_load_ = h_load[idx]
    h_ibe_ = h_ibe[idx]

    # Add load to tide cor
    h_tide_ += h_load_

    h_unc_ = h_res_ + h_tide_ + h_ibe_ + h_bs_    # unapply cor
    #h_res_ = h_unc_ - h_tide_ * ss               # apply cors
    h_res_ = h_unc_ - h_ibe_  # apply cors


    COR = 'Tide correction (blue=uncor, orange=cor)'

    idx, = np.where(~np.isnan(h_res_) & ~np.isnan(h_unc_) & ~np.isnan(h_bs_))
    t_year_ = t_year_[idx]
    h_res_ = h_res_[idx]
    h_res_unf_ = h_res_unf_[idx]
    h_bs_ = h_bs_[idx]
    h_unc_ = h_unc_[idx]

    h_unc_ -= np.nanmean(h_unc_)
    h_res_ -= np.nanmean(h_res_)

    h_trn_cor = get_trend(t_year_, h_res_)
    h_trn_unc = get_trend(t_year_, h_unc_)

    std1 = np.nanstd(h_unc_-h_trn_unc, ddof=1)
    std2 = np.nanstd(h_res_-h_trn_cor, ddof=1)
    std_chg = (100*(std2-std1)/std1)

    trn1 = np.polyfit(t_year_-t_year_.mean(), h_unc_-h_unc_.mean(), 1)[0]
    trn2 = np.polyfit(t_year_-t_year_.mean(), h_res_-h_res_.mean(), 1)[0]
    trn_chg = (100*(trn2-trn1)/trn1)

    print 'std_unc: %.4f' % std1
    print 'std_cor: %.4f' % std2
    print 'change : %.1f%%' % std_chg

    print 'trn_unc: %.4f' % trn1
    print 'trn_cor: %.4f' % trn2
    print 'change : %.1f%%' % trn_chg
    print ''

    plt.subplot(len(locations),1,k+1)
    plt.plot(t_year_, h_unc_, '.', color=blue)
    plt.plot(t_year_, h_res_, '.', color=oran)
    plt.plot(t_year_, h_trn_unc, '-', linewidth=.75, color=blue)
    plt.plot(t_year_, h_trn_cor, '-', linewidth=.75, color=oran)

    text = 'Std %.3f => %.3f (%.1f%%)  Trend %.3f => %.3f (%.1f%%)' \
            % (std1, std2, std_chg, trn1, trn2, trn_chg)
    plt.annotate(text, (0.2,0.88), xycoords='axes fraction')

    plt.ylabel('Height change (m)')
    plt.xlim(1992.1, 1996.5)

    if k == 1:
        plt.ylim(-1.75, 1.76)
    elif k == 5:
        plt.ylim(-1.99, 1.99)
    else:
        plt.ylim(-1.25, 1.26)

    if k == 0:
        plt.title(COR)

plt.xlabel('Year')
plt.show()


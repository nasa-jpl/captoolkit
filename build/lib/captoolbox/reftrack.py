"""
Identifies repeat tracks and calculate the reference ground tracks.

"""
import sys
import h5py
import math
import pyproj
import tables as tb  # to read EArray file
import numpy as np
import statsmodels.api as sm
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from datetime import datetime
from scipy.spatial.distance import cdist


def distance_matrix(xy1, xy2):
    """
    Euclidean distance matrix between two sets of points.

    Calculates distances between all combinations of points.
    """
    d1 = np.subtract.outer(xy1[:,0], xy2[:,0])
    d2 = np.subtract.outer(xy1[:,1], xy2[:,1])
    return np.hypot(d1, d2)


def mean_distance(xy1, xy2):
    """
    Mean distance between two sets of points (tracks).

    Calculates the Euclidean distance matrix and returns
    the mean of the k-th smallest distances.
    """
    kth = min(len(xy1), len(xy2))  # NOTE: Set kth=10 ???
    #dist = distance_matrix2(xy1, xy2)
    dist = cdist(xy1, xy2, metric='euclidean')  # faster
    kth_vals = dist.ravel()[dist.ravel().argpartition(kth)[:kth]]  
    return kth_vals.mean()


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the earth.

    Input coords in decimal degrees -> output distance in km.
    """
    EARTH_RADIUS = 6378.137  # km 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
            math.cos(lat1) * math.cos(lat2) * \
            math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.asin(math.sqrt(a)) 
    return c * EARTH_RADIUS


haversine = np.vectorize(haversine)


def search_radius(x0, y0, x, y, radius=1):
    """
    Search all pts (x,y) within a radius from (x0,x0).

    x,y are lon/lat in decimal degrees.
    radius is distance in km.
    """
    idx, = np.where( haversine(x0, y0, x, y) <= radius )
    return x[idx], y[idx]


def transform_coord(proj1, proj2, x, y):
    """ Transform coordinates from proj1 to proj2 (EPSG num). """

    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)

    # Convert coordinates
    return pyproj.transform(proj1, proj2, x, y)



# Extract subset
if 0:
    with tb.open_file('ers1_test_large.h5') as f1, \
            h5py.File('ers1_test_medium.h5', 'w') as f2:
        d = f1.root.data
        lat = d[:,2]
        lon = d[:,3]
        #region = (lon > 193) & (lon < 202) & (lat > -80) & (lat < -78.6) # box
        region = (lon > 155) & (lon < 216.) & (lat < -77) & (lat > -82)  # Ross
        k, = np.where(region)
        f2['data'] = d[k,:]
        sys.exit()

# Load data
with tb.open_file('ers1_test_medium.h5') as f:
    d = f.root.data
    orbit = d[:,0]
    time = d[:,1]
    lat = d[:,2]
    lon = d[:,3]
    ascend = d[:,11]

# select region
if 1:
    #region = (lon > 155) & (lon < 216.) & (lat < -77) & (lat > -82)  # Ross
    region = (lat < -79) #& (lat > -81.5)  # Ronne #NOTE: Check this!!!
    k, = np.where(region)
    orbit = orbit[k]
    time = time[k]
    lat = lat[k]
    lon = lon[k]
    ascend = ascend[k] 

# Select asc/des #NOTE: I'ts always more efficient to separate first
if 1:
    k, = np.where(ascend == 1)
    orbit = orbit[k]
    time = time[k]
    lat = lat[k]
    lon = lon[k]
    ascend = ascend[k]

if 0:
    k, = np.where(orbit == orbits[1])
    j, = np.where(orbit == orbits[3])

    xy1 = np.column_stack((lon[k], lat[k]))
    xy2 = np.column_stack((lon[j], lat[j]))

    print np.sum(distance(xy1, xy2))

    plt.plot(lon, lat, '.')
    plt.plot(lon[k], lat[k], 'or')
    plt.plot(lon[j], lat[j], 'or')
    plt.show()

# Get tracks
track_ids = np.unique(orbit)  # get 1 id per track
print 'number of tracks:', len(track_ids)


# NOTE: What happens if the ref track is too small?

from itertools import combinations
id_pairs = combinations(track_ids, 2)

repeats = []
nonrepeats = []

# Iterate over each track pair
for id1,id2 in id_pairs:

    #TODO: Find if both ids were already accepted, then skip distance calc
    #TODO: extract all lats and lons only once and store them to use here

    k1, = np.where((orbit == id1) & (lat > -80))
    k2, = np.where((orbit == id2) & (lat > -80))

    if len(k1) < 2 or len(k2) < 2: continue

    xy1 = np.column_stack((lon[k1], lat[k1]))
    xy2 = np.column_stack((lon[k2], lat[k2]))

    mean_dist = mean_distance(xy1, xy2) * 111.3 # deg -> km

    if mean_dist < 4:
        #TODO: Find if any of the ids already exist
        #TODO: Join unique ids
        repeats.append([id1, id2])
    else:
        nonrepeats.append([id1, id2])

    #repeats = {k for k,v in distances.items() if v < 4}
    if len(repeats) == 20: break

sys.exit()


lon2, lat2 = transform_coord('4326', '3031', lon, lat)
plt.plot(lon2, lat2, '.', color='.7')  # plot all tracks
#plt.plot(lon, lat, '.', color='.7')  # plot all tracks

for j,ids in enumerate(repeats):

    k = np.in1d(orbit, ids)  # get the set of repeat ids 
    x, y = lon[k], lat[k]

    x_fit = np.arange(x.min(), x.max(), 0.1) 

    ### 1) Using Least Squares #NOTE: Doesn't work well at the ends
    if 0:
        p = np.polyfit(x, y, 2)
        y_fit = np.polyval(p, x_fit)

    ### 2) Using LOWESS
    if 0:
        lowess = sm.nonparametric.lowess(y, x, frac=1/3.)
        x_fit, y_fit = lowess[:,0], lowess[:,1]


    ### 3) Using great circle given min and max lat/lon
    if 1:
        
        i_min, i_max = y.argmin(), y.argmax()
        startlong, startlat = x[i_max], y[i_max]
        endlong, endlat = x[i_min], y[i_min]

        # calculate distance between points
        g = pyproj.Geod(ellps='WGS84')
        (az12, az21, dist) = g.inv(startlong, startlat, endlong, endlat)

        # calculate line string along path with segments <= 1 km
        lonlats = g.npts(startlong, startlat, endlong, endlat,
                         1 + int(dist / 5000))  # 1 pt every 5km

        # npts doesn't include start/end points, so prepend/append them
        lonlats.insert(0, (startlong, startlat))
        lonlats.insert(-1, (endlong, endlat))

        x_fit = np.array([i for i,j in lonlats])
        y_fit = np.array([j for i,j in lonlats])

        x_fit[x_fit<0] += 360

        # Adjust fitted x/y possitions
        for k,(x0,y0) in enumerate(zip(x_fit, y_fit)):
            xr, yr = search_radius(x0, y0, x, y, radius=5)
            if len(xr) > 1:
                x_fit[k] = np.median(xr)
                y_fit[k] = np.median(yr)


    if 0:
        x, y = transform_coord('4326', '3031', x, y)
        x_fit, y_fit = transform_coord('4326', '3031', x_fit, y_fit)

    plt.plot(x, y, '.', color='.5')
    plt.plot(x_fit, y_fit, 'o') 

    #plt.xlim(150, 215)
    
    if j == 20:
        break

plt.show()

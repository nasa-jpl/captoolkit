import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt


# Subset using 'geodetic' or 'steregraphic' coords
if 0:
    # Using lon/lat
    lon1, lon2, lat1, lat2 = -170, -95, -80, -60  # Amundsen Sea sector
    #lon1, lon2, lat1, lat2 = -95, 0, -80, -60  # Drawning Maud sector
    stereo = False
else:
    # Using x/y
    lon1, lon2, lat1, lat2 = -600000, 400000, -1400000, -800000  # Ross Ice Shelf
    stereo = True


# Variables to save in sub-setted file
fields = ['lon', 'lat', 'h_res', 't_year', 't_sec', 'bs', 'lew', 'tes']
#fields = ['lon', 'lat', 'd_trend', 'd_std', 'r2']


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


assert len(sys.argv[1:]) == 2, 'need input and output file names'

ifile = sys.argv[1]
ofile = sys.argv[2]

fi = h5py.File(ifile, 'r')

lon = fi['lon'][:]
lat = fi['lat'][:]

if stereo:
    lon, lat = transform_coord(4326, 3031, lon, lat)

idx, = np.where( (lon > lon1) & (lon < lon2) & (lat > lat1) & (lat < lat2) )


# Plot for testing
if 0:
    plt.figure()
    plt.plot(lon, lat, '.', rasterized=True)
    plt.figure()
    plt.plot(lon[idx], lat[idx], '.', rasterized=True)
    plt.show()
    sys.exit()


with h5py.File(ofile, 'w') as fo:
    for var in fields:
        fo[var] = fi[var][:][idx]

fi.close()

print 'done.'

"""
Extracts geographic region.

Example:
    python subset.py '/u/devon-r0/shared_data/ers/floating_/latest/AntIS_E2_REAP_ERS_ALT*'

Notes:
    Bedmap boundaries: -b -3333000 3333000 -3333000 3333000
    Ross boundaries: -b -600000 400000 -1400000 -400000

"""
import os
import sys
import h5py
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

#=== Edit ============================================================

# Geographic boundaries
if 0:
    # Using lon/lat (geodetic)
    #lon1, lon2, lat1, lat2 = -170, -95, -80, -60  # Amundsen Sea sector
    #lon1, lon2, lat1, lat2 = -95, 0, -80, -60  # Drawning Maud sector
    lon1, lon2, lat1, lat2 = -180, 0, -90, -60  # Half Ross 
    stereo = False
else:
    # Using x/y (polar stereo)
    lon1, lon2, lat1, lat2 = -600000, 400000, -1400000, -400000  # Ross Ice Shelf
    stereo = True

# Suffix for output files
suffix = '_ROSS'

# Variables to save in output files 
#fields = ['lon', 'lat', 'h_res', 'h_cor', 'h_bs', 't_year', 't_sec', 'bs', 'lew', 'tes']
#fields = ['lon', 'lat', 'd_trend', 'd_std', 'r2']
fields = None  # all fields

# Parallel processing
njobs = 16

#=== End Edit ======================================================== 


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


def gen_ofile_name(ifile, suffix='_subset'):
    path, ext = os.path.splitext(ifile)
    ofile = path + suffix + ext
    return ofile


'''
assert len(sys.argv[1:]) == 2, 'need input and output file names'

ifile = sys.argv[1]
ofile = sys.argv[2]
'''

if len(sys.argv[1:]) > 1:
    files = sys.argv[1:]
else:
    files = glob(sys.argv[1])


def main(ifile):

    fi = h5py.File(ifile, 'r')

    lon = fi['lon'][:]
    lat = fi['lat'][:]

    # +/- 180 -> 0/360
    #lon[lon<0] += 360

    if stereo:
        lon, lat = transform_coord(4326, 3031, lon, lat)

    idx, = np.where( (lon > lon1) & (lon < lon2) & (lat > lat1) & (lat < lat2) )

    if len(idx) == 0:
        return

    # Plot for testing
    if 0:
        plt.figure()
        plt.plot(lon, lat, '.', rasterized=True)
        plt.figure()
        plt.plot(lon[idx], lat[idx], '.', rasterized=True)
        plt.show()
        sys.exit()

    if fields is None:
        fields_ = list(fi.keys())
    else:
        fields_ = fields

    ofile = gen_ofile_name(ifile, suffix=suffix)

    with h5py.File(ofile, 'w') as fo:
        for var in fields_:
            fo[var] = fi[var][:][idx]

    print(('output ->', ofile))

    fi.close()


if njobs == 1:
    print('running sequential code...')
    [main(f) for f in files]

else:
    print(('running parallel code (%d jobs)...' % njobs))
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

print('done.')

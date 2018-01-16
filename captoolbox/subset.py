import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

lon1, lon2 = -170, -95

fname = sys.argv[1]

fi = h5py.File(fname, 'r')

lon = fi['lon'][:]
lat = fi['lat'][:]

idx, = np.where( (lon > -170) & (lon < -95) & (lat > -80) )

fields = ['lon', 'lat', 'h_res', 't_year', 'bs', 'lew', 'tes']

with h5py.File(fname+'_subset', 'w') as fo:
    for var in fields:
        fo[var] = fi[var][:][idx]

fi.close()

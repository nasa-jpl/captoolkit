import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

#lon1, lon2 = -170, -95  # Amundsen sector
lon1, lon2 = -95, 0  # Drawming sector
lat1 = -80

fname = sys.argv[1]

fi = h5py.File(fname, 'r')

lon = fi['lon'][:]
lat = fi['lat'][:]

idx, = np.where( (lon > lon1) & (lon < lon2) & (lat > lat1) )

fields = ['lon', 'lat', 'h_res', 'h_cor', 't_year', 'bs', 'lew', 'tes']

with h5py.File(fname+'_subset2', 'w') as fo:
    for var in fields:
        fo[var] = fi[var][:][idx]

fi.close()

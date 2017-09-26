"""
The sea level increases (decreases) by approximately 1 cm when air
pressure decreases (increases) by approximately 1 mbar). The inverse
barometer correction (IBE) that must be subtracted from the sea surface
height is simply given by:

    h_ibe = (-1/rho g) * (P - P_ref)

where Pref is the global "mean" pressure (reference pressure) over the
ocean (rho is sea water density and g gravity). For most applications,
Pref is assumed to be a constant (e.g., 1013.3 mbar).

(Dorandeu and Le Traon, 1999)
'http://journals.ametsoc.org/doi/full/10.1175/
1520-0426%281999%29016%3C1279%3AEOGMAP%3E2.0.CO%3B2'

Our correction:

    h_ibe(x,y,t) = (-1/rho g) * (P(x,y,t) - P_ref(x,y))

where P_ref(x,y) is the climatological mean at each location.

Several refereces here:
https://link.springer.com/chapter/10.1007/978-3-662-04709-5_88

NOTES:

The IBE correction should be applied as:

    h_cor = h - h_ibe

"""
import sys
import h5py
import numpy as np
from netCDF4 import Dataset


# Default sea-level pressure file (Era-Interim)
SLPFILE = 'SLP_antarctica_daily_00-12_3-12_19900101-20170331.nc'


def slp_to_ibe(P, rho=1028., g=9.80665, P_ref=None):
    """
    Convert sea-level pressure to inverse barometer correction.

        h_ibe(x,y,t) = -1/(rho g) (P(x,y,t) - P_ref(x,y))

    P     - Pressure at sea level [Pa = kg/m.s2], P = P(t,y,x).
    rho   - Density of sea water [kg/m3].
    g     - Gravity [m/s2].
    P_ref - Reference pressure, the climatological mean: P_ref = P_ref(y,x).
            If P_ref is not None (e.g. the mean global pressure is given:
            1013.25 * 100 [Pa]), then P_ref = const.
    h_ibe - Inverse Baromenter Correction [m].

    """
    P_ref = np.nanmean(P, axis=0)  # P = P(t,y,x)
    h_ibe = (-1 / (rho * g)) * (P - P_ref[np.newaxis,:,:])  # m
    return h_ibe


infile = sys.argv[1:]
infile = SLPFILE if not infile else infile[0]

# Read NetCDF
print 'loading SLP file ...'
ds = Dataset(infile, "r")

lon = ds.variables['longitude'][:]  # [deg]
lat = ds.variables['latitude'][:]  # [deg]
time = ds.variables['time'][:]  # [hours since 1900-01-01 00:00:0.0]
msl = ds.variables['msl']#[:]  # msl(t,y,x) [Pa]; WARNING: data are too big!!!

#NOTE: Do not apply these! netCDF4 applies it at read time!
scale = getattr(ds['msl'], 'scale_factor')
offset = getattr(ds['msl'], 'add_offset')
missing = getattr(ds['msl'], 'missing_value')

# Subset region for testing (inclusive)
# Do not load full data into memory
if 0:

    # Filter time
    time = time/8760. + 1900  # hours since 1900 -> years
    #idx1, = np.where((time >= 2000) & (time <= 2009))
    #k1, k2 = idx1[0], idx1[-1]+1

    # Filter latitude
    #idx2, = np.where(lat < -60)
    #i1, i2 = idx2[0], idx2[-1]+1

    # Subset
    #time = time[k1:k2]
    #lat = lat[i1:i2]
    #msl = msl[k1:k2,i1:i2,:]
    #msl = msl[:,i1:i2,:]

    #msl = slp_to_ibe(msl)  # P -> IBE

    # Plot
    import pandas as pd
    import matplotlib.pyplot as plt

    find_nearest = lambda arr, val: (np.abs(arr-val)).argmin()
    j = find_nearest(lon, (297.5-360))  # LC
    i = find_nearest(lat, -67.5)
    #j = find_nearest(lon, (333.3-360))  # Brunt
    #i = find_nearest(lat, -75.6)
    p = msl[:,i,j]

    p = (-1 / (1028. * 9.80665)) * (p - np.mean(p))  # m

    p = pd.Series(p).rolling(window=8, center=True).mean()

    time = (time-2007) * 365 - 26  # 26 Leap days from 1900 to 2007

    plt.plot(time, p, linewidth=1.5)
    plt.xlim(324.753, 388.66)  # LC
    #plt.xlim(100, 160)  # Brunt
    plt.show()
    sys.exit()


print 'variables:', ds.variables
print 'Resolution:'
print 'Dlon (deg):', np.diff(lon)
print 'Dlat (deg):', np.diff(lat)
print 'Dtime (year):', np.diff(time)
print 'time steps:', time
print 'msl pressure:', msl
print 'scale_factor:', scale
print 'add_offset:', offset
print 'missing_value:', missing

# Convert sea-level pressure to inverse barometer correction
print 'converting SLP to IBE ...'
ibe = slp_to_ibe(msl)

# Save data
outfile = infile.replace('SLP_', 'IBE_').replace('.nc', '.h5')

with h5py.File(outfile, 'w') as f:
    kw = {'chunks': True, 'compression': 'gzip', 'compression_opts': 9}
    f.create_dataset('lon', data=lon, **kw)
    f.create_dataset('lat', data=lat, **kw)
    f.create_dataset('time', data=time, **kw)
    f.create_dataset('ibe', data=ibe, **kw)

print 'Output file:', outfile

import sys
import h5py
import pyproj
import numpy as np
import xarray as xr

# from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans


fname = "/Users/paolofer/code/captoolkit/captoolkit/work/cube_full/FULL_CUBE_v2.h5"
fmask = "/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5"


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            try:
                f[k] = np.squeeze(v)
            except:
                f[k][:] = np.squeeze(v)


def xregrid3d(x, y, t, h, x_new, y_new):
    da = xr.DataArray(h, [("y", y), ("x", x), ("t", t)])  # NOTE: Check this
    return da.interp(x=x_new, y=y_new).values


def xregrid2d(x, y, z, x_new, y_new):
    da = xr.DataArray(z, [("y", y), ("x", x)])
    return da.interp(x=x_new, y=y_new).values


def make_grid(xmin, xmax, ymin, ymax, dx, dy, return_2d=False):
    """Construct output grid-coordinates."""
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1
    xi = np.linspace(xmin, xmax, num=Ne)
    yi = np.linspace(ymin, ymax, num=Nn)
    if return_2d:
        return np.meshgrid(xi, yi)
    else:
        return xi, yi


def interp2d(xd, yd, data, xq, yq, **kwargs):
    """Bilinear interpolation from grid."""
    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)
    xd = xd[0, :]
    yd = yd[:, 0]
    nx, ny = xd.size, yd.size
    (x_step, y_step) = (xd[1] - xd[0]), (yd[1] - yd[0])
    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])
    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq * ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq * ones(xq.size)
    xp = (xq - xd[0]) * (nx - 1) / (xd[-1] - xd[0])
    yp = (yq - yd[0]) * (ny - 1) / (yd[-1] - yd[0])
    coord = np.vstack([yp, xp])
    zq = map_coordinates(data, coord, **kwargs)
    return zq


def filt_velocity_boundary(
    x, y, melt=None, blon=186, blat=-82.68, dlon=13, dlat=0.06, return_index=False
):
    """
    params for 3km grid: blon=186, blat=-82.68, dlon=13, dlat=0.06
    """
    X, Y = np.meshgrid(x, y)
    lon, lat = transform_coord(3031, 4326, X, Y)
    lon[lon < 0] += 360
    (ii, jj) = np.where(
        (lon > blon - dlon)
        & (lon < blon + dlon)
        & (lat > blat - dlat)
        & (lat < blat + dlat)
    )
    if return_index:
        return (ii, jj)
    else:
        melt[ii, jj] = np.nan
        return melt


def mask_boxes(x, y, z, boxes=None, fill_value=np.nan):
    # box = (x1, x2, y1, y2, abs)
    if boxes is None:
        boxes = [
            (-2370500, -2316000, 1264000, 1293000, 3),  # Larsen B Calving
            # (-2230000, -2210000, 1255000, 1275000, 3),  # Larsen C Bauden
            (-2206000, -2150000, 1202000, 1264000, 3),  # Larsen C Iceberg
            (-2137500, -2052000, 1128000, 1162500, 1),  # Larsen Crack
            (-896000, -849000, 914000, 990000, 5),  # Filchner Front
            (-817000, -749000, 918000, 995000, 2),  # Filchenr Crack
            (-68000, 50000, 2170000, 2220000, 3),  # QM Fimbul
            (321000, 347000, 2155000, 2185000, 3),  # QM Vigrid
            (1815000, 1854000, 1605500, 1652000, 4),  # Prince Harald
            (2211500, 2245000, 662000, 754000, 4.5),  # Amery Front
            (2147000, 2188000, 610000, 657000, 5),  # Amery Side
            (2283000, 2325750, -1151000, -1125500, 5),  # Totten Front
            (2157000, 2195000, -1383000, -1363000, 4),  # Moscoww Front
            (1399000, 1443000, -2091500, -2059000, 4),  # Mertz Iceberg
            (387500, 402500, -1542500, -1522500, 5),  # Drygalsky Front
            (-377500, 135000, -1360000, -1125000, 1),  # Ross Cracks/Front
            (-2113000, -2043000, 570000, 646000, 5),  # Wikins
        ]
    X, Y = np.meshgrid(x, y)
    for box in boxes:
        x1, x2, y1, y2, abs = box
        ij = np.where((X > x1) & (X < x2) & (Y > y1) & (Y < y2) & (np.abs(z) > abs))
        z[ij] = fill_value
    return z


def transform_coord(proj1, proj2, x, y):
    """Transform coordinates from proj1 to proj2 (EPSG num).

    Examples EPSG proj:
        Geodetic (lon/lat): 4326
        Stereo AnIS (x/y):  3031
        Stereo GrIS (x/y):  3413
    """
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:" + str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:" + str(proj2))
    return pyproj.transform(proj1, proj2, x, y)


"""
WorldLimits are cell centers.

EPSG 3031

XWorldLimits: [-2677927.5 2816152.5]
YWorldLimits: [-2153752.5 2259367.5]
RasterSize: [4598 5724]
RasterInterpretation: 'postings'
ColumnsStartFrom: 'north'
RowsStartFrom: 'west'
SampleSpacingInWorldX: 960
SampleSpacingInWorldY: 960
RasterExtentInWorldX: 5494080
RasterExtentInWorldY: 4413120
XIntrinsicLimits: [1 5724]
YIntrinsicLimits: [1 4598]
TransformationType: 'rectilinear'
CoordinateSystemType: ‘planar'
"""


x, y, t, melt, mask_zeros = h5read(fname, ["x", "y", "t", "dHdt_melt", "mask_zeros"])

# t, melt = t[:5], melt[:,:,:5]

# Get new grid coords
print("loading cube ...")
if 0:
    x_new = np.arange(x.min(), x.max() + 1000, 1000)
    y_new = np.arange(y.min(), y.max() + 1000, 1000)
else:
    x_new, y_new = make_grid(-2677927.5, 2816152.5, -2153752.5, 2259367.5, 960, 960)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    xx_new, yy_new = X_new.ravel(), Y_new.ravel()

print("Nx:", len(x_new))
print("Ny:", len(y_new))
print("dx:", x_new[1] - x_new[0])
print("dy:", y_new[1] - y_new[0])
print("Min/Max x:", x_new[0], x_new[-1])
print("Min/Max y:", y_new[0], y_new[-1])

# Get mask at higher res
print("regriding mask ...")
X_mask, Y_mask, mask = h5read(fmask, ["x", "y", "mask"])
mask_new = interp2d(X_mask, Y_mask, mask, xx_new, yy_new, order=1)
mask_new = mask_new.reshape(X_new.shape)
mask_new = np.flipud(mask_new)
Y_new = np.flipud(Y_new)
del mask

"""
plt.matshow(mask_new)
plt.figure()
plt.pcolormesh(X_new, Y_new, mask_new)
plt.show()
sys.exit()
"""

# Filter artifacts and extend boundaries
for k in range(melt.shape[2]):
    print(k)
    melt_k = melt[:, :, k]
    melt_k = filt_velocity_boundary(x, y, melt_k)
    melt_k = filt_positives(x, y, melt_k, [(-1574000, -1543250, -500000, -440000)])

    melt[:, :, k] = interpolate_replace_nans(
        melt_k, Gaussian2DKernel(1), boundary="extend"
    )

# Mask with zeros
if 0:
    mask3d_zeros = np.repeat(mask_zeros[:, :, np.newaxis], melt.shape[2], axis=2)
    melt[mask3d_zeros == 1] = 0.0

"""
plt.matshow(np.nanmean(melt, axis=2), vmin=-5, vmax=5, cmap='RdBu')
plt.show()
sys.exit()
"""

# Regrid (comes out upside down)
print("regriding cube ...")
melt = xregrid3d(x, y, t, melt, x_new, y_new)
melt = np.flipud(melt)
y_new = y_new[::-1]

# Mask high res ice shelves
mask3d_new = np.repeat(mask_new[:, :, np.newaxis], melt.shape[2], axis=2)
melt[mask3d_new != 1] = np.nan

# Get high res mean melt
melt_mean = np.nanmean(melt, axis=2)

if 0:
    fname = "cube_full/Melt_960m_nozeros.h5"
    h5save(
        fname,
        {
            "dHdt_melt_960m": melt,
            "dHdt_melt_mean_960m": melt_mean,
            "x_960m": x_new,
            "y_960m": y_new,
            "mask_floating_960m": mask_new,
        },
        "a",
    )
    print("saved.")


print("plotting ...")
plt.figure()
plt.matshow(melt_mean, vmin=-5, vmax=5, cmap="RdBu", rasterized=True)
plt.figure()
plt.pcolormesh(x_new, y_new, melt_mean)
plt.show()

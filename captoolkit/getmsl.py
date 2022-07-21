"""
Calculate and extend MSL grid for the ice shelves.

Recipe (new):
    1. Grid Geoid to Tom's original MSL
    2. Regrid GMDT to Toms's original MSL
    3. Remove Geoid from Tom's MSL => MDT
    4. Remove coastal data from GMDT (keep open ocean)
    5. Compute difference between Tom's MDT and GMDT
    6. Remove offset to Tom's MDT
    7. Get land mask for Tom's MDT
    8. Extend Tom's MDT and GMDT to ice shelves (w/cosine taper)
    9. Extend and regrid regional SLT
    10. Regrid extended MDT, GMDT and Geoid to Cube grid
    11. Reconstruct MSL and GMSL (MDT_extended + Geoid)
    12. Mask land on ocean products and save to cube

Notes:
    - In future, add climatology to MDT => MDT/MSL cube ?
    - In future, use Tom's regional SLT
    - Use cosine taper, which affects large ice shelves only

Sources:
    See download recipes in respective data folders.

    Geoid (GOCO05c, latest combined):
    http://icgem.gfz-potsdam.de/calcgrid

    MDT (global):
    https://www.aviso.altimetry.fr/no_cache/en/my-aviso-plus/my-products.html

    SLT (global):
    https://www.aviso.altimetry.fr/en/data/products/ \
        ocean-indicators-products/mean-sea-level/products-images.html

    MSL (Tom's CS2):
    ~/data/msl/CS2_MSS_2011_2016_5km.nc

"""
import h5py
import numpy as np
import pyproj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from astropy.convolution import (
    Gaussian2DKernel,
    convolve,
    interpolate_replace_nans,
)
try:
    from gdalconst import GA_ReadOnly
except ImportError as e:
    from osgeo.gdalconst import GA_ReadOnly
from netCDF4 import Dataset
from osgeo import gdal, osr
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.signal import tukey

file_msk = "/Users/paolofer/data/masks/jpl/ANT_groundedice_240m.tif.h5"

file_msk2 = "/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5"

file_cube = "/Users/paolofer/work/melt/data/FULL_CUBE_v3.h5"
xcub = "x"
ycub = "y"

file_geo = "/Users/paolofer/data/geoid/goco/GOCO05c_01deg.gdf"
xgeo = 0
ygeo = 1
zgeo = 2

file_msl = "/Users/paolofer/data/msl/CS2_MSS_2011_2016_5km_lonlat.nc"
xmsl = "lon"
ymsl = "lat"
zmsl = "mss"

file_mdt = "/Users/paolofer/data/msl/mdt_cnes_cls2013_global.nc"
xmdt = "lon"
ymdt = "lat"
zmdt = "mdt"

file_slt = (
    "/Users/paolofer/data/msl/MSL_Map_MERGED_Global_AVISO_NoGIA_Adjust.nc"
)
xslt = "longitude"
yslt = "latitude"
zslt = "sea_level_trends"


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def ncread(ifile, vnames):
    ds = Dataset(ifile, "r")  # NetCDF4
    d = ds.variables

    return [d[v][:] for v in vnames]


def tifread(ifile, metaData="A"):
    """Read raster from file."""
    file = gdal.Open(ifile, GA_ReadOnly)
    projection = file.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(projection)
    # proj = src.ExportToWkt()
    Nx = file.RasterXSize
    Ny = file.RasterYSize
    trans = file.GetGeoTransform()
    dx = trans[1]
    dy = trans[5]

    if metaData == "A":
        xp = np.arange(Nx)
        yp = np.arange(Ny)
        (Xp, Yp) = np.meshgrid(xp, yp)
        X = (
            trans[0] + (Xp + 0.5) * trans[1] + (Yp + 0.5) * trans[2]
        )  # FIXME: bottleneck!
        Y = trans[3] + (Xp + 0.5) * trans[4] + (Yp + 0.5) * trans[5]

    if metaData == "P":
        xp = np.arange(Nx)
        yp = np.arange(Ny)
        (Xp, Yp) = np.meshgrid(xp, yp)
        X = trans[0] + Xp * trans[1] + Yp * trans[2]  # FIXME: bottleneck!
        Y = trans[3] + Xp * trans[4] + Yp * trans[5]
    band = file.GetRasterBand(1)
    Z = band.ReadAsArray()
    dx = np.abs(dx)
    dy = np.abs(dy)
    # return X, Y, Z, dx, dy, proj

    return X, Y, Z


def h5save(fname, vardict, mode="a"):
    """Generic HDF5 writer.

    vardict : {'name1': var1, 'name2': va2, 'name3': var3}
    """
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(v)
            else:
                f[k] = np.squeeze(v)


def extend_field(grid, mask=None, gauss_widths=(1, 11), taper=None):
    """Extrapolate w/Gaussian average of increasing width and taper weights.

    Extend field (internal and external borders) by applying a Gaussian
    average of increasing width at each iteration, and scaling the result
    by a cosine/linear taper (from 1 to 0 with iterations).

    Args:
        grid: 2D field to extrapolate.
        mask: region to exclude from the Gaussian average.
        gauss_widths: width (n_pixels) of Gaussian average at each iteration.
        taper: cosine|linear|None, shape of the weight window.
    Notes:
        Cosine taper goes from 1 at gauss_widths[0] to 0 at gauss_widths[-1].
        gauss_widths define the number of iterations (one per width).
        Widths can be passed as a range (1, N) or explicitly (1, 3, 5, ...).
    """

    if len(gauss_widths) == 2:
        gauss_widths = range(*gauss_widths)

    if taper == "cosine":
        half_window = len(gauss_widths)
        taper_weights = tukey(half_window * 2)[half_window:]
    elif taper == "linear":
        taper_weights = np.linspace(1, 0, len(gauss_widths))
    else:
        taper_weights = np.full_like(gauss_widths, 1)

    print("taper weights:\n", taper_weights)

    if mask is not None:
        grid[mask == 1] = np.nan  # <= this is key !!!

    for i, k in enumerate(gauss_widths):
        print("gauss kernel size:", k)

        mask_before = np.isnan(grid)

        kernel = Gaussian2DKernel(k)
        grid = interpolate_replace_nans(grid, kernel, boundary="extend")

        if mask is not None:
            grid[mask == 1] = np.nan

        mask_after = np.isnan(grid)

        mask_extended = mask_before & ~mask_after

        grid[mask_extended] *= taper_weights[i]

    return grid


def interp2d(xd, yd, data, xq, yq, **kwargs):
    """Bilinear interpolation from grid."""
    xd = np.flipud(xd)
    yd = np.flipud(yd)
    data = np.flipud(data)
    xd = xd[0, :]
    yd = yd[:, 0]
    nx, ny = xd.size, yd.size
    assert (ny, nx) == data.shape
    assert (xd[-1] > xd[0]) and (yd[-1] > yd[0])

    if np.size(xq) == 1 and np.size(yq) > 1:
        xq = xq * np.ones(yq.size)
    elif np.size(yq) == 1 and np.size(xq) > 1:
        yq = yq * np.ones(xq.size)
    xp = (xq - xd[0]) * (nx - 1) / (xd[-1] - xd[0])
    yp = (yq - yd[0]) * (ny - 1) / (yd[-1] - yd[0])
    coord = np.vstack([yp, xp])
    zq = map_coordinates(data, coord, **kwargs)

    return zq


def get_mask(file_msk, X, Y):
    """ Given mask file and x/y grid coord return boolean mask.

    X/Y can be either 1D or 2D.
    """

    if ".h5" in file_msk:
        Xm, Ym, Zm = h5read(file_msk, ["x", "y", "mask"])
    else:
        Xm, Ym, Zm = tifread(file_msk)

    # Interpolation of grid to points for masking
    mask = interp2d(Xm, Ym, Zm, X.ravel(), Y.ravel(), order=1)
    mask = mask.reshape(X.shape)

    # Set all NaN's to zero
    mask[np.isnan(mask)] = 0

    # Convert to boolean

    return mask == 1


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
    # Convert coordinates

    return pyproj.transform(proj1, proj2, x, y)


def regrid2d(x1, y1, z1, x2, y2, method="linear"):
    """Regrid z1(x1,y1) onto z2(x2,y2).

    Args:
        z1 is a 2D array.
        x1/y1/x2/y2 can be either 1D or 2D arrays.
    """

    if np.ndim(x1) == 1:
        X1, Y1 = np.meshgrid(x1, y1)
    else:
        X1, Y1 = x1, y1

    if np.ndim(x2) == 1:
        X2, Y2 = np.meshgrid(x2, y2)
    else:
        X2, Y2 = x2, y2

    Z2 = griddata(
        (X1.ravel(), Y1.ravel()), z1.ravel(), (X2, Y2), method=method
    )

    return Z2.reshape(X2.shape)


print("loading cube coords ...")
x_cube, y_cube = h5read(file_cube, [xcub, ycub])
X, Y = np.meshgrid(x_cube, y_cube)  # common grid

# Get Tom's MSL (2D @ 5km)
print("loading MSL ...")
lon_msl, lat_msl, z_msl = ncread(file_msl, [xmsl, ymsl, zmsl])
x_msl, y_msl = transform_coord(4326, 3031, lon_msl, lat_msl)

# Get Geoid

if 0:
    print("gridding Geoid to MSL grid ...")
    lon_geo, lat_geo, z_geo = np.loadtxt(
        file_geo, usecols=(xgeo, ygeo, zgeo), unpack=True
    )
    # Lon/lat -> x/y
    x_geo, y_geo = transform_coord(4326, 3031, lon_geo, lat_geo)

    plt.scatter(x_geo, y_geo, c=z_geo)
    plt.show()

    z_geo = griddata((x_geo, y_geo), z_geo, (x_msl, y_msl), method="linear")
    z_geo = z_geo.reshape(x_msl.shape)

    fout = file_geo.replace("01deg.gdf", "5km.h5")
    h5save(fout, {"geoid": z_geo, "x": x_msl, "y": y_msl}, "w")
else:
    print("loading geoid ...")
    (z_geo,) = h5read(file_geo.replace("01deg.gdf", "5km.h5"), ["geoid"])

# Get Global MDT

if 0:
    print("loading GMDT ...")
    lon_gmdt, lat_gmdt, z_gmdt = ncread(file_mdt, [xmdt, ymdt, zmdt])
    z_gmdt = np.squeeze(z_gmdt)
    lat_gmdt, z_gmdt = np.flipud(lat_gmdt), np.flipud(z_gmdt)

    (ii,) = np.where(lat_gmdt < -20)  # reduce to match MSL grid
    lat_gmdt, z_gmdt = lat_gmdt[ii], z_gmdt[ii, :]

    print("regriding GMDT onto MSL grid ...")
    LON_gmdt, LAT_gmdt = np.meshgrid(lon_gmdt, lat_gmdt)
    X_gmdt, Y_gmdt = transform_coord(4326, 3031, LON_gmdt, LAT_gmdt)

    # NOTE 1: Must fillin NaNs for regridding
    # NOTE 2: Use a big number so it's easy to filter those values later
    z_min, z_max = np.nanmin(z_gmdt), np.nanmax(z_gmdt)
    z_gmdt = np.ma.filled(z_gmdt, -9999)

    z_gmdt_grid = regrid2d(X_gmdt, Y_gmdt, z_gmdt, x_msl, y_msl)

    z_gmdt_grid[(z_gmdt_grid < z_min) | (z_gmdt_grid > z_max)] = np.nan

    plt.matshow(z_gmdt_grid)

    h5save(
        file_mdt.replace(".nc", "_5km.h5"),
        {"mdt": z_gmdt_grid, "x": x_msl, "y": y_msl},
        "w",
    )

    z_gmdt = z_gmdt_grid
else:
    print("loading regrided GMDT ...")
    (z_gmdt,) = h5read(file_mdt.replace(".nc", "_5km.h5"), ["mdt"])

# Compute MDT from Tom's MSL (remove Geoid)
z_mdt = z_msl - z_geo

plt.matshow(z_gmdt, vmin=-2, vmax=1.5)
plt.colorbar()

# Remove coastal data from GMDT
lon_mdt, lat_mdt = transform_coord(3031, 4326, x_msl, y_msl)
z_gmdt_open = z_gmdt.copy()
z_gmdt_open[lat_mdt < -60] = np.nan

z_diff = z_mdt - z_gmdt_open
z_diff = z_diff[~np.isnan(z_diff)]  # 2D -> 1D

# plt.matshow(z_geo)
# plt.colorbar()
# plt.matshow(z_gmdt_open, vmin=-2, vmax=1.5)
# plt.colorbar()
# plt.matshow(z_mdt, vmin=-2, vmax=1.5)
# plt.colorbar()

# plt.hist(z_diff, bins=100)

print("Mean offset:", np.mean(z_diff))
print("Median offset:", np.median(z_diff))

# Remove offset between MDTs
offset = np.nanmean(z_diff)
z_mdt -= offset

# --- Extend MDT w/Gaussian average tappering to zero at the GL --- #

# Get Land mask for MDT
print("generating MDT mask ...")
mask = get_mask(file_msk, x_msl, y_msl)  # grounded ice

vmin, vmax = np.nanmin(z_mdt), np.nanmax(z_mdt)

plt.matshow(z_mdt, cmap="RdBu", vmin=vmin, vmax=vmax)
plt.title("Before")
plt.matshow(z_gmdt, cmap="RdBu", vmin=vmin, vmax=vmax)
plt.title("Before")

print("extending MDT and GMDT ...")
# NOTE: Gaussian widths/range optimized for 5 km grid
widths = [3, 5, 7, 9, 11]
z_mdt = extend_field(z_mdt, mask, gauss_widths=widths, taper=None)
z_gmdt = extend_field(z_gmdt, mask, gauss_widths=widths, taper=None)

# Smooth artefacts including GLs with zeros over land
z_mdt[mask == 1] = 0.0
z_gmdt[mask == 1] = 0.0

z_mdt = convolve(z_mdt, Gaussian2DKernel(3), boundary="extend")
z_gmdt = convolve(z_gmdt, Gaussian2DKernel(3), boundary="extend")

z_mdt[mask == 1] = np.nan
z_gmdt[mask == 1] = np.nan

plt.matshow(z_mdt - np.nanmean(z_mdt), cmap="RdBu", vmin=-1.5, vmax=1.5)
plt.title("After")
plt.colorbar()
plt.matshow(z_gmdt - np.nanmean(z_gmdt), cmap="RdBu", vmin=-1.5, vmax=1.5)
plt.title("After")
plt.colorbar()

# --- Regrid all fields to Cube grid and save --- #

# Regrid fields
print("Regridding MDT and GMDT to Cube grid ...")
z_mdt[np.isnan(z_mdt)] = 0.0
z_gmdt[np.isnan(z_gmdt)] = 0.0
z_mdt = regrid2d(x_msl, y_msl, z_mdt, X, Y)
z_gmdt = regrid2d(x_msl, y_msl, z_gmdt, X, Y)

# Regrid geoid from original point data
print("Regridding Geoid to Cube grid ...")
lon_geo, lat_geo, z_geo = np.loadtxt(
    file_geo, usecols=(xgeo, ygeo, zgeo), unpack=True
)
x_geo, y_geo = transform_coord(4326, 3031, lon_geo, lat_geo)
z_geo = griddata((x_geo, y_geo), z_geo, (X, Y), method="linear")
z_geo = z_geo.reshape(X.shape)

# Compute back MSL
z_msl = z_geo + z_mdt
z_gmsl = z_geo + z_gmdt

# Mask land using cube mask
(mask_grounded,) = h5read(file_cube, ["mask_grounded"])
z_mdt[mask_grounded == 1] = np.nan
z_gmdt[mask_grounded == 1] = np.nan
z_msl[mask_grounded == 1] = np.nan
z_gmsl[mask_grounded == 1] = np.nan

# Save all grids

if 1:
    data = {
        "geoid": z_geo,
        "mdt": z_mdt,
        "gmdt": z_gmdt,
        "msl": z_msl,
        "gmsl": z_gmsl,
    }
    h5save(file_cube, data, "a")
    print("all grids saved")

# Get SLT

if 1:
    print("regridding SLT ...")
    lon_slt, lat_slt, z_slt = ncread(file_slt, [xslt, yslt, zslt])
    # lat_slt, z_slt = np.flipud(lat_slt), np.flipud(z_slt)

    z_slt *= 1e-3

    (ii,) = np.where(lat_slt < -20)
    lat_slt, z_slt = lat_slt[ii], z_slt[ii, :]

    z_slt = np.ma.filled(z_slt, np.nan)

    widths = [7, 9, 11]
    z_slt = extend_field(z_slt, gauss_widths=widths)

    plt.matshow(z_slt)

    LON_slt, LAT_slt = np.meshgrid(lon_slt, lat_slt)

    x_slt, y_slt = transform_coord(4326, 3031, LON_slt, LAT_slt)

    z_slt = np.ma.filled(z_slt, 0)

    z_slt = regrid2d(x_slt, y_slt, z_slt, X, Y)

    z_slt[mask_grounded == 1] = 0.0

    z_slt = convolve(z_slt, Gaussian2DKernel(3), boundary="extend")

    z_slt[mask_grounded == 1] = np.nan

    plt.matshow(z_slt, cmap="RdBu", vmin=-0.003, vmax=0.003)
    plt.colorbar()

    h5save(file_cube, {"slt": z_slt}, "a")
    print("SLT grid saved")


fig = plt.figure(figsize=[12, 5])
proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)
ax1 = plt.subplot(1, 3, 1, projection=proj)
ax2 = plt.subplot(1, 3, 2, projection=proj)
ax3 = plt.subplot(1, 3, 3, projection=proj)

vmin, vmax = np.nanmin(z_mdt), np.nanmax(z_mdt)

m1 = ax1.pcolormesh(X, Y, z_geo, cmap="RdBu", rasterized=True)
m2 = ax2.pcolormesh(
    X, Y, z_mdt, cmap="RdBu", rasterized=True, vmin=vmin, vmax=vmax
)
m3 = ax3.pcolormesh(
    X, Y, z_gmdt, cmap="RdBu", rasterized=True, vmin=vmin, vmax=vmax
)

plt.colorbar(m1, shrink=0.4, ax=ax1, orientation="horizontal")
plt.colorbar(m2, shrink=0.4, ax=ax2, orientation="horizontal")
plt.colorbar(m3, shrink=0.4, ax=ax3, orientation="horizontal")

ax1.coastlines(resolution="50m", color="black", linewidth=1)
ax2.coastlines(resolution="50m", color="black", linewidth=1)
ax3.coastlines(resolution="50m", color="black", linewidth=1)

plt.matshow(z_mdt - z_gmdt, cmap="RdBu")
plt.title("Difference")

plt.matshow(z_slt, cmap="RdBu")
plt.title("SLT")

plt.show()

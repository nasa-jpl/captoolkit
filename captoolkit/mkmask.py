"""
Make raster mask.

1. From polygon(s): shapefile (vector) -> hdf5 (raster)
2. From raster: mask1 -> Euclidean dist -> mask2

MEASURES v2 Boundaries (poly and raster):

    https://nsidc.org/data/nsidc-0709

"""
import sys

import cartopy.crs as ccrs
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
# import netCDF4
import pyproj
import regionmask
import xarray as xr

from utils import make_grid

# from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature

HOME = "/Users/paolofer"

# To get coords from
FILE_RASTER = HOME + "/work/melt/data/FULL_CUBE_v3.h5"

# To save mask
# FILE_OUT = HOME + "/work/melt/paper/data/mask_boundaries_1992_2012_3km.h5"
# FILE_OUT = HOME + "/work/melt/paper/data/mask_boundaries_2012_2018_3km.h5"
FILE_OUT = HOME + "/work/melt/paper/data/mask_buffer_floating.h5"

# SIO v2
# fname = (
#     "/Users/paolofer/data/masks/scripps_new/scripps_antarctica_polygons_v2/"
#     "scripps_antarctica_polygons_v2.shp"
# )
# fname = (
#     "/Users/paolofer/data/masks/scripps_new/Coastline_high_res_polygon/"
#     "Coastline_high_res_polygon.shp"
# )

# MEASURES v1
# fname = (
#     "/Users/paolofer/data/masks/scripps_new/IceShelf_Antarctica/"
#     "IceShelf_Antarctica_v1.shp"
# )

# MEASURES v2
# fname = (
#     r"/Users/paolofer/data/masks/measures_rignot/128409162/"
#     "IceShelf_Antarctica_v02.shp"
# )
# fname = (
#     r"/Users/paolofer/data/masks/measures_rignot/128409163/"
#     "GroundingLine_Antarctica_v02.shp"
# )
# fname = (
#     r"/Users/paolofer/data/masks/measures_rignot/128409161/"
#     "Coastline_Antarctica_v02.shp"
# )
# Original IceBoundaries from measures
# fname = (
#     HOME+"/data/masks/measures_rignot/128409166/"
#     "IceBoundaries_Antarctica_v02.shp"
# )
# NOTE: New IceBoundaries EDITED
# fname = (
#     r"/Users/paolofer/data/masks/measures_rignot/128409164/"
#     "Basins_IMBIE_Antarctica_v02.shp"
# )
fname = (
    # "/Users/paolofer/work/melt/analyses/qgis/ant_boundaries_edited_1992_2012/"
    # "ant_boundaries_edited_1992_2012.shp"
    "/Users/paolofer/work/melt/analyses/qgis/ant_boundaries_edited_2012_2018/"
    "ant_boundaries_edited_2012_2018.shp"
)

# GROUNDED PYLYGONS
# fname = (
#     HOME + "/work/melt/analyses/qgis/grounded-polygons/"
#     "grounded-polygons.shp"
# )


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            try:
                f[k] = np.squeeze(v)
            except IOError:
                f[k][:] = np.squeeze(v)


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


########################################################
# --- Create raster mask from polygons (shapefile) --- #
########################################################

# NOTE: This is the preferred method

if 0:
    gdf = gpd.read_file(fname)
    print(gdf)

    # Plot to check

    if 0:
        gdf.plot(edgecolor="k")
        plt.show()
        sys.exit()

    # Filter (reduce dataframe to specific polygon)

    if 0:
        gdf = gdf[(gdf.NAME == "Ross_West")]

    numbers = gdf.index
    names = gdf.NAME
    abbrevs = gdf.NAME
    outlines = gdf.geometry

    print("creating region mask ...")
    region_mask = regionmask.Regions_cls(
        "Antarctica", numbers, names, abbrevs, outlines
    )

    """
    crs = ccrs.SouthPolarStereo(central_longitude=0.0, true_scale_latitude=-71)
    region_mask.plot(proj=crs)
    plt.show()
    sys.exit()
    """

    # Get x/y coords to define mask resolution

    if 1:
        # Load x/y from CUBE file
        print("loading melt cube x/y ...")
        x, y = h5read(FILE_RASTER, ["x", "y"])

        # Load x/y from velocity file
        # fn = (
        #     HOME+"/data/velocity/merged/"
        #     "ANT_G0240_0000_PLUS_450m_v2.h5"
        # )
        # print("loading velocity x/y ...")
        # x, y = h5read(fn, ["x", "y"])

    elif 0:
        # Make x/y from prescribed values: ITS_LIVE grid
        print("making grid coordinates ...")
        x, y = make_grid(-2677927.5, 2816152.5, -2153752.5, 2259367.5, 240, 240)

    elif 0:
        # Make x/y from prescribed values: MELT PAPER
        print("making grid coordinates ...")
        x, y = make_grid(-2677927.5, 2816152.5, -2153752.5, 2259367.5, 1500, 1500)

    print("creating raster mask ...")
    mask = region_mask.mask(x, y, xarray=False)

    # Save mask

    if 1:
        names = list(names)
        names[0] = "Islands"
        """
        names = [
            n.encode("ascii", "ignore") for n in names
        ]  # unicode -> byte string
        """

        FILE_OUT_TXT = FILE_OUT.replace(".h5", ".txt")

        # h5save(FILE_RASTER, {"basins_mask": mask, "basins_name": names}, "a")
        h5save(FILE_OUT, {"mask": mask, "x": x, "y": y})

        np.savetxt(FILE_OUT_TXT, np.column_stack((numbers, names)), fmt="%s")

        print("out ->", FILE_OUT)
        print("out ->", FILE_OUT_TXT)

    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    sys.exit()

################################################################
# --- Rasterized ice shelf polygons (alternative approach) --- #
################################################################

if 0:

    import rasterio
    from rasterio import features
    from rasterio.mask import mask

    # from rasterio.plot import show

    shp_fn = (
        "/Users/paolofer/data/masks/measures_rignot/128409162/"
        "IceShelf_Antarctica_v02.shp"
    )
    rst_fn = (
        "/Users/paolofer/data/masks/measures_rignot/128409146/"
        "Mask_Antarctica_v02.tif"
    )
    out_fn = (
        "/Users/paolofer/data/masks/measures_rignot/128409146/"
        "IceShelf_Antarctica_v02.tif"
    )

    shelves = gpd.read_file(shp_fn)
    raster = rasterio.open(rst_fn)

    shelves = shelves[shelves.NAME == "Pine_Island"]

    # ax = shelves.plot(facecolor='none', edgecolor='blue')
    # ax = show(raster)
    # plt.show()

    meta = raster.meta.copy()
    # meta.update(compress='lzw')
    print(meta)

    with rasterio.open(out_fn, "w+", **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of (geom, value)
        # pairs to use in rasterizing
        shapes = ((geom, 1) for geom, value in zip(shelves.geometry, shelves.index))

        burned = features.rasterize(
            shapes=shapes, fill=0, out=out_arr, transform=out.transform
        )
        out.write_band(1, burned)

    # ax = shelves.plot()
    plt.matshow(burned)
    plt.colorbar()
    plt.show()

################################################
# --- Make buffer mask (distance treshold) --- #
################################################

if 0:

    if 0:

        # From geotiff raster
        fname = (
            r"/Users/paolofer/data/masks/measures_rignot/128409146/"
            "Mask_Antarctica_v02.tif"
        )

        import rasterio

        img = rasterio.open(fname).read(1)

        floating_mask = np.zeros_like(img)
        grounded_mask = np.zeros_like(img)
        ocean_mask = np.zeros_like(img)

        floating_mask[img == 125] = 1
        grounded_mask[img == 255] = 1
        ocean_mask[img != 255] = 1

    else:

        # From hdf5 raster
        shelves_mask, coastline_mask = h5read(
            # FILE_RASTER, ["ice_shelf_mask", "coastline_mask", "h_mean_cs2"]
            FILE_RASTER, ["h_mean_cs2", "dh_ground_johan"]
        )

        coastline_mask = np.nanmean(coastline_mask, axis=2)

        # coastline_mask ^= 1  # flip 0->1
        shelves_mask[~np.isnan(shelves_mask)] = 1
        shelves_mask[np.isnan(shelves_mask)] = 0
        shelves_mask = shelves_mask.astype("i4")
        grounded_mask = coastline_mask.copy()
        grounded_mask[shelves_mask == 1] = 0

        # Plot to check

        if 0:
            plt.matshow(coastline_mask)
            plt.colorbar()
            plt.matshow(shelves_mask)
            plt.colorbar()
            plt.matshow(grounded_mask)
            plt.colorbar()
            plt.show()
            sys.exit()

    print("calculating euclidean dists ...")
    from scipy import ndimage

    buffer = np.zeros_like(shelves_mask)

    if 0:
        # Calculate 3, 6, 9.. km buffer for coastline perimeter
        # sampling = pixel_size (define units)
        distance = ndimage.distance_transform_edt(coastline_mask, sampling=3000)

        buffer[(distance > 0) & (distance <= 3000)] = 1
        buffer[(distance > 3000) & (distance <= 6000)] = 2
        buffer[(distance > 6000) & (distance <= 9000)] = 3
        buffer[(distance > 9000) & (distance <= 12000)] = 4
        buffer[(distance > 12000) & (distance <= 15000)] = 5

        # h5save(FILE_RASTER, {'mask_buffer_coastline': buffer}, 'a')
        print("mask saved.")

    if 1:
        # Calculate 3, 6, 9.. km buffer for floating ice shelves
        # sampling = pixel_size (define units)
        distance = ndimage.distance_transform_edt(shelves_mask, sampling=3000)

        buffer[(distance > 0) & (distance <= 3000)] = 1
        buffer[(distance > 3000) & (distance <= 6000)] = 2
        buffer[(distance > 6000) & (distance <= 9000)] = 3
        buffer[(distance > 9000) & (distance <= 12000)] = 4
        buffer[(distance > 12000) & (distance <= 15000)] = 5

        h5save(FILE_RASTER, {"mask_buffer_floating2": buffer}, "a")
        print("mask saved.")

    if 0:
        # Calculate 3, 6, 50, 100 km buffer for grounded ice
        # sampling = pixel_size (define units)
        distance = ndimage.distance_transform_edt(grounded_mask, sampling=3000)

        buffer[(distance > 0) & (distance <= 3000)] = 1
        buffer[(distance > 3000) & (distance <= 6000)] = 2
        buffer[(distance > 6000) & (distance <= 50000)] = 3
        buffer[(distance > 50000) & (distance <= 100000)] = 4

        h5save(FILE_RASTER, {"mask_buffer_grounded": buffer}, "a")
        print("mask saved.")

    plt.matshow(buffer)
    plt.colorbar()
    plt.matshow(distance)
    plt.colorbar()
    plt.show()
    sys.exit()

################################################
# --- Plot Antarctic boundaries (polygons) --- #
################################################

if 0:
    fname = (
        r"/Users/paolofer/data/masks/measures_rignot/128409162/"
        "IceShelf_Antarctica_v02.shp"
    )
    # fname = (
    #     r"/Users/paolofer/data/masks/measures_rignot/128409161/"
    #     "Coastline_Antarctica_v02.shp"
    # )
    # fname = (
    #     r"/Users/paolofer/data/masks/measures_rignot/128409163/"
    #     "GroundingLine_Antarctica_v02.shp"
    # )
    # fname = (
    #     r"/Users/paolofer/data/masks/measures_rignot/128409164/"
    #     "Basins_IMBIE_Antarctica_v02.shp"
    # )
    # fname = (
    #     r"/Users/paolofer/data/masks/measures_rignot/128409165/"
    #     "Basins_Antarctica_v02.shp"
    # )
    # fname = (
    #     r"/Users/paolofer/data/masks/measures_rignot/128409166/"
    #     "IceBoundaries_Antarctica_v02.shp"
    # )
    gdf = gpd.read_file(fname)
    print(gdf.head())

    # Calcualte area of polygons
    gdf["Area"] = gdf["geometry"].area / 10 ** 6  # m2 -> km2
    print(gdf.head())
    print("Total Area:", gdf["Area"].sum())

    # gdf.plot(column='Area', categorical=False, legend=False)
    # plt.title('Areas')

    # geopandas -> cartopy

    # Here's what the plot looks like in GeoPandas
    gdf = gdf.to_crs(epsg=3031)

    ax = gdf.plot(column="Area", categorical=True, cmap="flag")

    # Calculate centroids
    gdf["geometry"].centroid.plot(ax=ax, markersize=5, color="r")

    plt.title("GeoPandas")

    """
    # Define the CartoPy CRS object.
    crs = ccrs.SouthPolarStereo(central_longitude=0.0, true_scale_latitude=-71)
    fig, ax = plt.subplots(subplot_kw={'projection': crs})
    ax.add_geometries(gdf['geometry'], crs=crs)
    ax.set_extent([-180, 180, -64, -90], ccrs.PlateCarree())
    plt.title('Cartopy')
    plt.show()
    """

    plt.show()
    sys.exit()

######################################################
# --- Create region mask for xarray (regionmask) --- #
######################################################

if 0:
    fname = (
        r"/Users/paolofer/data/masks/measures_rignot/128409162/"
        "IceShelf_Antarctica_v02.shp"
    )
    gdf = gpd.read_file(fname)

    if 1:
        # Filter (reduce dataframe)
        gdf = gdf[gdf.NAME == "Pine_Island"]

    # Transform polygons coords x/y -> lon/lat
    gdf_wgs84 = gdf.to_crs({"init": "epsg:4326"})

    numbers = gdf_wgs84.index
    names = gdf_wgs84.NAME
    abbrevs = gdf_wgs84.NAME
    outlines = gdf_wgs84.geometry  # lon/lat

    # Mask function (lon/lat)
    shelf_mask = regionmask.Regions_cls(
        "Antarctic Ice Shelves", numbers, names, abbrevs, outlines
    )

    # Filter regions

    if 0:
        regions = list(gdf[gdf.Regions == "Peninsula"]["NAME"])
    else:
        regions = ["Pine_Island"]

    print(regions)

    print("loading hdf5 ...")
    x, y, t, dHdt_melt = h5read(FILE_RASTER, ["x", "y", "t", "H"])
    X, Y = np.meshgrid(x, y)
    lon, lat = transform_coord(3031, 4326, X, Y)

    print("creating xarray ...")
    da = xr.DataArray(
        dHdt_melt,
        dims=["y", "x", "time"],
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
            "time": t,
            "x": x,
            "y": y,
        },
    )
    print(da)

    print("creating mask ...")
    mask = shelf_mask.mask(da, lon_name="lon", lat_name="lat")
    # mask.plot()

    print("applying mask ...")
    da_pig = da.where(mask == shelf_mask.map_keys("Pine_Island"))  # by abbrev

    print("calculating mean ...")
    ts_pig = da_pig.mean(dim=("y", "x"))

    ts_pig.plot.line(label="PIG")
    plt.show()
    sys.exit()

    """ Plot """

    # Coastline
    fname2 = (
        r"/Users/paolofer/data/masks/measures_rignot/128409161/"
        "Coastline_Antarctica_v02.shp"
    )
    gdf_coast = gpd.read_file(fname)

    # GL
    fname3 = (
        r"/Users/paolofer/data/masks/measures_rignot/128409163/"
        "GroundingLine_Antarctica_v02.shp"
    )
    gdf_gl = gpd.read_file(fname3)

    # Define the CartoPy CRS object.
    crs = ccrs.SouthPolarStereo(central_longitude=0.0, true_scale_latitude=-71)
    ax = plt.subplot(111, projection=crs)
    ax.set_global()

    da_pig.isel(time=10).plot.pcolormesh(vmin=-10, vmax=10, cmap="RdBu", ax=ax)

    ax = shelf_mask.plot(
        proj=crs, coastlines=False, regions=regions, add_ocean=False, ax=ax
    )

    ax.add_geometries(
        gdf_coast.geometry, crs=crs, edgecolor="0.3", facecolor="none", linewidth=0.5
    )
    # ax.add_geometries(gdf_gl.geometry, crs=crs, edgecolor='red')

    ax.set_extent([-180, 180, -65, -90], ccrs.PlateCarree())

    plt.show()

##################################
# --- Make binary mask (0|1) --- #
##################################

if 0:

    print('Making binary mask ...')

    gdf = gpd.read_file(fname)

    x, y, mask = h5read(FILE_OUT, ['x', 'y', 'mask'])

    print(gdf)

    # Plot to check

    gdf.plot(edgecolor="k")
    plt.matshow(mask)

    # Reduce dataframe to specific polygons

    if 1:
        gdf = gdf[(gdf.TYPE == "FL")]

    numbers = gdf.index
    names = gdf.NAME

    for n in numbers:
        mask[mask == n] = -1

    mask[(mask != -1) & ~np.isnan(mask)] = 0
    mask[mask == -1] = 1

    gdf.plot(edgecolor="k")
    plt.matshow(mask)
    plt.show()

    # h5save(FILE_OUT, {'mask_floating': mask})
    # print('saved.')

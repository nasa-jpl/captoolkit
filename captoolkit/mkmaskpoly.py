"""
Make raster mask.

1. From polygon(s): shapefile (vector) -> hdf5 (raster)

"""
import sys

import h5py
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import pyproj
import regionmask

# import cartopy.crs as ccrs
# import netCDF4

from utils import make_grid

# from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature

HOME = "/Users/paolofer"

FILE_RASTER = HOME + "/work/melt/data/FULL_CUBE_v3_REDUCED.h5"

# GROUNDED PYLYGONS
fname = (
    HOME + "/work/melt/analyses/qgis/grounded-polygons/"
    "grounded-polygons.shp"
)


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode="a"):
    """Generic HDF5 writer.

    vardict : {'name1': var1, 'name2': va2, 'name3': var3}
    """
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(v)
                print("dataset updated")
            else:
                f[k] = np.squeeze(v)
                print("dataset created")


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

if 1:
    gdf = gpd.read_file(fname)
    print(gdf)

    # Plot to check
    if 0:
        gdf.plot(edgecolor="k")
        plt.show()
        sys.exit()

    # Filter (reduce dataframe)

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

    # Get x/y coords for mask resolution
    if 1:
        # Load x/y from CUBE file
        print("loading melt cube x/y ...")
        x, y = h5read(FILE_RASTER, ["x", "y"])
    else:
        # Make x/y from prescribed values
        print("making grid coordinates ...")
        x, y = make_grid(
            -2677927.5, 2816152.5, -2153752.5, 2259367.5, 240, 240
        )

    print("creating raster mask ...")
    mask = region_mask.mask(x, y, xarray=False)

    # Save mask
    if 1:
        names = list(names)

        outfile_h5 = fname.replace(".shp", ".h5")
        outfile_txt = fname.replace(".shp", ".txt")

        h5save(FILE_RASTER, {"mask_grounded_polygons": mask})
        h5save(outfile_h5, {"mask": mask, "x": x, "y": y})
        np.savetxt(
            outfile_txt, np.column_stack((numbers, names)), fmt="%s",
        )
        print("out ->", outfile_h5)
        print("out ->", outfile_txt)

    plt.imshow(mask)
    plt.colorbar()
    plt.show()

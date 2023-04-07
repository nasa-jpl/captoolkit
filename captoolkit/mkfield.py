"""
Make 2d fields for ploting maps.

- Reduce 3D -> 2D
- Regrid 2D fields
- Make raster masks
- Save to fields.h5

"""
import warnings

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

from utils import get_trend, get_accel, h5read, h5save, make_grid

warnings.filterwarnings("ignore")


print("loading data ...")

fname = "/Users/paolofer/work/melt/data/FULL_CUBE_v3_REDUCED.h5"

fmask = (
    "/Users/paolofer/work/melt/analyses/qgis/ant_boundaries_edited_1992_001/"
    "ant_boundaries_edited_1992_001.shp"
)

OFILE = "../data/fields.h5"


# --- Load 3D data --- #

if 1:
    vnames = [
        "x",
        "y",
        "t",
        "dHdt_net9",
        "dHdt_melt9",
        "dh_ground_johan",
        "mask_floating",
        "mask_grounded",
    ]

    (
        x,
        y,
        t,
        dHdt_net,
        dHdt_melt,
        dh_ground,
        mask_float,
        mask_ground,
    ) = h5read(fname, vnames)

    # Create xarray
    ds = xr.Dataset(
        {
            "dHdt_net": (("y", "x", "t"), dHdt_net),
            "dHdt_melt": (("y", "x", "t"), dHdt_melt),
            "dh_ground": (("y", "x", "t"), dh_ground),
        },
        coords={
            "y": y,
            "x": x,
            "t": t,
            "mask_float": (("y", "x"), mask_float),
            "mask_ground": (("y", "x"), mask_ground),
        },
    )

    print(ds)

# Regridding coordinates
x_new, y_new = make_grid(
    # -2677927.5, 2816152.5, -2153752.5, 2259367.5, 960, 960
    -2677927.5,
    2816152.5,
    -2153752.5,
    2259367.5,
    1500,
    1500,
)

# --- Make mask (burn polygons) --- #

# NOTE: If making mask not needed, regrid fields below

if 0:
    print("making mask ...")

    import regionmask

    gdf = gpd.read_file(fmask)
    print(gdf)

    gdf.plot(edgecolor="k")
    plt.show()

    numbers = gdf.index
    names = gdf.NAME
    abbrevs = gdf.NAME
    outlines = gdf.geometry

    print("creating region mask ...")
    region_mask = regionmask.Regions_cls(
        "Antarctica", numbers, names, abbrevs, outlines
    )

    print("creating raster mask ...")
    mask = region_mask.mask(x_new, y_new, xarray=False)
    mask = np.flipud(mask)

    names = list(names)
    names[0] = "Islands"
    names = [
        n.encode("ascii", "ignore") for n in names
    ]  # unicode -> byte string

    h5save(
        OFILE,
        {
            "mask": mask,
            "name": names,
            "number": numbers,
            "x": x_new,
            "y": y_new,
        },
        "a",
    )
    print("saved.")

    plt.imshow(mask)
    plt.show()

# --- Interpolate/Regrid mask --- #

if 1:
    fmask_float = "/Users/paolofer/data/masks/jpl/ANT_floatingice_240m.tif.h5"
    fmask_ground = "/Users/paolofer/data/masks/jpl/ANT_groundedice_240m.tif.h5"

    mask_float_hi, x_, y_ = h5read(fmask_float, ["mask", "x", "y"])
    mask_ground_hi, x_, y_ = h5read(fmask_ground, ["mask", "x", "y"])

    x_, y_ = x_[0, :], y_[:, 0]

    ds_mask = xr.Dataset(
        {
            "mask_float": (("y", "x"), mask_float_hi),
            "mask_ground": (("y", "x"), mask_ground_hi),
        },
        coords={"y": y_, "x": x_},
    )

    print(ds_mask)

    # Regrid
    ds_mask_interp = ds_mask.interp(x=x_new, y=y_new, method="nearest")


# --- Get REMA image --- #

if 0:
    fdem = "/Users/paolofer/data/dem/REMA_1km_dem_filled.tif.h5"
    ximg, yimg, img = h5read(fdem, ["x", "y", "elev"])
    img[img == -9999] = np.nan

    ds_img = xr.Dataset(
        {"rema": (("y", "x"), img)}, coords={"y": yimg, "x": ximg},
    )

    ds_img = ds_img.interp(x=x_new, y=y_new)

    h5save(OFILE, {"rema": np.flipud(ds_img.rema.values)}, "a")

    plt.pcolormesh(x_new, y_new, ds_img.rema)
    plt.show()


# --- Get LIMA image --- #

if 0:
    print("getting img from geotiff ...")
    from skimage.exposure import rescale_intensity
    import scipy.ndimage as ndi

    fimg = "/Users/paolofer/data/lima/tiff_90pct/00000-20080319-092059124.tif"
    da = xr.open_rasterio(fimg)

    # Compute a greyscale out of the rgb image
    greyscale = da.mean(dim="band")

    # Contrast stretching and filtering
    p1, p2 = np.percentile(greyscale.values, (2, 98))
    values = rescale_intensity(greyscale.values, in_range=(p1, p2))
    values = ndi.median_filterer(values, 3)

    da["greyscale"] = xr.DataArray(values, dims=("y", "x"))

    da_interp = da.interp(x=x_new, y=y_new)

    h5save(OFILE, {"lima": np.flipud(da_interp.greyscale.values)}, "a")

# --- Reduce fields --- #

if 1:
    print("computing 2D fields ...")

    def slope(x, y):
        (k,) = np.where(~np.isnan(y))
        if k.size < 3:
            return np.nan
        x_, y_ = x[k], y[k]
        return np.polyfit(x_, y_, 1)[0]

    def get_slope(x, y, dim="t"):
        """Reduce time dim: 3D -> 2D."""
        return xr.apply_ufunc(
            slope, x, y, vectorize=True, output_dtypes=[float],
        )

    print("taking mean ...")

    dHdt_net_mean = ds.dHdt_net.mean("t")
    dHdt_melt_mean = ds.dHdt_melt.mean("t")

    print("fitting trend ...")

    # FIXME: This fucking shit doesn't work... use numpy (!!)
    # dHdt_net_trend = get_slope(ds.t, ds.dHdt_net, "t").compute()
    # dHdt_melt_trend = get_slope(ds.t, ds.dHdt_melt, "t").compute()

    dHdt_net_trend = np.apply_along_axis(
        get_trend, 2, ds.dHdt_net.values, ds.t
    )
    dHdt_melt_trend = np.apply_along_axis(
        get_trend, 2, ds.dHdt_melt.values, ds.t
    )
    dhdt_ground_mean = np.apply_along_axis(
        get_trend, 2, ds.dh_ground.values, ds.t
    )
    dhdt_ground_trend = np.apply_along_axis(
        get_accel, 2, ds.dh_ground.values, ds.t
    )

    # Create xarray
    ds_fields = xr.Dataset(
        {
            "dHdt_net_mean": (("y", "x"), dHdt_net_mean),
            "dHdt_net_trend": (("y", "x"), dHdt_net_trend),
            "dHdt_melt_mean": (("y", "x"), dHdt_melt_mean),
            "dHdt_melt_trend": (("y", "x"), dHdt_melt_trend),
            "dhdt_ground_mean": (("y", "x"), dhdt_ground_mean),
            "dhdt_ground_trend": (("y", "x"), dhdt_ground_trend),
        },
        coords={"y": y, "x": x},
    )

# --- Regrid/Filter fields --- #

if 1:
    print("regridding/filtering fields ...")

    # Extend

    for field in ds_fields.keys():

        ds_fields[field].values[:] = interpolate_replace_nans(
            ds_fields[field].values, Gaussian2DKernel(2), boundary="extend"
        )

    # Regrid

    if 1:
        ds_fields = ds_fields.interp(x=x_new, y=y_new, method="linear")

    # Filter

    for field in ds_fields.keys():

        ds_fields[field].values[:] = median_filter(
            ds_fields[field].values, size=5
        )

        if "dHdt" in field:
            ds_fields[field].values[ds_mask_interp.mask_float == 0] = np.nan

        if "dhdt" in field:
            ds_fields[field].values[ds_mask_interp.mask_ground == 0] = np.nan

# --- Save fields --- #

if 1:
    h5save(
        OFILE,
        {
            "dHdt_mean": ds_fields.dHdt_net_mean.values,
            "dHdt_accel": ds_fields.dHdt_net_trend.values,
            "melt_mean": ds_fields.dHdt_melt_mean.values,
            "melt_accel": ds_fields.dHdt_melt_trend.values,
            "dhdt_ground_mean": ds_fields.dhdt_ground_mean.values,
            "dhdt_ground_accel": ds_fields.dhdt_ground_trend.values,
            "x": ds_fields.coords["x"].values,
            "y": ds_fields.coords["y"].values,
        },
        "a",
    )

    print("saved.")

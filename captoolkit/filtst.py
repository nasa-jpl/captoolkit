"""
Filter point-cloud data in space and time.

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Fernando Paolo (paolofer@jpl.nasa.gov)
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""
__version__ = 0.2

import argparse
import os
import sys
import warnings

import h5py
import numpy as np
import pyproj
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# Append suffix to output file
SUFFIX = "_STFILT"

# Time steps for binning
STEP = 3 / 12.0

# Window for binning
WINDOW = 5 / 12.0

# Num of std to filter outliers (localy)
N_STD = 10

# Absolute treshold to filter outilers (globaly)
MAX_ABS = 30

# Defaul grid spacing in x and y (km)
DXY = [3, 3]

# Defaul fixed search radius (km). If 0, lat-variable radius
RADIUS = 0

# Default min obs within search radius to compute solution
MINOBS = 25

# Default projection EPSG for solution (AnIS=3031, GrIS=3413)
PROJ = 3031

# Default njobs for parallel processing of *tiles*
NJOBS = 1


def get_args():
    # Output description of solution
    description = "Filter point-cloud data in space and time"

    # Define command-line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "ifile",
        metavar="ifile",
        type=str,
        nargs="+",
        help="file(s) to process (ASCII, HDF5 or Numpy)",
    )
    parser.add_argument(
        "-d",
        metavar=("dx", "dy"),
        dest="dxy",
        type=float,
        nargs=2,
        help=("spatial resolution for grid-solution (deg or m)"),
        default=DXY,
    )
    parser.add_argument(
        "-r",
        metavar=("radius"),
        dest="radius",
        type=float,
        nargs=1,
        help=("min and max search radius (km)"),
        default=[RADIUS],
    )
    parser.add_argument(
        "-z",
        metavar="min_obs",
        dest="minobs",
        type=int,
        nargs=1,
        help=("minimum obs to compute solution"),
        default=[MINOBS],
    )
    parser.add_argument(
        "-j",
        metavar=("epsg_num"),
        dest="proj",
        type=str,
        nargs=1,
        help=("projection: EPSG number (AnIS=3031, GrIS=3413)"),
        default=[str(PROJ)],
    )
    parser.add_argument(
        "-v",
        metavar=("t", "x", "y", "h"),
        dest="vnames",
        type=str,
        nargs=4,
        help=("name of t/lon/lat/h in the HDF5"),
        default=["t_year", "lon", "lat", "h_res"],
    )
    parser.add_argument(
        "-n",
        metavar=("n_jobs"),
        dest="njobs",
        type=int,
        nargs=1,
        help="for parallel processing of multiple tiles, optional",
        default=[NJOBS],
    )
    parser.add_argument(
        "-b",
        metavar=("e", "w", "s", "n"),
        dest="bbox",
        type=float,
        nargs=4,
        help="full bbox in case of processing tiles (for consistency)",
        default=[None],
    )

    return parser.parse_args()


def transform_coord(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (EPSG num).

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


def get_bbox(fname, key="bbox"):
    """Extract tile bbox info from file name."""
    fname = fname.split("_")  # fname -> list
    i = fname.index(key)

    return list(map(float, fname[i + 1 : i + 5]))  # m


##NOTE: For stfilter we don't want to pass a subgrid because we
# also want to filter overlapping points (outside the tile bbox).
def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """ Construct output grid-coordinates. """
    Nn = int((np.abs(ymax - ymin)) / dy) + 1  # grid dimensions
    Ne = int((np.abs(xmax - xmin)) / dx) + 1
    x_i = np.linspace(xmin, xmax, num=Ne)
    y_i = np.linspace(ymin, ymax, num=Nn)

    return np.meshgrid(x_i, y_i)


def get_limits(x, y, bbox):
    """Get indices (where) of tile limits from bbox."""
    xmin, xmax, ymin, ymax = bbox
    (i,) = np.where((y >= ymin) & (y <= ymax))
    (j,) = np.where((x >= xmin) & (x <= xmax))

    return (i[0], i[-1] + 1, j[0], j[-1] + 1)


def get_radius(x, y, r0=8, lat0=-81.5, proj=3031):
    """Radius as a linear function of latitude."""
    lon, lat = transform_coord(proj, 4326, x, y)

    return r0 * np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat0)) * 1e3


def rename_file(fname, suffix="_COR"):
    path, ext = os.path.splitext(fname)
    os.rename(fname, path + suffix + ext)


""" Helper functions. """


def subset_data(
    t, x, y, z, e, k, tlim=(1995.25, 1995.5), xlim=(-1, 1), ylim=(-1, 1)
):
    """ Subset data domain (add NaNs to undesired values). """
    tt = (t >= tlim[0]) & (t <= tlim[1])
    xx = (x >= xlim[0]) & (x <= xlim[1])
    yy = (y >= ylim[0]) & (y <= ylim[1])
    (ii,) = np.where(tt & xx & yy)

    return t[ii], x[ii], y[ii], z[ii], e[ii], k[ii]


def remove_invalid(z, variables):
    """Filter NaNs using z var."""
    (ii,) = np.where(np.isfinite(z))

    return [v[ii] for v in variables]


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """

    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def has_alpha(string):
    """Return True if any char is alphabetic."""

    return any(c.isalpha() for c in string)


def load_data(ifile, xvar, yvar, tvar, hvar, step=1):
    with h5py.File(ifile, "r") as f:
        lon = f[xvar][::step]
        lat = f[yvar][::step]
        time = f[tvar][::step]
        obs = f[hvar][::step]

    return lon, lat, time, obs


def overlap(x1, x2, y1, y2):
    """ Return True if x1-x2/y1-y2 ranges overlap. """

    return (x2 >= y1) & (y2 >= x1)


def intersect(x1, x2, y1, y2, a1, a2, b1, b2):
    """ Return True if (x1,x2,y1,y2) rectangles intersect. """

    return overlap(x1, x2, a1, a2) & overlap(y1, y2, b1, b2)


def binning(
    x,
    y,
    xmin=None,
    xmax=None,
    dx=1 / 12.0,
    window=3 / 12.0,
    interp=False,
    median=False,
):
    """Time-series binning (w/overlapping windows).

    Args:
        x,y: time and value of time series.
        xmin,xmax: time span of returned binned series.
        dx: time step of binning.
        window: size of binning window.
        interp: interpolate binned values to original x points.
    """

    if xmin is None:
        xmin = np.nanmin(x)

    if xmax is None:
        xmax = np.nanmax(x)

    steps = np.arange(xmin, xmax + dx, dx)  # time steps
    bins = [(ti, ti + window) for ti in steps]  # bin limits

    N = len(bins)
    yb = np.full(N, np.nan)
    xb = np.full(N, np.nan)
    eb = np.full(N, np.nan)
    nb = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    for i in range(N):

        t1, t2 = bins[i]
        (idx,) = np.where((x >= t1) & (x <= t2))

        if len(idx) == 0:
            continue

        ybv = y[idx]
        xbv = x[idx]

        if median:
            yb[i] = np.nanmedian(ybv)
        else:
            yb[i] = np.nanmean(ybv)

        xb[i] = 0.5 * (t1 + t2)
        eb[i] = mad_std(ybv)
        nb[i] = np.sum(~np.isnan(ybv))
        sb[i] = np.sum(ybv)

    if interp:
        yb = np.interp(x, xb, yb)
        eb = np.interp(x, xb, eb)
        sb = np.interp(x, xb, sb)
        xb = x

    return xb, yb, eb, nb, sb


def detrend_binned(x, y, order=1, dx=1 / 12, window=3 / 12.0):
    """Bin data (Med), compute trend (OLS) on binned, detrend original data"""
    x_b, y_b = binning(x, y, median=True, dx=dx, window=window, interp=False)[
        :2
    ]
    i_valid = ~np.isnan(y_b) & ~np.isnan(x_b)
    x_b, y_b = x_b[i_valid], y_b[i_valid]
    try:
        coef = np.polyfit(x_b, y_b, order)
        y_trend = np.polyval(coef, x)  # NOTE: Eval on full time
    except IOError:
        y_trend = np.zeros_like(y)

    return y - y_trend, y_trend


def get_radius_idx(x, y, x0, y0, radius, Tree):
    """ Get data within search radius (inversion cell). """

    return Tree.query_ball_point((x0, y0), radius)


def get_residuals(tc, hc, order=1, dx=3 / 12.0, window=5 / 12.0):
    hc_cycle, hc_trend = detrend_binned(
        tc, hc, order=order, dx=dx, window=window
    )  # detrend
    hc_bin = binning(
        tc, hc_cycle, dx=dx, window=window, interp=True, median=True
    )[
        1
    ]  # bin

    return hc_cycle - hc_bin  # residual


def stfilter(
    data, xxx_todo_changeme,
    radius=None,
    min_obs=25,
    n_std=3,
    step=1 / 12.0,
    window=1 / 12.0,
):

    (xi, yi) = xxx_todo_changeme
    t, x, y, z = data  # full file/tile data

    xi_uniq, yi_uniq = np.unique(xi), np.unique(yi)
    dx, dy = xi_uniq[1] - xi_uniq[0], yi_uniq[1] - yi_uniq[0]  # cell size

    # Create output container
    i_invalid = np.full(z.shape, True, dtype=bool)

    # Get data limits
    xmin_d, xmax_d, ymin_d, ymax_d = (
        np.nanmin(x),
        np.nanmax(x),
        np.nanmin(y),
        np.nanmax(y),
    )

    print("building KDTree ...")
    Tree = cKDTree(np.column_stack((x, y)))

    print("entering spatial loop ...")

    for i_node in range(xi.shape[0]):

        if i_node % 500 == 0:
            print("node:", i_node)

        x0, y0 = xi[i_node], yi[i_node]  # prediction pt (grid node)

        ##NOTE: For topofit we don't want to pass a subgrid because we
        # also want to detrend overlapping points (outside the tile bbox).

        if radius == 0:
            radius = get_radius(
                x0, y0
            )  ##FIXME: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # If search radius falls outside tile bbox, skip

        if not intersect(
            x0 - radius,
            x0 + radius,
            y0 - radius,
            y0 + radius,
            xmin_d,
            xmax_d,
            ymin_d,
            ymax_d,
        ):
            continue

        # Get index of pts within search radius
        i_cap = np.array(Tree.query_ball_point((x0, y0), radius))

        if len(i_cap) < min_obs:
            continue

        tc, xc, yc, zc = t[i_cap], x[i_cap], y[i_cap], z[i_cap]

        """ Temporal filtering """

        zc_res = get_residuals(tc, zc, order=1, dx=step, window=window)

        cond1 = (
            np.abs(zc_res - np.nanmedian(zc_res)) < mad_std(zc_res) * n_std
        )  # boolean full radius
        cond2 = (
            (xc > x0 - dx / 2.0)
            & (xc < x0 + dx / 2.0)
            & (yc > y0 - dy / 2.0)
            & (yc < y0 + dy / 2.0)
        )  # within cell

        i_cell_valid = np.array(cond1 & cond2)

        i_cap = i_cap[i_cell_valid]  # update cap pts -> valid cell points

        i_invalid[i_cap] = False  # update full tile pts

    return i_invalid


def stfilter2(
    data, xxx_todo_changeme1,
    radius=None,
    min_obs=25,
    n_std=3,
    step=1 / 12.0,
    window=1 / 12.0,
):
    """Does the same as above but globally."""
    (xi, yi) = xxx_todo_changeme1
    t, x, y, z = data  # full file/tile data

    # Create output container
    i_invalid = np.full(z.shape, True, dtype=bool)

    # Get data limits
    xmin_d, xmax_d, ymin_d, ymax_d = (
        np.nanmin(x),
        np.nanmax(x),
        np.nanmin(y),
        np.nanmax(y),
    )

    tc, xc, yc, zc = t.copy(), x.copy(), y.copy(), z.copy()

    zc_res = get_residuals(tc, zc, order=1, dx=step, window=window)

    i_invalid = (
        np.abs(zc_res - np.nanmedian(zc_res)) > mad_std(zc_res) * n_std
    )  # boolean full radius

    return i_invalid


def absfilter(t, h, max_abs=50, order=1, step=1 / 12.0, window=1 / 12.0):
    hc_res = detrend_binned(t, h, order=order, dx=step, window=window)[0]
    i_invalid = np.abs(hc_res - np.nanmedian(hc_res)) > max_abs

    return i_invalid


def main(ifile, args):

    print(ifile)

    # ifile = args.ifile[0]
    bbox = args.bbox[:]
    vnames = args.vnames[:]
    dx = args.dxy[0] * 1e3
    dy = args.dxy[1] * 1e3
    radius = args.radius[0] * 1e3
    proj = args.proj[0]

    min_obs = MINOBS

    tvar, xvar, yvar, zvar = vnames

    print("loading data ...")
    time, lon, lat, obs = load_data(ifile, tvar, xvar, yvar, zvar, step=1)

    if len(obs) < MINOBS:
        return

    # Convert to stereo coordinates
    x, y = transform_coord(4326, proj, lon, lat)

    xmin_d, xmax_d, ymin_d, ymax_d = (
        np.nanmin(x),
        np.nanmax(x),
        np.nanmin(y),
        np.nanmax(y),
    )

    if bbox[0]:
        xmin, xmax, ymin, ymax = bbox
    else:
        # If no bbox given, limits are defined by data
        xmin, xmax, ymin, ymax = xmin_d, xmax_d, ymin_d, ymax_d

    # Generate 2D prediction grid
    Xi, Yi = make_grid(xmin, xmax, ymin, ymax, dx, dy)
    xi, yi = Xi.ravel(), Yi.ravel()

    # Filter data in sapace and time (locally)
    i_invalid = stfilter(
        [time, x, y, obs],
        (xi, yi),
        radius=radius,
        min_obs=min_obs,
        n_std=N_STD,
        step=STEP,
        window=WINDOW,
    )
    """

    i_invalid = stfilter2([time, x, y, obs], (xi,yi), radius=radius,
                         min_obs=min_obs, n_std=N_STD, step=STEP,
                         window=WINDOW)
    """

    # Filter hard treshold (globally)
    i_invalid2 = absfilter(
        time, obs, max_abs=MAX_ABS, step=STEP, window=WINDOW
    )

    # Plot

    if 0:
        import matplotlib.pyplot as plt

        plt.plot(time, obs, ".")

    obs[i_invalid] = np.nan
    obs[i_invalid2] = np.nan

    # Plot

    if 0:
        plt.plot(time, obs, ".")
        plt.show()

    # Save data

    if 1:
        with h5py.File(ifile, "a") as f:
            f[zvar][:] = obs
            f.flush()

        rename_file(ifile, suffix=SUFFIX)


# Get command line args
args = get_args()
files = args.ifile[:]
njobs = args.njobs[0]

print("parameters:")

for p in vars(args).items():
    print(p)


if njobs == 1:
    print("running serial code ...")
    [main(f, args) for f in files]

else:
    print("running parallel code (%d jobs) ..." % njobs)
    from joblib import Parallel, delayed

    Parallel(n_jobs=njobs, verbose=1)(delayed(main)(f, args) for f in files)

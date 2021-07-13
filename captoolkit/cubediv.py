"""
Calculate time-variable flux divergence.

Input:
    <- Thickness time series (3D)
    <- Velocity map (2D) or time series (3D)

Output:
    -> Full Divergence time series (3D)
    -> Stretching time series (3D)
    -> Advection time series (3D)

Notes:
    It references all time series to the REF_TIME of Freeboard/DEM

    The flux-divergence correction (for a dh/dt grid) is computed as:

        dh/dt_cor = dh/dt + div(h*v)

    where h is height at the center of dt interval,
    and v is velocity, assumed constant.

    For point (residual) height data, we then have:

        h(dt) = dh_cor(t-REF_TIME) = dh(t-REF_TIME) + div[h(t)*v] * dt

    where dt = t - REF_TIME.

    IMPORTANT: Both h_res(t) and dh/dt_dyn(t) should have the same REF_TIME!

    Slope (gradients) are saved with units of radians (see below).

Units:
    Gradient:
        slope = dy/dx [m/m] => [radians]
        slope_angle = arctan(slope) * 180/pi => [degrees]

    Divergence:
        div(h*v) = (dx,dy) * (Hu,Hv) => [m/yr]

    Spatial derivatives:
        poly fit -> delta_height / grid_spacing * window_size (m / m x n)

"""
# import sys
import argparse
import warnings

import h5py
import pyproj
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans
import matplotlib.pyplot as plt

from utils import sgolay2d

warnings.filterwarnings("ignore")


# === Edit =======================================================

FILE_CUBE = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

x_var = "x"
y_var = "y"
t_var = "t"

H_var = "H10"  # H = (h - fac) * 9.26

u_var = "u10"  # new 3D u (u for 2D)
v_var = "v10"  # new 3D v (v for 2D)

m_var = "mask_floating"

# ref time of Freeboard/DEM (see TIMESPANS.txt)
REF_TIME = 2014.0

# window size for espatial derivatives
# NOTE: 3 is equal to central differences (e.g. numpy.gradient)
WINDOW_SIZE = 5

SAVE = True

TEST = False

# NOTE: Change the name of saved variables bellow.

# ================================================================


def get_args():
    """ Get command-line arguments. """
    msg = "calculate heihgt change due to flux divergence."
    parser = argparse.ArgumentParser(description=msg)
    """
    parser.add_argument(
            'file', type=str, nargs='+',
            help='HDF5 file(s) with height (h) grid')
    """
    parser.add_argument(
        "-f",
        metavar=("vfile"),
        dest="vfile",
        type=str,
        nargs=1,
        help="NetCDF4 file with velocity (u,v) grids",
        default=[None],
    )
    parser.add_argument(
        "-o",
        metavar=("fout"),
        dest="fout",
        type=str,
        nargs=1,
        help=("output file name"),
        default=[None],
    )
    parser.add_argument(
        "-g",
        metavar=None,
        dest="grid",
        type=str,
        nargs=1,
        help=("grid resolution of fluxdiv: same as (h)eight or (v)elocity"),
        choices=("h", "v"),
        default=["h"],
    )
    parser.add_argument(
        "-w",
        metavar=("window"),
        dest="window",
        type=float,
        nargs=1,
        help=("window size (kernel) for polynomial fit"),
        default=[None],
    )
    parser.add_argument(
        "-p",
        metavar=("order"),
        dest="order",
        type=int,
        nargs=1,
        help=("degree for polynomial fit"),
        default=[None],
    )
    parser.add_argument(
        "-a",
        metavar=("x", "y", "h"),
        dest="hnames",
        type=str,
        nargs=3,
        help=("name of x/y/h variables in the HDF5 (use t=None if 2D)"),
        default=None,
    )
    parser.add_argument(
        "-b",
        metavar=("x", "y", "u", "v"),
        dest="vnames",
        type=str,
        nargs=4,
        help=("name of x/y/u/v variables in the NetCDF4"),
        default=None,
    )
    parser.add_argument(
        "-l",
        metavar=("x1", "x2", "y1", "y2"),
        dest="region",
        type=float,
        nargs=4,
        help=("geographic limits in polar stereo (m)"),
        default=None,
    )
    parser.add_argument(
        "-n",
        metavar=("njobs"),
        dest="njobs",
        type=int,
        nargs=1,
        help="for parallel processing of multiple fields",
        default=[1],
    )

    return parser.parse_args()


def print_args(args):
    print("Input arguments:")

    for arg in vars(args).iteritems():
        print(arg)


# NOTE: This function is the key to get the right units
def gradient(z, window_size=5, order=1, axis=None, dx=1.0):
    """Compute gradient using the Savitzky-Golay filter.

    Computes a smooth surface by piecewise 2D polynomial fit.

    For axis=None returns both deriv: dzdx, dzdy (reversed of numpy.gradient).

    dx is the grid spacing.

    """
    derivative = {0: "col", 1: "row", None: "both"}

    # TODO: Check the units for window > 3
    return sgolay2d(z, window_size, order, derivative[axis]) / dx


def gradient2(u, v, window_size=10, order=1, dx=1.0, dy=1.0):
    """Compute gradient of a two-component vector field.

    dx, dy are grid spacings in x and y dimensions.
    """

    u_mask = ~np.isfinite(u)
    v_mask = ~np.isfinite(v)

    u[u_mask] = 0
    v[v_mask] = 0

    dudx = gradient(u, window_size, order, axis=1, dx=dx)
    dvdy = gradient(v, window_size, order, axis=0, dx=dy)

    dudx[u_mask] = np.nan
    dvdy[v_mask] = np.nan

    # ------------------------------------------------------
    # NOTE: Uncomment for comparison to np.gradient/np.diff
    '''
    if 1:
        dudy_, dudx_ = np.gradient(u, dx)
        dvdy_, dvdx_ = np.gradient(v, dy)
    else:
        dudx_ = np.diff(u, axis=1) / dx
        dvdy_ = np.diff(v, axis=0) / dy

    std = np.nanstd(dudx)
    plt.matshow(dudx, cmap="RdBu", vmin=-std, vmax=std)
    plt.colorbar()
    plt.matshow(dudx_, cmap="RdBu", vmin=-std, vmax=std)
    plt.colorbar()
    plt.matshow(dudx[:, :-1] - dudx_, cmap="RdBu", vmin=-std / 3, vmax=std / 3)
    plt.colorbar()

    std = np.nanstd(dvdy)
    plt.matshow(dvdy, cmap="RdBu", vmin=-std, vmax=std)
    plt.colorbar()
    plt.matshow(dvdy_, cmap="RdBu", vmin=-std, vmax=std)
    plt.colorbar()
    plt.matshow(dvdy[:-1, :] - dvdy_, cmap="RdBu", vmin=-std / 3, vmax=std / 3)
    plt.colorbar()

    plt.show()
    '''
    # ------------------------------------------------------

    return dudx, dvdy


""" Utility functions. """


def find_nearest(arr, val):
    """Find index for "nearest" value.

    Parameters
    ----------
    arr : array_like, shape nd
        The array to search in (nd). No need to be sorted.
    val : scalar or array_like
        Value(s) to find.

    Returns
    -------
    out : tuple
        The index (or tuple if nd array) of nearest entry found. If `val` is a
        list of values then a tuple of ndarray with the indices of each value
        is return.

    See also
    --------
    find_nearest2

    """
    idx = []

    if np.ndim(val) == 0:
        val = np.array([val])

    for v in val:
        idx.append((np.abs(arr - v)).argmin())
    idx = np.unravel_index(idx, arr.shape)

    return idx


def rad2deg(x):
    return np.arctan(x) * 180 / np.pi


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


def div(u, v, window_size, order, dx=1.0, dy=1.0):
    """ Divergence of velocity: div(V) = (d/dx,d/dy)*(u,v) """

    # Gradient
    dudx, dvdy = gradient2(
        u, v, window_size=window_size, order=1, dx=dx, dy=dy
    )

    # Reverse sign (y coord is positive upwards)
    dvdy *= -1

    # Divergence
    div = dudx + dvdy

    return div, dudx, dvdy


def fluxdiv(h, u, v, window_size, order, dx=1.0, dy=1.0):
    """Flux divergence: div(hv) = (d/dx, d/dy) * (hu, hv).

    Args:
        h : 2D thickness field.
        u, v : 2D velocity fields.

    """
    # Flux fields
    hu = h * u
    hv = h * v

    # Gradient
    dhudx, dhvdy = gradient2(
        hu, hv, window_size=window_size, order=order, dx=dx, dy=dy
    )

    # Reverse sign (y coord is positive upwards)
    dhvdy *= -1

    # Flux divergence
    divergence = dhudx + dhvdy

    return divergence, dhudx, dhvdy, hu, hv


def stretch(h, u, v, window_size, order, dx=1.0, dy=1.0):
    """Stretching component: h * div(v) = h * [(d/dx, d/dy) * (u, v)].

    Args:
        h : 2D thickness field.
        u, v : 2D velocity fields.
    """

    # Gradient
    dudx, dvdy = gradient2(
        u, v, window_size=window_size, order=order, dx=dx, dy=dy
    )

    # Reverse sign (y coord is positive upwards)
    dvdy *= -1

    # Dynamic thinning
    stretching = h * (dudx + dvdy)

    return stretching, dudx, dvdy


def advect(h, u, v, window_size, order, dx=1.0, dy=1.0):
    """Advection component: v * grad(h) = (u, v) * (dh/dx, dh/dy).

    Args:
        h : 2D thickness field.
        u, v : 2D velocity fields.
    """

    # Gradient -> slopes
    dhdx, dhdy = gradient2(
        h, h, window_size=window_size, order=order, dx=dx, dy=dy
    )

    # Reverse sign (y coord is positive upwards)
    dhdy *= -1

    # Advection of slopes
    advection = u * dhdx + v * dhdy

    return advection, dhdx, dhdy


# To execute in parallel for each 'h' field
def get_fluxdiv(h, u, v, x, y, window=3, order=1):

    # Grid spacing
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])

    # Full flux divergence
    divhv, dhudx, dhvdy, hu, hv = fluxdiv(h, u, v, window, order, dx=dx, dy=dy)

    # Stretching component
    strhv, dudx, dvdy = stretch(h, u, v, window, order, dx=dx, dy=dy)

    # Advection component
    advhv, dhdx, dhdy = advect(h, u, v, window, order, dx=dx, dy=dy)

    # radians -> degrees (slopes)
    # dhdx = rad2deg(dhdx)
    # dhdy = rad2deg(dhdy)

    return (divhv, dhudx, dhvdy, hu, hv, strhv, dudx, dvdy, advhv, dhdx, dhdy)


def h5read(ifile, vnames):
    with h5py.File(ifile, "r") as f:
        return [f[v][()] for v in vnames]


def h5save(fname, vardict, mode="a"):
    with h5py.File(fname, mode) as f:
        for k, v in vardict.items():
            if k in f:
                f[k][:] = np.squeeze(v)
                print('dataset updated')
            else:
                f[k] = np.squeeze(v)
                print('dataset created')


def main():

    print("loading data ...")

    vnames = [x_var, y_var, t_var, H_var, u_var, v_var, m_var]
    x, y, t, H, u, v, mask = h5read(FILE_CUBE, vnames)

    mask = ~mask  # 1 -> 0

    if TEST:
        t = t[:5]
        H = H[:, :, :5]

    mask3d = np.repeat(mask[:, :, np.newaxis], H.shape[2], axis=2)

    if not TEST:
        # Extend outer boundary to avoid boundary effects
        print("extending boundaries ...")

        for k in range(H.shape[2]):
            H[:, :, k] = interpolate_replace_nans(
                H[:, :, k], Gaussian2DKernel(1), boundary="extend"
            )

        if u.ndim == 3:
            for k in range(u.shape[2]):
                u[:, :, k] = interpolate_replace_nans(
                    u[:, :, k], Gaussian2DKernel(1), boundary="extend"
                )
                v[:, :, k] = interpolate_replace_nans(
                    v[:, :, k], Gaussian2DKernel(1), boundary="extend"
                )
        else:
            u = interpolate_replace_nans(
                u, Gaussian2DKernel(1), boundary="extend"
            )
            v = interpolate_replace_nans(
                v, Gaussian2DKernel(1), boundary="extend"
            )

    # --- Generate time-evolving FLUXDIV --- #

    # Mask remaining NaNs to Zeros (new boundary-extended mask)
    H_mask = np.isnan(H)
    v_mask = np.isnan(v)
    H[H_mask] = 0
    u[v_mask] = 0
    v[v_mask] = 0

    print("output containers ...")

    # Output containers
    divHv = np.full_like(H, np.nan)
    dHudx = np.full_like(H, np.nan)
    dHvdy = np.full_like(H, np.nan)
    Hu = np.full_like(H, np.nan)
    Hv = np.full_like(H, np.nan)
    strHv = np.full_like(H, np.nan)
    dudx = np.full_like(H, np.nan)
    dvdy = np.full_like(H, np.nan)
    advHv = np.full_like(H, np.nan)
    dHdx = np.full_like(H, np.nan)
    dHdy = np.full_like(H, np.nan)
    H_div = np.full_like(H, np.nan)
    H_str = np.full_like(H, np.nan)
    H_adv = np.full_like(H, np.nan)

    # Calc one fluxdiv field per time step

    for k in range(H.shape[2]):
        print("calculating div: %d/%d" % (k + 1, H.shape[2]), "...")

        _H = H[:, :, k]  # must be thickness!

        if u.ndim == 3:
            _u = u[:, :, k]
            _v = v[:, :, k]
        else:
            _u = u
            _v = v

        (
            divHv[:, :, k],
            dHudx[:, :, k],
            dHvdy[:, :, k],
            Hu[:, :, k],
            Hv[:, :, k],
            strHv[:, :, k],
            dudx[:, :, k],
            dvdy[:, :, k],
            advHv[:, :, k],
            dHdx[:, :, k],
            dHdy[:, :, k],
        ) = get_fluxdiv(_H, _u, _v, x, y, window=WINDOW_SIZE)

    divHv[mask3d] = np.nan
    dHudx[mask3d] = np.nan
    dHvdy[mask3d] = np.nan
    Hu[mask3d] = np.nan
    Hv[mask3d] = np.nan
    strHv[mask3d] = np.nan
    dudx[mask3d] = np.nan
    dvdy[mask3d] = np.nan
    advHv[mask3d] = np.nan
    dHdx[mask3d] = np.nan
    dHdy[mask3d] = np.nan

    print("converting dH/dt_div (rate) -> H_div (anom) ...")

    # dH/dt_div(t) -> H_div(t)
    dt = t - REF_TIME
    H_div = divHv[:, :, :] * dt[None, None, :]
    H_adv = advHv[:, :, :] * dt[None, None, :]
    H_str = strHv[:, :, :] * dt[None, None, :]

    # TODO: Change name of saved variables

    data = {
        "dHdt_div10": divHv,
        "dHdt_adv10": advHv,
        "dHdt_str10": strHv,
        "dHudx10": dHudx,
        "dHvdy10": dHvdy,
        "Hu10": Hu,
        "Hv10": Hv,
        "dudx10": dudx,
        "dvdy10": dvdy,
        "dHdx10": dHdx,
        "dHdy10": dHdy,
        "H_div10": H_div,
        "H_str10": H_str,
        "H_adv10": H_adv,
    }

    if SAVE:
        h5save(FILE_CUBE, data, "a")
        print("saved.")

    # --- For testing only --- #

    # Plot: Calculate dynamic correction and plot fields
    # NOTE: Change vmin/vmax for H or h

    if 1:

        # TODO: Change name of saved variables

        if not TEST:
            advHv, strHv, divHv = h5read(
                FILE_CUBE, ["dHdt_adv10", "dHdt_str10", "dHdt_div10"]
            )

        adv = advHv[:, :, 4]
        str = strHv[:, :, 4]
        div = divHv[:, :, 4]

        cmap = plt.cm.RdBu

        plt.matshow(adv, cmap=cmap, vmin=-15, vmax=15)
        plt.title("dH/dt from Advection (m/yr)")
        plt.colorbar()

        plt.matshow(str, cmap=cmap, vmin=-15, vmax=15)
        plt.title("dH/dt from Stretching (m/yr)")
        plt.colorbar()

        plt.matshow(div, cmap=cmap, vmin=-15, vmax=15)
        plt.title("dH/dt from Divergence (m/y)")
        plt.colorbar()

        plt.matshow(mask)

        plt.show()


if __name__ == "__main__":
    main()

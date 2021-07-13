"""
Calculate (smooth) time-variable Freeboard, Draft and Thickness.

Input:
    <- Height change time series (3D)
    <- Mean height field (2D) [from satellite or DEM]
    <- FAC time series (3D)
    <- MSL field (2D)  # TODO: In future 3D
    <- SLT field (2D)  [m/yr -> m]

Output:
    -> Height time series (3D)
    -> Freeboard time series (3D)
    -> Draft time series (3D)
    -> Thickness time series (3D)
    (all corrected for FAC and SLT)

Notes:
    - It references all time series to the REF_TIME of the Mean height/DEM
    - All fields must be on the same grid (i.e. previously re-gridded)
    - Edit header for constant parameters.
    - Smooth FAC further to account for RA penetration?

"""
# import sys
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    read_h5,
    save_h5,
    sgolay1d,
    find_nearest,
    test_ij_3km,
)

warnings.filterwarnings("ignore")

# === Edit ======================================================

REF_TIME = 2014.0  # ref time of Mean height/DEM (see TIMESPANS.txt)

FILE_CUBE = "/Users/paolofer/work/melt/data/FULL_CUBE_v3.h5"
FILE_OUT = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

x_var = "x"
y_var = "y"
t_var = "t"
h_var = "h_mean_cs2"
dh_var = "dh_xcal_filt"
fac_var = "fac_gemb8"
msl_var = "msl"
slt_var = "slt"

SMOOTH_WINDOW = 3

SAVE = False

PLOT = True

# ===============================================================


# NOTE: Not being used at the moment
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
        metavar=("FILE_OUT"),
        dest="FILE_OUT",
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


def smooth_series(y, window=SMOOTH_WINDOW, order=1):
    y_ = np.full_like(y, np.nan)
    (idx,) = np.where(~np.isnan(y))

    if len(idx) < window:
        return y

    # NOTE: For deriv=0 no need dt
    # NOTE: Always check the mode (default 'nearest')!
    y_[idx] = sgolay1d(y[idx], window, order, 0)

    return y_


def main():
    # TODO: COMBINE CUBEDIV.PY AND CUBEDEM.PY?
    """
    1. Reference all time series
    2. Correc dh for dFAC
    3. Correct dh for SLT
    4. Compute h(t) time series: h_mean + dh(t)
    5. Compute freeboard: H_freeb = h(t) - MSL
    6. Compute thickness and draft

    """
    print("loading ...")

    x, y, t, dh, h_mean, fac, msl, slt = read_h5(
        FILE_CUBE,
        [x_var, y_var, t_var, dh_var, h_var, fac_var, msl_var, slt_var],
    )

    # TODO: Maybe do this from the beguining (in cubefilt2.py)?
    # Mask out constant values (pole hole)
    dhdt = np.apply_along_axis(np.gradient, 2, dh)

    dh[dhdt == 0] = np.nan

    # Generate time series of sea-level trend (2D -> 3D)
    slt = slt[:, :, None] * (t - REF_TIME)

    # --- Smooth and reference series --- #

    if SMOOTH_WINDOW != 0:
        print("smoothing ...")

        dh = np.apply_along_axis(smooth_series, 2, dh)
        fac = np.apply_along_axis(smooth_series, 2, fac)

    print("referencing ...")

    # Correct mean height CS2 for FAC (before referencing)
    k_ref, = find_nearest(t, REF_TIME)
    h_mean = h_mean - fac[:, :, k_ref]

    # Reference all time series to a common epoch
    dh = np.apply_along_axis(lambda y: y - y[k_ref], 2, dh)
    fac = np.apply_along_axis(lambda y: y - y[k_ref], 2, fac)

    if PLOT:

        i_, j_ = test_ij_3km["PEAK_2"]

        plt.figure()
        plt.plot(t, dh[i_, j_, :], label="dh")
        plt.plot(t, fac[i_, j_, :], label="fac")
        plt.plot(t, slt[i_, j_, :], label="slt")
        plt.legend()
        plt.show()

        plt.pcolormesh(fac[:, :, 10], cmap='RdBu', rasterized=True)
        plt.plot([j_], [i_], 'or')
        plt.show()

    # Correct dh(t)
    dh_cor = dh - fac - slt

    # Compute time-evolving DEM (and correct for FAC)
    h = h_mean[:, :, None] + dh_cor

    if PLOT:

        plt.figure()
        plt.plot(t, dh[i_, j_, :], label="dh")
        plt.plot(t, dh_cor[i_, j_, :], label="dh_cor")
        plt.legend()

        plt.figure()
        plt.plot(t, h[i_, j_, :], label="h_mean + dh_cor 1")
        plt.legend()
        plt.show()

    # h(t) -> Freeboard, Draft, Thickness
    rho_ocean = 1028.0
    rho_ice = 917.0

    H_freeb = h - msl[:, :, None]
    H_draft = H_freeb * ((rho_ocean / (rho_ocean - rho_ice)) - 1)
    H = H_freeb * rho_ocean / (rho_ocean - rho_ice)

    # if PLOT:
    #     plt.figure()
    #     plt.plot(t, H_freeb[i_, j_, :] / 1000.0)
    #     plt.plot(t, -H_draft[i_, j_, :] / 1000.0)

    if SAVE:
        data = {"h10": h, "H_freeb10": H_freeb, "H_draft10": H_draft, "H10": H}
        save_h5(FILE_OUT, data, 'a')
        print("saved.")


if __name__ == "__main__":
    main()

"""
Calculate melt-rate and net-mass time series.

Input:
    <- Thickness change time series (3D) [corrected for FAC]
    <- Advection & Divergence time series (3D) [corrected for FAC]
    <- Acummulation rate time series (3D)

Output:
    -> Melt rate time series (3D) -> b(t) [m/yr]
    -> Net mass rate time series (3D) -> m(t) [m/yr]
    -> Mean melt rate map (2D) -> mean[b(t)] [m/yr]

Recipe:
    1. Fill in incomplete time series (e.g. polehole, GMs)
    2. Filter out temporal outliers and interpolate
    3. Compute derivative of H(t) -> dH/dt(t)
    4. Convert accumulation (units and buoyacy)  # NOTE: Maybe do this outside?
    5. Filter out spatial outliers and interpolate
    6. Add b(t) = dH/dt(t) + adv(t) + div(t) - accum(t) [m of i.eq./yr]
    7. Convert thickness rate time series to mass:
        m(t) = RHO_ICE * dH/dt(t) [Gt/yr]

Notes:
    m/yr melt grid -> mean Gt/yr number:
        sum(m/yr) x 3000 x 3000 x 917 x 1e-12

"""
import sys

import numpy as np
import matplotlib.pyplot as plt

# TODO: Move the filtering to cubefilt3.py? <<<<<<<<<<<<<<<<<<<<<<<< YES

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans

from utils import (
    read_h5,
    save_h5,
    sgolay1d,
    mad_std,
    transform_coord,
    test_ij_3km,
)

# --- EDIT ----------------------------------------

# TODO: If you change var names, change output vars at the end.

FCUBE = "/Users/paolofer/work/melt/data/FULL_CUBE_v4.h5"

xvar = "x"
yvar = "y"
tvar = "t"
hvar = "H10"
avar = "dHdt_adv10"  # advection [m/yr]
svar = "dHdt_str10"  # stretching [m/yr]
dvar = "dHdt_div10"  # divergence [m/yr]
mvar = "smb_gemb8"   # accumulation [m/yr] # NOTE: Didn't get updated to v9,v10
kvar = "mask_floating"

WINDOW = 5  # 3 is equal to np.gradient  # NOTE: Changed to 5 for v10

RHO_ICE = 917.0

SAVE = True

PLOT = False

# -------------------------------------------------


def filter_gradient(t, h_, n_std=3):
    """Filter outliers by evaluating the derivative.

    Take derivative and evaluate outliers in derivative.
    """
    h = h_.copy()

    # NOTE: This must be a separate step
    # dh/dt = 0 -> invalid
    dhdt = np.gradient(h)
    invalid = np.round(dhdt, 6) == 0.0
    dhdt[invalid] = np.nan

    invalid = np.isnan(dhdt) | (
        np.abs(dhdt - np.nanmedian(dhdt)) > mad_std(dhdt) * n_std
    )
    if sum(invalid) == 0:
        return h

    h[invalid] = np.nan

    return h


def filter_deviations(t, h_, n_std=3):
    """Filter deviations from the median and from the gradient."""
    h = h_.copy()
    dhdt = np.gradient(h)
    outlier = (np.abs(h - np.nanmedian(h)) > mad_std(h) * n_std) | (
        np.abs(dhdt - np.nanmedian(dhdt)) > mad_std(dhdt) * n_std
    )
    h[outlier] = np.nan

    return h


def filter_end_points(t, h_, n_std=3):
    """Filter end points by evaluating against the derivative.

    Take derivative and evaluate outliers in derivative,
    """
    h = h_.copy()

    # NOTE: This must be a separate step
    # dh/dt = 0 -> invalid
    dhdt = np.gradient(h)
    invalid = np.round(dhdt, 6) == 0.0
    dhdt[invalid] = np.nan

    invalid = np.abs(dhdt - np.nanmedian(dhdt)) > mad_std(dhdt) * n_std

    if sum(invalid) == 0:
        return h

    # Leave only the 3 end points on each side
    invalid[3:-3] = False
    h[invalid] = np.nan

    return h


def filter_spatial_outliers(hh_, n_std=5):
    hh = hh_.copy()
    i, j = np.where((np.abs(hh) - np.nanmedian(hh)) > mad_std(hh) * n_std)
    hh[i, j] = np.nan

    return hh


# NOTE: NOT BEING USED IN THIS CODE
# NOTE: Ad-hoc function for the 3-km grid
def filter_velocity_boundary(
    x, y, melt_=None, blon=186, blat=-82.68, dlon=13, dlat=0.06, return_index=False,
):
    """Filter artifacts at the junction between velocity fields (Alex/Eric).

    params for 3km grid: blon=186, blat=-82.68, dlon=13, dlat=0.06
    """
    melt = melt_.copy()
    X, Y = np.meshgrid(x, y)
    lon, lat = transform_coord(3031, 4326, X, Y)
    lon[lon < 0] += 360
    ii, jj = np.where(
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


# ------------------------------------------------

print("loading data ...")
x, y, t, H, adv, str, div, smb, mask = read_h5(
    FCUBE, [xvar, yvar, tvar, hvar, avar, svar, dvar, mvar, kvar]
)

# Output containers
dHdt = np.full_like(H, np.nan)
dt = t[1] - t[0]

count = 0
count_plot = 0
n_plots = 10  # number of plots to exit

for i in range(H.shape[0]):
    for j in range(H.shape[1]):

        # Only for ploting purposes
        if PLOT:
            i, j = list(test_ij_3km.values())[count]
            count += 1
            print("grid cell: %d/%d" % (count, H.shape[0] * H.shape[1]))

        H_ij = H[i, j, :]
        adv_ij = adv[i, j, :]
        str_ij = str[i, j, :]
        div_ij = div[i, j, :]
        smb_ij = smb[i, j, :]

        if sum(~np.isnan(H_ij)) < 10:
            continue

        if PLOT:
            H_orig = H_ij.copy()
            div_orig = div_ij.copy()

        # TODO: Maybe move the filtering below to cubefilt3.py? <<<<<<<<<<<<<<<<<<< YES

        # --- Begin filtering --- #

        # Use higher threshold for end points
        H_ij = filter_end_points(t, H_ij, n_std=5)

        # Use lower trheshold for inner points
        H_ij = filter_gradient(t, H_ij, n_std=5)

        invalid = np.isnan(H_ij)

        H_ij[invalid] = np.interp(t[invalid], t[~invalid], H_ij[~invalid])

        # FIXME: Revisit the filtering on the Divergence
        # Filter divergence fiedls
        div_ij = filter_end_points(
            t, div_ij, n_std=5
        )  # FIXME: Test, not in the original
        div_ij = filter_gradient(t, div_ij, n_std=5)  # FIXME: Test, not in the original
        div_ij[invalid] = np.nan
        # div_ij = filter_deviations(t, div_ij, n_std=5)

        outlier = np.isnan(div_ij)

        if sum(outlier) > 0:
            div_ij[outlier] = np.interp(t[outlier], t[~outlier], div_ij[~outlier])
            adv_ij[outlier] = np.interp(t[outlier], t[~outlier], adv_ij[~outlier])
            str_ij[outlier] = np.interp(t[outlier], t[~outlier], str_ij[~outlier])

        # --- End filtering --- #

        # Take derivative (interpolating NaNs)
        dHdt_ij = sgolay1d(
            H_ij,
            window=WINDOW,
            order=1,
            deriv=1,
            dt=dt,
            time=t,
            mode="nearest",  # FIXME: Test, original had 'interp'
        )

        # Add mean to SMB for dh/dt = 0
        const = np.round(dHdt_ij, 6) == 0.0
        smb_ij[const] = np.nanmean(smb_ij[~const])

        if 1:
            # Smooth with same time window as dH/dt
            adv_ij = sgolay1d(adv_ij, window=WINDOW, order=1, deriv=0, time=t)
            str_ij = sgolay1d(str_ij, window=WINDOW, order=1, deriv=0, time=t)
            div_ij = sgolay1d(div_ij, window=WINDOW, order=1, deriv=0, time=t)
            smb_ij = sgolay1d(smb_ij, window=WINDOW, order=1, deriv=0, time=t)

        dHdt[i, j, :] = dHdt_ij

        # TODO: If moving filtering, to cubefilt3.py,  <<<<<<<<<<<<<<<<<<< YES
        # then move the cube updates as well.

        # Update cubes
        H[i, j, :] = H_ij
        adv[i, j, :] = adv_ij
        str[i, j, :] = str_ij
        div[i, j, :] = div_ij
        smb[i, j, :] = smb_ij

        # --- Plot --- #

        if PLOT:
            count_plot += 1

            b = dHdt[i, j, :] + div[i, j, :] - smb[i, j, :]

            dHdt_ = dHdt[i, j, :]
            adv_ = adv[i, j, :]
            str_ = str[i, j, :]
            div_ = div[i, j, :]
            smb_ = smb[i, j, :]

            plt.figure(figsize=(6, 10))
            plt.subplot(511)
            plt.plot(t, H_orig, "r")
            plt.plot(t, H_ij)
            plt.ylabel("H (m)")

            plt.subplot(512)
            plt.plot(t, dHdt_)
            plt.ylabel("dHdt (m/y)")

            plt.subplot(513)
            plt.plot(t, div_orig, "r")
            plt.plot(t, div_, label="div")
            plt.plot(t, adv_, label="adv")
            plt.plot(t, str_, label="str")
            plt.legend()
            plt.ylabel("Div (m/y)")

            plt.subplot(514)
            plt.plot(t, smb_)
            plt.ylabel("SMB (m/y)")

            ax = plt.subplot(515)
            plt.plot(t, b)
            plt.ylabel("Melt (m/y)")
            text = "mean melt rate %.2f m/y" % np.nanmean(b)
            plt.text(
                0.5, 0.9, text, ha="center", va="center", transform=ax.transAxes,
            )

            plt.show()

            if count_plot == n_plots:
                sys.exit()

# --- Spatial filtering --- #

if 1:
    kernel = Gaussian2DKernel(1)

    for k in range(H.shape[2]):
        print("interpolating slice:", k)

        dHdt[:, :, k] = interpolate_replace_nans(
            dHdt[:, :, k], kernel, boundary="extend"
        )
        H[:, :, k] = interpolate_replace_nans(H[:, :, k], kernel, boundary="extend")
        adv[:, :, k] = interpolate_replace_nans(adv[:, :, k], kernel, boundary="extend")
        str[:, :, k] = interpolate_replace_nans(str[:, :, k], kernel, boundary="extend")
        div[:, :, k] = interpolate_replace_nans(div[:, :, k], kernel, boundary="extend")
        smb[:, :, k] = interpolate_replace_nans(smb[:, :, k], kernel, boundary="extend")

print("applying mask ...")
mask3d = np.repeat(mask[:, :, np.newaxis], H.shape[2], axis=2)
dHdt[~mask3d] = np.nan
H[~mask3d] = np.nan
adv[~mask3d] = np.nan
str[~mask3d] = np.nan
div[~mask3d] = np.nan
smb[~mask3d] = np.nan

melt = dHdt + div - smb
melt_steady = div - smb
mass = dHdt * RHO_ICE * 1e-12  # kg/yr -> Gt/yr
melt_mean = np.nanmean(melt, axis=2)

# NOTE: Change name of variables for new version

data = {
    "dHdt_melt10": melt,
    "dHdt_steady10": melt_steady,
    "dMdt_net10": mass,
    "dHdt_net10": dHdt,
    "dHdt_melt_mean10": melt_mean,
    "H_filt10": H,
    "dHdt_adv_filt10": adv,
    "dHdt_str_filt10": str,
    "dHdt_div_filt10": div,
    "smb_gemb_filt10": smb,
}
if SAVE:
    save_h5(FCUBE, data, "a")
    print("saved.")


plt.matshow(melt_mean, cmap="RdBu", vmin=-5, vmax=5)
plt.show()

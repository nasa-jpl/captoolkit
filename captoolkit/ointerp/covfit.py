#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit analitical model to empirical covariances.

Example:
    python covfit.py ~/data/ers1/floating/filt_scat_det/*.cov?

Notes:
    This code is part of the following set:
    - covxx.py
    - covxy.py
    - covt.py
    - ointerp.py

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

# -----------------------------------------------------------

""" First guess for iterative fit. """

s = 1.0  # std [of full data] param  ##NOTE: for trend and accel
# s = 10.    # std [of full data] param  ##NOTE: for height
R = 1000.0  # corr length param

robust = False  # for robust nonlinear fit
median = True  # fit model to median values
window = 500  # window for running median
dx = 500  # steps for running median

""" Covariance models. """


def gauss(r, s, R):
    return s ** 2 * np.exp(-(r ** 2) / R ** 2)


def markov(r, s, R):
    return s ** 2 * (1 + r / R) * np.exp(-r / R)


def generic(r, s, R):
    return s ** 2 * (1 + (r / R) - 0.5 * (r / R) ** 2) * np.exp(-r / R)


def exp(r, s, R):
    return np.exp(-(r ** 2) / R ** 2)


# -----------------------------------------------------------


def get_args():
    """ Get command-line arguments. """
    des = "Optimal Interpolation of spatial data"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
        "ifiles",
        metavar="ifiles",
        type=str,
        nargs="+",
        help='name of ifile, numpy binary or ascii (for binary ".npy")',
    )
    parser.add_argument(
        "-o",
        metavar="ofile",
        dest="ofile",
        type=str,
        nargs=1,
        help='name of ofile, numpy binary or ascii (for binary ".npy")',
        default=[None],
    )
    return parser.parse_args()


def print_args(args):
    print("Input arguments:")
    for arg in vars(args).items():
        print(arg)


def mad_std(x, axis=None):
    """ Robust standard deviation (using MAD). """
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis)), axis)


def binning(
    x, y, xmin=None, xmax=None, dx=1 / 12.0, window=3 / 12.0, interp=False, median=False
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


# -----------------------------------------------------------

""" Helper functions. """


def zero_crossing(x, y):
    return x[np.gradient(np.sign(y)) != 0]


def robust_fit(func, x, y, p0=[1.0, 1.0]):
    """Robust nonlinear regression."""
    # Function to computing the residuals
    res = lambda p, x, y: func(x, *p) - y
    # Iterative robust minimization of residuals
    fit = least_squares(res, p0, args=(x, y), loss="soft_l1", f_scale=0.1)
    return fit.x, fit.fun  # return fitted params


def fit_model(func, x, y, p0=[1.0, 1.0], robust=True):
    if robust:
        return robust_fit(func, x, y, p0)[0]
    else:
        return curve_fit(func, x, y, p0)[0]


def remove_nans(x, y):
    (ii,) = np.where(~np.isnan(x) & ~np.isnan(y))
    return x[ii], y[ii]


def print_params(name, param):
    print('model, R, s = "%s", %.4f, %.4f' % (name, param[-1], param[0]))


# -----------------------------------------------------------

# Parser argument to variable
args = get_args()

# Read input from terminal
ifiles = args.ifiles[:]
ofile = args.ofile[0]

# Print parameters to screen
print_args(args)

print("reading input file ...")

lag, cov = [], []

# Read and stack all covariance files
for fi in ifiles:

    l, c = np.loadtxt(fi, unpack=True)

    # NOTE: Do not use cov at lag zero for the fit!
    l[0] = np.nan
    c[0] = np.nan

    # Append
    lag = np.hstack((lag, l))
    cov = np.hstack((cov, c))

# Running median
cov_bin = binning(lag, cov, dx=dx, window=window, median=True, interp=True)[1]

if median:
    # Fit to median-binned values
    cov = cov_bin
else:
    # Fit to orig values w/o outliers
    cov[np.abs(cov - cov_bin) > mad_std(cov - cov_bin) * 5] = np.nan

# Remove NaNs
lag, cov = remove_nans(lag, cov)

""" Fit covariance model. """

c = cov  # dependent var
r = lag  # independent var

rr = np.linspace(r.min(), r.max(), 500)

# Fit models to data
param_g = fit_model(gauss, r, c, p0=[s, R], robust=robust)
param_m = fit_model(markov, r, c, p0=[s, R], robust=robust)
param_y = fit_model(generic, r, c, p0=[s, R], robust=robust)
param_e = fit_model(exp, r, c, p0=[s, R], robust=robust)

g = gauss(rr, *param_g)
m = markov(rr, *param_m)
y = generic(rr, *param_y)
e = exp(rr, *param_e)

print_params("gauss", param_g)
print_params("markov", param_m)
print_params("generic", param_y)
print_params("exp", param_e)

""" Find characteristic scales. """

try:
    # First zero crossing (from generic cov model)
    zero_cross = zero_crossing(rr, v)[0]
except:
    zero_cross = np.nan

# Normalize cov fun -> corr fun
corr = (m - m.min()) / (m.max() - m.min())

# Decorrelation length
i = np.argmin(np.abs(corr - 0.5))
corr_length = rr[i]

# e-folding scale
j = np.argmin(np.abs(corr - 1 / np.e))
e_fold = rr[j]

""" Save parameters. """

##NOTE: For now just copy and paste from screen
if 0:
    with file(ofile, "w") as f:
        f.write(param)
        pass

""" Plot results. """

try:
    plt.figure()
    plt.plot(r, c, ".")
    plt.plot(rr, g, "-", linewidth=2.5, label="gauss")
    plt.plot(rr, m, "-", linewidth=2.5, label="markov")
    plt.plot(rr, y, "-", linewidth=2.5, label="generic")
    plt.title("Sample covariance and model fit")
    plt.xlabel("Lag")
    plt.ylabel("Covariance")
    plt.legend()

    plt.figure()
    plt.plot(rr, corr, "-", linewidth=3, label="corr func")
    plt.axvline(x=corr_length, color="b")
    plt.axvline(x=e_fold, color="g")
    plt.axvline(x=zero_cross, color="r")
    plt.title(
        "decorr = %.2f, e-fold = %.2f, zero-cross = %.2f"
        % (corr_length, e_fold, zero_cross)
    )
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()

    plt.show()
except:
    print("skipping plot...")

print("done.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit analitical model to empirical covariances.

Example:
    python covfit.py ~/data/ers1/floating/filt_scat_det/*.cov?

"""

import os
import sys
import h5py
import pyproj
import argparse
import numpy as np
from time import time as tim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares


def get_args():
    """ Get command-line arguments. """

    des = 'Optimal Interpolation of spatial data'
    parser = argparse.ArgumentParser(description=des)

    parser.add_argument(
            'ifiles', metavar='ifiles', type=str, nargs='+',
            help='name of i-file, numpy binary or ascii (for binary ".npy")')

    parser.add_argument(
            '-o', metavar='ofile', dest='ofile', type=str, nargs=1,
            help='name of o-file, numpy binary or ascii (for binary ".npy")',
            default=[None])

    return parser.parse_args()


def print_args(args):
    print 'Input arguments:'
    for arg in vars(args).iteritems():
        print arg

#----------------------------------------------------------

""" Covariance models. """

def gauss(r, s, R):
    return s**2 * np.exp(-r**2/R**2)

def markov(r, s, R):
    return s**2 * (1 + r/R) * np.exp(-r/R)

def generic(r, s, R):
    return s**2 * (1 + (r/R) - 0.5 * (r/R)**2) * np.exp(-r/R)

def exp(r, R):
     return np.exp(-r**2/R**2) 

#----------------------------------------------------------

""" Helper functions. """

def zero_crossing(x, y):
    return x[np.diff(np.sign(y)) != 0]


def model_fit(model, x, y, p0=[1.,1.]):

    # Function computing the residulas
    fun = lambda p, x, y: model(x, *p) - y

    # Iterative robust minimization of residuals
    fit = least_squares(fun, p0, args=(x,y), loss='soft_l1', f_scale=0.1)

    # Return fitted params
    return fit.x, fit.fun


def remove_nan(x, y):
    ii, = np.where(~np.isnan(x) & ~np.isnan(y))
    return x[ii], y[ii]


# Parser argument to variable
args = get_args() 

# Read input from terminal
ifiles = args.ifiles[:]
ofile = args.ofile[0]

# Print parameters to screen
print_args(args)

print "reading input file ..."

lag = []
cov = []

# Read and stack all covariance files
for fi in ifiles:

    l, c = np.loadtxt(fi, unpack=True)

    ##NOTE: Do not use cov at lag zero in the fit!
    l[0] = np.nan
    c[0] = np.nan

    # Append
    lag = np.hstack((lag, l))
    cov = np.hstack((cov, c))

    ##NOTE: Test averaging lag values before fit
    """
    # Create matrix
    lag = l
    try:
        cov = np.column_stack((cov, c))
    except:
        cov = c
    """

#cov = np.nanmean(cov, axis=1)

# Remove NaNs
lag, cov = remove_nan(lag, cov)

""" Fit covariance model. """

c = cov            # dependent var
r = lag            # independent var
s = 1.             # std param [full data] (first guess)
#R = 1000.         # corr length param (first guess)
R = 0.25           # corr length param (first guess)

rr = np.linspace(r.min(), r.max(), 500)

# Fit models to data
param, _ = curve_fit(gauss, r, c, p0=[s,R])
g = gauss(rr, *param)
print 'gauss', param

param, _ = curve_fit(markov, r, c, p0=[s,R])
m = markov(rr, *param)
print 'markov', param

param, _ = curve_fit(generic, r, c, p0=[s,R])
v = generic(rr, *param)
print 'generic', param

param, err = curve_fit(exp, r, c, p0=[R])
e = exp(rr, *param)
print 'exp', param

""" Find characteristic scales. """

# First zero crossing (from generic cov function)
zero_cross = zero_crossing(rr, v)[0]

# Normalize cov fun -> corr fun
corr = (m - m.min()) / (m.max() - m.min())

# Decorrelation length
i = np.argmin(np.abs(corr-0.5))
corr_length = rr[i]

# e-folding scale
j = np.argmin(np.abs(corr-1/np.e))
e_fold = rr[j]

""" Save parameters. """

if 0:
    with h5py.File(ofile, 'w') as f:
        pass

""" Plot results. """

plt.figure()
plt.plot(r, c, '.')
plt.plot(rr, g, '-', linewidth=2.5, label='gauss')
plt.plot(rr, m, '-', linewidth=2.5, label='markov')
plt.plot(rr, v, '-', linewidth=2.5, label='generic')
plt.title('Sample covariance and model fit')
plt.xlabel('Lag')
plt.ylabel('Covariance')
plt.legend()

plt.figure()
plt.plot(rr, corr, '-', linewidth=3, label='corr func')
plt.axvline(x=corr_length, color='b')
plt.axvline(x=e_fold, color='g')
plt.axvline(x=zero_cross, color='r')
plt.title('Correlation function, e-fold = %.2f, 0-cross = %.2f' \
        % (e_fold, zero_cross))
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.legend()

plt.show()

print 'done.'

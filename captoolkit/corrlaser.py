#!/usr/bin/env python
"""
Compute and apply corrections for ICESat Laser 2 and 3.

From Borsa et al. (2019):
    Subtract 1.7 cm from Laser 2 and add 1.1 cm to Laser 3

Notes:
    Edit some parameters bellow.

Example:
    python corrlaser.py /path/to/files/*.h5

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit

    Fernando Paolo (paolofer@jpl.nasa.gov)
    Johan Nilsson (johan.nilsson@jpl.nasa.gov)
    Alex Gardner (alex.s.gardner@jpl.nasa.gov)

    Jet Propulsion Laboratory, California Institute of Technology

"""

import os
import sys

import h5py
import numpy as np
from astropy.time import Time
from future import standard_library

standard_library.install_aliases()

# === EDIT HERE ====================================== #

# time variable
tvar = "t_year"

# height variable
hvar = "h_cor"

# apply or only store the correction for each height
apply_ = True

# number of jobs in parallel
njobs = 16

# === END EDIT ======================================= #

# Corrections for lasers/campaigns (in meters)
# Cor will be subtracted from height: h - l{1,2,3}
bias = {"l1": 0.0, "l2": 0.017, "l3": -0.011, None: 0.0}

# Definition of ICESat campaigns/lasers
# https://nsidc.org/data/icesat/laser_op_periods.html
campaigns = [
    (Time("2003-02-20").decimalyear, Time("2003-03-21").decimalyear, "l1a"),
    (Time("2003-03-21").decimalyear, Time("2003-03-29").decimalyear, "l1b"),
    (
        Time("2003-09-25").decimalyear,
        Time("2003-11-19").decimalyear,
        "l2a",
    ),  # 8-d + 91-d + 91-d (NSIDC)
    (Time("2004-02-17").decimalyear, Time("2004-03-21").decimalyear, "l2b"),
    (Time("2004-05-18").decimalyear, Time("2004-06-21").decimalyear, "l2c"),
    (Time("2004-10-03").decimalyear, Time("2004-11-08").decimalyear, "l3a"),
    (Time("2005-02-17").decimalyear, Time("2005-03-24").decimalyear, "l3b"),
    (Time("2005-05-20").decimalyear, Time("2005-06-23").decimalyear, "l3c"),
    (Time("2005-10-21").decimalyear, Time("2005-11-24").decimalyear, "l3d"),
    (Time("2006-02-22").decimalyear, Time("2006-03-28").decimalyear, "l3e"),
    (Time("2006-05-24").decimalyear, Time("2006-06-26").decimalyear, "l3f"),
    (Time("2006-10-25").decimalyear, Time("2006-11-27").decimalyear, "l3g"),
    (Time("2007-03-12").decimalyear, Time("2007-04-14").decimalyear, "l3h"),
    (Time("2007-10-02").decimalyear, Time("2007-11-05").decimalyear, "l3i"),
    (Time("2008-02-17").decimalyear, Time("2008-03-21").decimalyear, "l3j"),
    (Time("2008-10-04").decimalyear, Time("2008-10-19").decimalyear, "l3k"),
    (Time("2008-11-25").decimalyear, Time("2008-12-17").decimalyear, "l2d"),
    (Time("2009-03-09").decimalyear, Time("2009-04-11").decimalyear, "l2e"),
    (Time("2009-09-30").decimalyear, Time("2009-10-11").decimalyear, "l2f"),
]


def _get_laser_bias(time, campaigns, bias):
    """ Map time (yr) to campaign to correction. """
    camp = [ca for (t1, t2, ca) in campaigns if t1 <= time < t2]  # ['c']|[]
    laser = camp[0][:2] if camp else None

    return bias[laser]


get_laser_bias = np.vectorize(_get_laser_bias, excluded=[1, 2], cache=True)


def main(fname):
    with h5py.File(fname, "a") as f:
        t = f[tvar][:]
        b = get_laser_bias(t, campaigns, bias)

        f["laser_bias"] = b  # save bias

        if apply_:
            f[hvar][:] -= b  # remove bias
        f.flush()

    os.rename(fname, fname.replace(".h5", "_LCOR.h5"))


files = sys.argv[1:]
print("Number of files:", len(files))


if njobs == 1:
    print("running sequential code ...")
    [main(f) for f in files]
else:
    print("running parallel code (%d jobs) ..." % njobs)
    from joblib import Parallel, delayed

    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)

print("done.")

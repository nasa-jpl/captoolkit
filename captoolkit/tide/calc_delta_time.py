#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
calc_delta_time.py
Written by Tyler Sutterley (08/2020)
Calculates the difference between universal time and dynamical time (TT - UT1)
    following Richard Ray's PERTH3 algorithms

Input:
    delta_file from
        http://maia.usno.navy.mil/ser7/deltat.data
        ftp://cddis.nasa.gov/products/iers/deltat.data
    iMJD: Modified Julian Day of times to interpolate

Requires:
    numpy: Scientific Computing Tools For Python
        http://www.numpy.org
        http://www.scipy.org/NumPy_for_Matlab_Users
    scipy: Scientific Tools for Python
        http://www.scipy.org/

History:
    Updated 08/2020: using scipy interpolation to calculate delta time
    Updated 11/2019: pad input time dimension if entering a single value
    Updated 07/2018: linearly extrapolate if using dates beyond the table
    Written 07/2018
"""
import os
import numpy as np
import scipy.interpolate

# PURPOSE: calculate the Modified Julian Day (MJD) from calendar date
# http://scienceworld.wolfram.com/astronomy/JulianDate.html
def calc_modified_julian_day(YEAR, MONTH, DAY):
    MJD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 - 2400000.5
    return np.array(MJD,dtype=np.float)

# interpolate delta time
def calc_delta_time(delta_file,t):
    """
    Calculates the difference between universal time and dynamical time

    Arguments
    ---------
    delta_file: file containing the delta times
    t: input times to interpolate (days since 1992-01-01T00:00:00)

    Returns
    -------
    deltat: delta time at the input time
    """
    # convert time from days relative to Jan 1, 1992 to Modified Julian Days
    # change dimensions if entering a single value
    iMJD = np.atleast_1d(48622.0 + t)
    # read delta time file
    dinput = np.loadtxt(os.path.expanduser(delta_file))
    # calculate julian days and convert to MJD
    MJD = calc_modified_julian_day(dinput[:,0],dinput[:,1],dinput[:,2])
    # use scipy interpolating splines to interpolate delta times
    spl = scipy.interpolate.UnivariateSpline(MJD,dinput[:,3],k=1,s=0,ext=0)
    # return the delta time for the input date converted to days
    return spl(iMJD)/86400.0

"""
Routines for time conversion.

"""
import os
import h5py
import argparse
import numpy as np
import pandas as pd
import datetime as dt

# Default variable name of t in the HDF5 files
TVAR = 'time'

# Default reference epoch for converting seconds
EPOCH = (1985,1,1,0,0,0)

# Default njobs for sequential/parallel run
NJOBS = 1


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Convert time variable in HDF5 files.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='HDF5 file(s) to convert')

parser.add_argument(
        '-v', metavar='tvar',  type=str, nargs=1,
        default=TVAR,
        help=('name of time variable in HDF5 file (-v time)'))

parser.add_argument(
        '-n', metavar='njobs',  type=int, nargs=1,
        default=[NJOBS],
        help=('number of jobs for parallel processing (-n 1)'))

parser.add_argument(
        '-e', metavar='epoch', type=int, nargs=6,
        default=EPOCH,
        help=('reference epoch for converting seconds (-n Y M D h m d)'))

parser.add_argument(
        '-s', metavar='expr', type=str, nargs=1,
        default=None,
        help=('string-expression to convert time (-s "t/3600 + 2000")'))


# Global variables
args = parser.parse_args()
files = args.files
tvar = args.v[0]
njobs = args.n[0]
epoch = args.e
expr = args.s[0]


#TODO: Check all the functions below.

def epoch_to_datetime(epochs):
    """
    Convert epoch as (Y,M,D,h,m,s) to datetime object.
    """
    dtimes = [dt.datetime(Y, M, D, h, m, s)
              for Y, M, D, h, m, s in epochs.astype('i4')]
    return np.array(dtimes)


def datetime_to_datenum(dtime):
    """
    Return (fractional) serial date number from datetime object.
    """
    main = dtime + dt.timedelta(days=366)
    frac = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day, 0, 0, 0)
            ).seconds / (24.0 * 60.0 * 60.0)
    return main.toordinal() + frac


def change_epoch(time, epoch1, epoch2, units='s'):
    """Convert time on reference epoch1 to reference epoch2."""
    epoch1 = dt.datetime(*epoch1)
    epoch2 = dt.datetime(*epoch2)
    secs_btw_epochs = (epoch1 - epoch2).total_seconds()
    if units=='s':
        time += secs_btw_epochs  # convert to time since epoch2
    elif units=='m':
        time += secs_btw_epochs/60.
    elif units=='h':
        time += secs_btw_epochs/3600.
    elif units=='d':
        time += secs_btw_epochs/86400.
    else:
        print('wrong time unit, chose between: s|m|h|d')
    return time

#TODO: Check all the functions above.


#--- Functions below were checked -------------------------------


def secs_to_dtime(secs, epoch=(1985,1,1,0,0,0)):
    """
    Convert seconds since epoch to datetime object.
    """
    secs = [secs] if np.ndim(secs) == 0 else secs  # -> iterable
    return np.asarray([dt.datetime(*epoch) + dt.timedelta(seconds=s)
                       for s in secs])


def dtime_to_secs(dtimes, epoch=(1985,1,1,0,0,0)):
    """
    Convert datetime object to seconds since epoch.
    """
    dtimes = [dtimes] if np.ndim(dtimes) == 0 else dtimes  # -> iterable
    return np.array([(d - dt.datetime(*epoch)).total_seconds()
                     for d in dtimes])


def secs_to_hours(secs, epoch1=(1985,1,1,0,0,0), epoch2=None):
    """
    Convert seconds since epoch1 to hours since epoch2.

    If epoch2 is None, keeps epoch1 as the reference.
    
    """
    epoch1 = dt.datetime(*epoch1)
    epoch2 = dt.datetime(*epoch2) if epoch2 is not None else epoch1
    secs_btw_epochs = (epoch2 - epoch1).total_seconds()
    return (secs - secs_btw_epochs) / 3600.  # subtract time diff


def datenum(*date):
    """
    Convert date as Y,M,D[,H,M,S] to Matlab's serial date number.

    These are number of days from January 0, 0000.
    """
    dtime = dt.datetime(*date)
    mdn = dtime + dt.timedelta(days=366)
    d = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day, 0, 0, 0))
    frac = d.seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def secs_to_datenum(secs, epoch):
    """Convert seconds to serial date number (days since 1-Jan-0000)."""
    return datenum(*epoch) + (secs/86400.)


#TODO: secs_to_year()


#--- Main --------------------------------------------------

#import numexpr

def main(infile):

    print('converting file:', infile, '...')
    print('evaluating expression:', expr.replace('t', tvar), '...')

    with h5py.File(infile, 'a') as f:

        t = f[tvar][:]

        if expr:
            t = eval(expr)
            #time = numexpr.evaluate(expr)
        else:
            # use a specified time conversion function
            pass

        f[tvar][:] = t
        f.flush()

    print('done.')


if njobs == 1:
    # Sequential code
    print('Running sequential code...')
    [main(f) for f in files]
else:
    # Parallel code
    print('Running parallel code (%d jobs)...' % njobs)
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(main)(f) for f in files)


#--- For testing -------------------------------------------

if 0:
    fname = sys.argv[1:]

    data = np.loadtxt('lat_lon_time')
    y, x, epochs = data[:,0], data[:,1], data[:,2:]

    dtimes = epoch_to_datetime(epochs)
    secs = datetime_to_seconds(dtimes, since_epoch=(1985,1,1))

if 0:
    dn = datenum(1953, 1, 9)
    dn2 = datetime_to_datenum(dt.datetime(1953, 1, 9))

    print(dn)
    print(dn2)

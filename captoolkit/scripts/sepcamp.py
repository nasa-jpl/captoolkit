#!/usr/bin/env python
"""
Identify the campaign that each file belongs to.

"""
import os
import sys
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta
from joblib import Parallel, delayed

#files = sys.argv[1:]
files = glob('GLAH12_634_*.h5')
print(('Number of files:', len(files)))

# Definition of ICESat campaigns
campaigns = {
    #(datetime(2003,  2, 20), datetime(2003,  3, 21)): 'l1a',
    #(datetime(2003,  3, 21), datetime(2003,  3, 29)): 'l1b',
    (datetime(2003,  2, 20), datetime(2003,  3, 29)): 'l1',
    (datetime(2003,  9, 25), datetime(2003, 11, 19)): 'l2a',   # 8-d + 91-d + 91-d (NSIDC)
    (datetime(2004,  2, 17), datetime(2004,  3, 21)): 'l2b',
    (datetime(2004,  5, 18), datetime(2004,  6, 21)): 'l2c',
    (datetime(2004, 10,  3), datetime(2004, 11,  8)): 'l3a',
    (datetime(2005,  2, 17), datetime(2005,  3, 24)): 'l3b',
    (datetime(2005,  5, 20), datetime(2005,  6, 23)): 'l3c',
    (datetime(2005, 10, 21), datetime(2005, 11, 24)): 'l3d',
    (datetime(2006,  2, 22), datetime(2006,  3, 28)): 'l3e',
    (datetime(2006,  5, 24), datetime(2006,  6, 26)): 'l3f',
    (datetime(2006, 10, 25), datetime(2006, 11, 27)): 'l3g',
    (datetime(2007,  3, 12), datetime(2007,  4, 14)): 'l3h',
    (datetime(2007, 10,  2), datetime(2007, 11,  5)): 'l3i',
    (datetime(2008,  2, 17), datetime(2008,  3, 21)): 'l3j',
    (datetime(2008, 10,  4), datetime(2008, 10, 19)): 'l3k',
    (datetime(2008, 11, 25), datetime(2008, 12, 17)): 'l2d',
    (datetime(2009,  3,  9), datetime(2009,  4, 11)): 'l2e',
    (datetime(2009,  9, 30), datetime(2009, 10, 11)): 'l2f',
}


def sec2date(secs, epoch=datetime(1970, 1, 1, 0, 0, 0)):
    """ Seconds since epoch -> datetime object. """
    return epoch + timedelta(seconds=secs)


def get_campaign(date, camp=campaigns):
    d = {date: v for k, v in list(camp.items()) \
            if k[0] <= date and date <= k[1]}
    if not d:
        # use only Y:M:D -> date()
        d = {date: v for k, v in list(camp.items()) \
                if k[0].date() <= date.date() and date.date() <= k[1].date()}
    return d[date]  # d -> {meandate: campaign}


def main(fname):
    """Identify file's campaign and tag file name."""
    with h5py.File(fname) as f:
        meandate = sec2date(np.nanmean(f['time'][:]))
        campaign = get_campaign(meandate)
    os.rename(fname, fname.replace('.h5', '_'+campaign+'.h5'))


'''
for f in files:
    main(f)
'''

Parallel(n_jobs=16, verbose=10)(
    delayed(main)(f) for f in files)


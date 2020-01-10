"""
zvar = 'h_res'          # dh w.r.t. topo + slp cor + bs cor
zvar = 'h_cor'          # h + slp cor 
zvar = 'h_cor_orig'     # h
zvar = 'bs'
zvar = 'lew'
zvar = 'tes'

NOTE: All heights will be corrected for bs in the xover code
"""

import os
import sys
from glob import glob

#--- Edit --------------------

sat = 'envisat'

folder = 'vostok'
#folder = 'thwaites_box'

inkey = 'VOSTOK'
#inkey = 'AnIS_BOX'

tspan = '2003 2009'

outkey = 'ICE_RA2'

subs = ['unc', 'det', 'dif']

#zvars = ['h_res', 'h_cor', 'h_cor_orig']
zvars = ['h_cor']

#-----------------------------

def xover(sub):

    # INPUT
    f1a = os.path.join('~/data/icesat', folder, 'ICE_'+inkey+'_HEIGHTS_2003_2009_A_BINARY_ORBIT_TIME.h5')
    f2d = glob(os.path.join('/Users/paolofer/data', sat, folder, sub, '*'+inkey+'*'+'_D_'+'*'))[0]

    f1d = os.path.join('~/data/icesat', folder, 'ICE_'+inkey+'_HEIGHTS_2003_2009_D_BINARY_ORBIT_TIME.h5')
    f2a = glob(os.path.join('/Users/paolofer/data', sat, folder, sub, '*'+inkey+'*'+'_A_'+'*'))[0]

    for zvar in zvars:

        # OUTPUT
        f3ad = os.path.join('~/data/xover', folder, sub, outkey+'_AD_'+zvar+'_BINARY_ORBIT_TIME.h5')
        f3da = os.path.join('~/data/xover', folder, sub, outkey+'_DA_'+zvar+'_BINARY_ORBIT_TIME.h5')

        cmd = 'python xover3.py %s %s -o %s -v orbit lon lat t_year %s -t %s -d 50 -p 3031 -k 10'

        os.system(cmd % (f1a, f2d, f3ad, zvar, tspan))
        os.system(cmd % (f1d, f2a, f3da, zvar, tspan))


# Run in parallel 
from joblib import Parallel, delayed
Parallel(n_jobs=3, verbose=5)(delayed(xover)(sub) for sub in subs)

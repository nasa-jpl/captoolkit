"""
zvar = 'h_res'          # dh w.r.t. topo + slp cor + bs cor
zvar = 'h_cor'          # h + slp cor 
zvar = 'h_cor_orig'     # h

NOTE: All heights will be corrected for bs in the xover code
"""

import os
import sys
from glob import glob


def get_files(files):
    ff = []
    for f in files:
        ff += glob(f)
    return ff


#--- Edit --------------------

folder = 'vostok'
#folder = 'thwaites_box'

key = 'ICE_RA2'

#zvars = ['h_res', 'h_cor', 'h_cor_orig']
zvars = ['h_cor']

# Pairs of files
files = [
        glob('/Users/paolofer/data/ers1/thwaites_box/det/*_?_RM_params.h5'),
        glob('/Users/paolofer/data/ers1/thwaites_box/dif/*_?_RM_params.h5'),
        glob('/Users/paolofer/data/ers2/thwaites_box/det/*_?_RM_params.h5'),
        glob('/Users/paolofer/data/ers2/thwaites_box/dif/*_?_RM_params.h5'),
        glob('/Users/paolofer/data/envisat/thwaites_box/det/*_2002_2010_*_RM_params.h5'),
        glob('/Users/paolofer/data/envisat/thwaites_box/dif/*_2002_2010_*_RM_params.h5'),
        glob('/Users/paolofer/data/cryosat2/thwaites_box/det/*_LRM_*_params.h5'),
        glob('/Users/paolofer/data/cryosat2/thwaites_box/dif/*_LRM_*_params.h5'),
        glob('/Users/paolofer/data/cryosat2/thwaites_box/det/*_SIN_*_params.h5'),
        glob('/Users/paolofer/data/cryosat2/thwaites_box/dif/*_SIN_*_params.h5'),
]

#-----------------------------

def main(ff):

    f1, f2 = ff
    f3 = f1.replace('_A_', '_AD_') 
    #f3 = os.path.join('~/data/xover', folder, sub, key+'_ALL_'+zvar+'.h5')
    
    os.system('python dummy.py -v ad -l 1 -f ' + f1)
    os.system('python dummy.py -v ad -l 0 -f ' + f2)
    os.system('python merge.py %s %s -o %s ' % (f1, f2, f3))


# Run in parallel 
from joblib import Parallel, delayed
Parallel(n_jobs=8, verbose=5)(delayed(main)(ff) for ff in files)

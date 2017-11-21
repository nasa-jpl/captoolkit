### VOSTOK

## unc ad
F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_A.h5
F2=~/data/envisat/vostok/unc/RA2_VOSTOK_HEIGHTS_2002_2010_D_RM.h5
F3=~/data/xover/vostok/unc/ICE_RA2_AD_ELEV_NOSLOPE.h5
python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10

# unc da
F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_D.h5
F2=~/data/envisat/vostok/unc/RA2_VOSTOK_HEIGHTS_2002_2010_A_RM.h5
F3=~/data/xover/vostok/unc/ICE_RA2_DA_ELEV_NOSLOPE.h5
python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10

### det ad
#F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_A.h5
#F2=~/data/envisat/vostok/det/RA2_VOSTOK_HEIGHTS_2002_2010_D_RM.h5
#F3=~/data/xover/vostok/det/ICE_RA2_AD_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## det da
#F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_D.h5
#F2=~/data/envisat/vostok/det/RA2_VOSTOK_HEIGHTS_2002_2010_A_RM.h5
#F3=~/data/xover/vostok/det/ICE_RA2_DA_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
### dif ad
#F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_A.h5
#F2=~/data/envisat/vostok/dif/RA2_VOSTOK_HEIGHTS_2002_2010_D_RM.h5
#F3=~/data/xover/vostok/dif/ICE_RA2_AD_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## dif da
#F1=~/data/icesat/vostok/ICE_VOSTOK_HEIGHTS_2003_2009_D.h5
#F2=~/data/envisat/vostok/dif/RA2_VOSTOK_HEIGHTS_2002_2010_A_RM.h5
#F3=~/data/xover/vostok/dif/ICE_RA2_DA_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10


### THWAITES

## unc ad
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_A.h5
#F2=~/data/envisat/thwaites_box/unc/RA2_AnIS_BOX_HEIGHTS_2002_2010_D_RM.h5
#F3=~/data/xover/thwaites_box/unc/ICE_RA2_AD_ELEV_NOSLOPE.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## unc da
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_D.h5
#F2=~/data/envisat/thwaites_box/unc/RA2_AnIS_BOX_HEIGHTS_2002_2010_A_RM.h5
#F3=~/data/xover/thwaites_box/unc/ICE_RA2_DA_ELEV_NOSLOPE.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10

## det ad
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_A.h5
#F2=~/data/envisat/thwaites_box/det/RA2_AnIS_BOX_HEIGHTS_2002_2010_D_RM.h5
#F3=~/data/xover/thwaites_box/det/ICE_RA2_AD_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## det da
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_D.h5
#F2=~/data/envisat/thwaites_box/det/RA2_AnIS_BOX_HEIGHTS_2002_2010_A_RM.h5
#F3=~/data/xover/thwaites_box/det/ICE_RA2_DA_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## dif ad
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_A.h5
#F2=~/data/envisat/thwaites_box/dif/RA2_AnIS_BOX_HEIGHTS_2002_2010_D_RM.h5
#F3=~/data/xover/thwaites_box/dif/ICE_RA2_AD_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10
#
## dif da
#F1=~/data/icesat/thwaites_box/ICE_BOX_HEIGHTS_2003_2009_D.h5
#F2=~/data/envisat/thwaites_box/dif/RA2_AnIS_BOX_HEIGHTS_2002_2010_A_RM.h5
#F3=~/data/xover/thwaites_box/dif/ICE_RA2_DA_ELEV.h5
#python xover3.py $F1 $F2 -o $F3 -v orbit lon lat t_year h_res -t 2003 2009 -d 50 -p 3031 -k 10

# MERGING INDIVIDUAL SMALL FILES AFTER READING/SLOPE CORRECTION
#
#python merge.py -o /mnt/devon-r0/shared_data/icesat/floating_/ANT_ICE_ISHELF_READ_A.h5 '/mnt/devon-r0/shared_data/icesat/floating_/READ/*_A.H5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/icesat/floating_/ANT_ICE_ISHELF_READ_D.h5 '/mnt/devon-r0/shared_data/icesat/floating_/READ/*_D.H5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers1/floating_/ANT_ER1_ISHELF_READ_A_RM.h5 '/mnt/devon-r0/shared_data/ers/floating_/READ/*_E1_*_A_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers1/floating_/ANT_ER1_ISHELF_READ_D_RM.h5 '/mnt/devon-r0/shared_data/ers/floating_/READ/*_E1_*_D_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers2/floating_/ANT_ER2_ISHELF_READ_A_RM.h5 '/mnt/devon-r0/shared_data/ers/floating_/READ/*_E2_*_A_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers2/floating_/ANT_ER2_ISHELF_READ_D_RM.h5 '/mnt/devon-r0/shared_data/ers/floating_/READ/*_E2_*_D_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2002_2010_READ_A_RM.h5 '/mnt/devon-r0/shared_data/envisat/floating_/READ_2002_2010/*_A_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2002_2010_READ_D_RM.h5 '/mnt/devon-r0/shared_data/envisat/floating_/READ_2002_2010/*_D_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2010_2012_READ_A_RM.h5 '/mnt/devon-r0/shared_data/envisat/floating_/READ_2010_2012/*_A_RM.h5' -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2010_2012_READ_D_RM.h5 '/mnt/devon-r0/shared_data/envisat/floating_/READ_2010_2012/*_D_RM.h5' -z lzf &
python merge.py -o /mnt/devon-r0/shared_data/cryosat2/floating_/ANT_CS2_ISHELF_READ_A.h5 '/mnt/devon-r0/shared_data/cryosat2/floating/*_A.h5' -z lzf &
python merge.py -o /mnt/devon-r0/shared_data/cryosat2/floating_/ANT_CS2_ISHELF_READ_D.h5 '/mnt/devon-r0/shared_data/cryosat2/floating/*_D.h5' -z lzf &
#
# MERGING SPLITTED FILES AFTER TIDE CORRECTION
#
#python merge.py -o /mnt/devon-r0/shared_data/icesat/floating_/ANT_ICE_ISHELF_READ_A_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/icesat/floating_/*_A_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/icesat/floating_/ANT_ICE_ISHELF_READ_D_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/icesat/floating_/*_D_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers1/floating_/ANT_ER1_ISHELF_READ_A_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/ers1/floating_/*_A_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers1/floating_/ANT_ER1_ISHELF_READ_D_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/ers1/floating_/*_D_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers2/floating_/ANT_ER2_ISHELF_READ_A_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/ers2/floating_/*_A_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/ers2/floating_/ANT_ER2_ISHELF_READ_D_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/ers2/floating_/*_D_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2002_2010_READ_A_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/envisat/floating_/*2002_2010*_A_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2002_2010_READ_D_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/envisat/floating_/*2002_2010*_D_*_TIDE.h5' -k IBE -z lzf &
#python merge.py -o /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2010_2012_READ_A_RM_TOPO_IBE_TIDE.h5 '/mnt/devon-r0/shared_data/envisat/floating_/*2010_2012*_A_*_TIDE.h5' -k IBE -z lzf &
